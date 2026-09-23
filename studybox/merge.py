"""Recover dropout pages from a second recording of the same tape program.

Both sides of a StudyBox cassette carry the same digital pages, so a page cut
by a dropout on one side can be completed from the other. This module compares
checksummed packet payloads only (tail padding after the last packet cannot
carry data and is ignored) and substitutes a whole donor page only when the
evidence is unambiguous:

* every payload matches (the longest raw bytes are kept),
* the base payload is a strict prefix of a longer, complete, parser-clean,
  structurally valid, checksum-clean candidate (a dropout),
* exactly one complete, parser-clean, structurally valid variant passed every
  checked packet checksum,
* a clean truncated fragment shares almost all of a clean complete payload.

Irreducible disagreements keep the base page and are reported as open; bytes
are never spliced between sources and no bytes are invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__, container, framing


@dataclass
class MergeOutcome:
    """A merged container plus the per-page source provenance."""

    box: container.StudyBox
    provenance: dict[str, Any]


@dataclass
class _Candidate:
    source: str
    data: bytes
    stats: dict[str, Any]


def _stats(data: bytes) -> dict[str, Any]:
    packets, desync, status = framing.parse_page(data)
    checked = [packet for packet in packets if packet.checked]
    span = max((packet.offset + packet.size for packet in packets), default=0)
    structural_errors = sum(1 for p in packets if not p.structural_ok)
    return {
        "payload": data[:span] if span else data,
        "desync": desync,
        "complete": status.complete,
        "checked": len(checked),
        "ok": sum(1 for packet in checked if packet.checksum_ok),
        "structural_errors": structural_errors,
        "span": span,
    }


def _fully_valid(stats: dict[str, Any]) -> bool:
    return (stats["desync"] == 0 and stats["checked"] > 0
            and stats["ok"] == stats["checked"]
            and stats["complete"]
            and stats["structural_errors"] == 0)


def _variants(candidates: list[_Candidate], payloads: list[bytes]) -> list[dict]:
    out = []
    for payload in payloads:
        matching = [c for c in candidates if c.stats["payload"] == payload]
        rep = max(matching, key=lambda c: len(c.data))
        checksum_clean = (rep.stats["checked"] > 0
                          and rep.stats["ok"] == rep.stats["checked"])
        out.append({
            "payload_bytes": len(payload),
            "page_bytes": len(rep.data),
            "sources": sorted({c.source for c in matching}),
            "checksum_clean": checksum_clean,
            "parser_clean": rep.stats["desync"] == 0,
            "structurally_valid": rep.stats["structural_errors"] == 0,
            "complete": rep.stats["complete"],
            "fully_valid": _fully_valid(rep.stats),
        })
    return out


def _choose(candidates: list[_Candidate], prefer: str
            ) -> tuple[bytes, str, str, bool, list[dict]]:
    """Pick page bytes for one index.

    Returns ``(data, source, resolution, conflict, variants)``. ``candidates``
    is ordered with the preferred source first, so ties keep the base.
    """
    payloads = {c.stats["payload"] for c in candidates}

    def pick(payload: bytes) -> _Candidate:
        return max((c for c in candidates if c.stats["payload"] == payload),
                   key=lambda c: len(c.data))

    if len(payloads) == 1:
        rep = pick(next(iter(payloads)))
        identical = len({c.data for c in candidates}) == 1
        name = "identical" if identical else "payload_identical"
        return rep.data, rep.source, name, False, []

    # A strict-prefix payload is a dropout fragment of a lengthier one.
    keep = [p for p in payloads
            if not any(p != other and other.startswith(p) for other in payloads)]
    if not keep:
        keep = list(payloads)
    if len(keep) == 1:
        rep = pick(keep[0])
        if _fully_valid(rep.stats):
            return (rep.data, rep.source, "prefix_dropout: used complete payload",
                    False, _variants(candidates, keep))
        # A unique longer variant is not enough evidence if it is damaged.
        # Keep the preferred input and make the unresolved choice explicit.
        preferred = next((candidate for candidate in candidates
                          if candidate.source == prefer), candidates[0])
        variants = sorted(payloads, key=lambda payload: (len(payload), payload))
        return (preferred.data, preferred.source,
                "prefix_dropout: longer payload is invalid", True,
                _variants(candidates, variants))

    def rank(payload: bytes) -> tuple:
        rep = pick(payload)
        return (_fully_valid(rep.stats), rep.stats["ok"], len(payload))

    ranked = sorted(keep, key=rank, reverse=True)
    variants = _variants(candidates, ranked)
    clean = [p for p in ranked if _fully_valid(pick(p).stats)]

    # Exactly one variant passed every packet checksum: it is the repair.
    if len(clean) == 1:
        rep = pick(clean[0])
        return (rep.data, rep.source,
                "checksum_repair: single checksum-clean variant", False, variants)

    # A much shorter clean variant sharing almost all of its payload with a
    # longer clean one is a dropout that still checksummed its survivors.
    if len(clean) == 2:
        long_payload, short_payload = clean[0], clean[-1]
        if len(short_payload) <= 0.9 * len(long_payload):
            common = 0
            for left, right in zip(short_payload, long_payload):
                if left != right:
                    break
                common += 1
            if common >= 0.95 * len(short_payload):
                rep = pick(long_payload)
                return (rep.data, rep.source,
                        "length_truncation: used complete variant", False, variants)

    # Irreducible disagreement: preserve the preferred input and record the
    # conflict. Even if it is damaged, selecting among multiple clean donors
    # would be an unsupported guess (and tied candidates may have no natural
    # ordering).
    rep = next((candidate for candidate in candidates
                if candidate.source == prefer), candidates[0])
    both_valid = (_fully_valid(pick(ranked[0]).stats)
                  and _fully_valid(pick(ranked[1]).stats))
    same_length = len(ranked[0]) == len(ranked[1])
    kind = ("checksum_preserving_ambiguity" if (same_length and both_valid)
            else "length_divergence")
    all_variants = sorted(payloads, key=lambda payload: (len(payload), payload))
    return (rep.data, rep.source, kind, True,
            _variants(candidates, all_variants))


def merge_boxes(base: container.StudyBox, others: list[container.StudyBox],
                names: list[str]) -> MergeOutcome:
    """Merge ``others`` into ``base`` by page index.

    ``names`` labels every input; ``names[0]`` belongs to ``base``. All inputs
    must carry the same page count (recordings of the same program), so pages
    align by index. Page bytes may come from any input; the lead-in and audio
    offsets always come from the base, whose audio is embedded in the result.
    """
    if len(others) != len(names) - 1:
        raise ValueError("merge needs one name per input container")
    counts = [len(base.pages)] + [len(box.pages) for box in others]
    if len(set(counts)) != 1:
        raise ValueError(
            "page counts differ (" + ", ".join(
                f"{name}: {count}" for name, count in zip(names, counts))
            + "); inputs must be recordings of the same program")
    pages: list[container.Page] = []
    entries: list[dict] = []
    repaired = 0
    conflicts = 0
    for index, base_page in enumerate(base.pages):
        candidates = [_Candidate(names[0], base_page.data, _stats(base_page.data))]
        candidates.extend(
            _Candidate(names[offset], box.pages[index].data,
                       _stats(box.pages[index].data))
            for offset, box in enumerate(others, start=1))
        data, source, resolution, conflict, variants = _choose(candidates, names[0])
        if data != base_page.data:
            repaired += 1
        if conflict:
            conflicts += 1
        pages.append(container.Page(base_page.lead_in_offset,
                                    base_page.audio_offset, data))
        entry: dict[str, Any] = {
            "index": index,
            "page_id": data[5] if len(data) > 5 else None,
            "source": source,
            "resolution": resolution,
            "bytes": len(data),
        }
        if conflict:
            entry["open"] = True
            entry["variants"] = variants
        entries.append(entry)
    box = container.StudyBox(pages=pages, audio=base.audio,
                             file_type=base.file_type)
    provenance = {
        "schema_version": 1,
        "tool": "studybox.merge",
        "version": __version__,
        "base": names[0],
        "sources": names,
        "page_count": len(pages),
        "repaired": repaired,
        "open_conflicts": conflicts,
        "pages": entries,
    }
    return MergeOutcome(box=box, provenance=provenance)


def merge_files(base_path: str | Path, other_paths: list[str | Path],
                max_bytes: int = container.DEFAULT_MAX_CONTAINER_BYTES) -> MergeOutcome:
    """Load ``.studybox`` inputs and merge them (see :func:`merge_boxes`)."""
    base = container.StudyBox.read(base_path, max_bytes=max_bytes)
    others = [container.StudyBox.read(path, max_bytes=max_bytes)
              for path in other_paths]
    names: list[str] = []
    for index, path in enumerate([base_path, *other_paths]):
        name = Path(path).name
        if name in names:  # keep sources distinguishable in the provenance
            name = f"{name}#{index}"
        names.append(name)
    return merge_boxes(base, others, names)
