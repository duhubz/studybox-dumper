"""Independent decode gate.

Re-parses the stored page byte streams (not the decoder's packet objects) and
checks the structural invariants: STBX/page presence, C5 page
headers, page-id sanity, monotonic audio offsets, packet XOR-checksum pass rate,
and a dropout map. Checks are strict by default; explicitly requested degraded
verification can waive advisory dropout and full-checksum checks while retaining
the requested checksum-rate floor and structural checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import container, framing

DEFAULT_MIN_CHECKSUM_RATE = 1.0
# Guards applied before parsing untrusted page data (memory-amplification cap).
DEFAULT_MAX_PAGE_BYTES = 1 << 20        # 1 MiB per page
DEFAULT_MAX_TOTAL_BYTES = 256 << 20     # 256 MiB across all pages
# These checks may be waived by opt-in tolerance; when waived the result is
# reported as degraded rather than as a clean pass. The requested checksum-rate
# floor stays mandatory: tolerance cannot drop below ``min_checksum_rate``.
ADVISORY_CHECKS = frozenset({"no_dropouts", "all_checksums"})
# Inter-page gaps are expected tape structure, not in-page loss.
BENIGN_DROPOUT_KINDS = frozenset({"gap"})


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class VerifyReport:
    passed: bool
    checks: list[Check]
    page_count: int
    checksum_ok: int
    checksum_total: int
    checksum_rate: float
    dropout_map: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "tool": "studybox.verify",
            "passed": self.passed,
            "degraded": self.degraded,
            "page_count": self.page_count,
            "checksum_ok": self.checksum_ok,
            "checksum_total": self.checksum_total,
            "checksum_rate": self.checksum_rate,
            "dropout_map": self.dropout_map,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }

    def render(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if self.degraded:
            status += " (degraded)"
        lines = [f"verify: {status}"]
        for check in self.checks:
            lines.append(f"  [{'ok' if check.passed else 'XX'}] {check.name}: {check.detail}")
        if self.dropout_map:
            lines.append(f"  dropout map: {len(self.dropout_map)} entr(ies)")
        return "\n".join(lines)


def _page_header(page_data: bytes) -> bool:
    return (len(page_data) >= 8 and page_data[0] == framing.PAGE_MAGIC
            and page_data[1:5] == bytes([1, 1, 1, 1])
            and page_data[5] == page_data[6]
            and _xor(page_data[:7]) == page_data[7])


def _xor(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
    return value


def evaluate_pages(page_data: list[bytes],
                   min_checksum_rate: float = DEFAULT_MIN_CHECKSUM_RATE,
                   dropout_map: list[dict[str, Any]] | None = None,
                   offsets: list[tuple[int, int]] | None = None,
                   allow_degraded: bool = False,
                   max_page_bytes: int = DEFAULT_MAX_PAGE_BYTES,
                   max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES) -> VerifyReport:
    """Run the decode gate over raw page byte streams.

    Strict by default: any in-page dropout or failed checksum fails the gate.
    Passing ``allow_degraded`` (or a sub-1.0 ``min_checksum_rate``) turns the
    dropout/full-checksum checks into advisory checks, but the requested
    ``min_checksum_rate`` floor stays mandatory; a result that passes only under
    tolerance is reported as ``degraded``. Oversized inputs are rejected before
    parsing to bound memory use.
    """
    if not math.isfinite(min_checksum_rate) or not 0.0 <= min_checksum_rate <= 1.0:
        raise ValueError(
            f"min_checksum_rate must be a finite number in [0, 1], "
            f"got {min_checksum_rate!r}")
    total = sum(len(data) for data in page_data)
    if total > max_total_bytes:
        raise ValueError(
            f"page data is {total} bytes, over the {max_total_bytes}-byte limit")
    for index, data in enumerate(page_data):
        if len(data) > max_page_bytes:
            raise ValueError(
                f"page {index} is {len(data)} bytes, over the "
                f"{max_page_bytes}-byte per-page limit")
    checks: list[Check] = []
    checks.append(Check("pages_present", len(page_data) > 0, f"{len(page_data)} page(s)"))

    bad_headers = sum(1 for data in page_data if not _page_header(data))
    checks.append(Check("c5_page_headers", bad_headers == 0,
                        f"{bad_headers} malformed header(s) of {len(page_data)}"))

    desynced = 0
    checksum_ok = checksum_total = 0
    structural_errors = 0
    incomplete = 0
    failures: list[dict[str, Any]] = []
    for index, data in enumerate(page_data):
        packets, desync, status = framing.parse_page(data)
        if not status.complete:
            incomplete += 1
            failures.append({"page_index": index, "kind": "page",
                             "reason": status.termination})
        if packets:
            # Desync is only meaningful inside the parsed packet span: bytes
            # after the last packet are the page's unchecksummed tail.
            span = max(packet.offset + packet.size for packet in packets)
            _, desync, _ = framing.parse_page(data, 0, span)
        if desync:
            desynced += 1
        for packet in packets:
            if not packet.structural_ok:
                structural_errors += 1
                failures.append({"page_index": index, "kind": packet.kind,
                                 "offset": packet.offset, "size": packet.size,
                                 "reason": "malformed structure"})
                continue
            if not packet.checked:
                continue
            checksum_total += 1
            if packet.checksum_ok:
                checksum_ok += 1
            else:
                failures.append({"page_index": index, "kind": packet.kind,
                                 "offset": packet.offset, "size": packet.size})
    rate = checksum_ok / checksum_total if checksum_total else 0.0
    dropout_map = list(dropout_map or []) + failures
    losses = [item for item in dropout_map
              if item.get("kind") not in BENIGN_DROPOUT_KINDS]
    checks.append(Check("page_complete", incomplete == 0,
                        f"{incomplete} incomplete page(s)"))
    checks.append(Check("packet_parse_clean", desynced == 0,
                        f"{desynced} page(s) with desync"))
    checks.append(Check("packet_structure", structural_errors == 0,
                        f"{structural_errors} malformed packet(s)"))
    checks.append(Check("no_dropouts", len(losses) == 0,
                        f"{len(losses)} in-page loss entr(ies), "
                        f"{len(dropout_map)} dropout entr(ies) total"))
    checks.append(Check("all_checksums", checksum_ok == checksum_total,
                        f"{checksum_ok}/{checksum_total} checksums pass"))
    checks.append(Check("packet_checksum_rate", rate >= min_checksum_rate,
                        f"{checksum_ok}/{checksum_total} = {rate:.4f} "
                        f"(threshold {min_checksum_rate})"))

    if offsets is not None:
        # One offset pair per page, each non-negative and lead-in <= audio, in
        # tape order. ``zip`` alone accepts a short/empty list for a non-empty
        # page set and negative pairs, which cannot be serialized (unsigned).
        count_ok = len(offsets) == len(page_data)
        shaped = count_ok and all(
            len(data) >= 8 and 0 <= lead <= audio
            for data, (lead, audio) in zip(page_data, offsets))
        monotonic = count_ok and all(
            offsets[i][0] <= offsets[i + 1][0]
            and offsets[i][1] <= offsets[i + 1][1]
            for i in range(len(offsets) - 1))
        checks.append(Check("offsets_valid", shaped and monotonic,
                            f"{len(offsets)} offset pair(s) for "
                            f"{len(page_data)} page(s)"))

    # Strict by default: any dropout or failed checksum is fatal. Tolerance is
    # opt-in and, when it accepts a non-clean result, the report is labelled
    # degraded instead of a clean pass.
    tolerance = allow_degraded or min_checksum_rate < 1.0
    fatal = ([c for c in checks if c.name not in ADVISORY_CHECKS]
             if tolerance else checks)
    passed = all(check.passed for check in fatal)
    degraded = passed and any(not check.passed for check in checks)
    return VerifyReport(passed=passed, checks=checks, page_count=len(page_data),
                        checksum_ok=checksum_ok, checksum_total=checksum_total,
                        checksum_rate=rate, dropout_map=dropout_map,
                        degraded=degraded)


def verify_studybox(box: container.StudyBox,
                    min_checksum_rate: float = DEFAULT_MIN_CHECKSUM_RATE,
                    allow_degraded: bool = False) -> VerifyReport:
    """Verify a parsed ``.studybox`` container page by page.

    In addition to the page gate, checks the self-contained audio contract:
    an ``AUDI`` chunk with file type 0, a parseable WAV, and page offsets that
    fall within the embedded audio.
    """
    page_data = [page.data for page in box.pages]
    offsets = [(page.lead_in_offset, page.audio_offset) for page in box.pages]
    report = evaluate_pages(page_data, min_checksum_rate, offsets=offsets,
                            allow_degraded=allow_degraded)
    checks = list(report.checks)
    # Container-only checks are appended separately: only these (plus the page
    # report's own verdict) decide the container result. Re-running the page
    # checks here would re-apply advisory failures that tolerance waived, which
    # is what made container verification stricter than evaluate_pages (#4).
    container_checks: list[Check] = []
    frames = container.wav_frames(box.audio)
    container_checks.append(Check("audio_present", bool(box.audio),
                                  f"{len(box.audio)} audio byte(s)"))
    container_checks.append(Check("file_type_wav",
                                  box.file_type == container.FILE_TYPE_WAV,
                                  f"file type {box.file_type}"))
    container_checks.append(Check("audio_valid_wav", frames is not None,
                                  f"{frames} frame(s)" if frames is not None
                                  else "unreadable/absent WAV"))
    if frames is not None:
        within = all(page.lead_in_offset <= frames and page.audio_offset <= frames
                     for page in box.pages)
        container_checks.append(Check("offsets_within_audio", within,
                                      f"offsets vs {frames} frame(s)"))
    checks.extend(container_checks)
    passed = report.passed and all(check.passed for check in container_checks)
    degraded = passed and report.degraded
    return VerifyReport(passed=passed, checks=checks, page_count=report.page_count,
                        checksum_ok=report.checksum_ok,
                        checksum_total=report.checksum_total,
                        checksum_rate=report.checksum_rate,
                        dropout_map=report.dropout_map, degraded=degraded)


def verify_studybox_file(path: str | Path,
                         min_checksum_rate: float = DEFAULT_MIN_CHECKSUM_RATE,
                         allow_degraded: bool = False) -> VerifyReport:
    return verify_studybox(container.StudyBox.read(path), min_checksum_rate,
                           allow_degraded)


def verify_decode_result(result: framing.DecodeResult,
                         min_checksum_rate: float = DEFAULT_MIN_CHECKSUM_RATE,
                         allow_degraded: bool = False) -> VerifyReport:
    """Verify an in-memory decode by re-parsing its raw page bytes."""
    page_data = [page.raw for page in result.pages]
    return evaluate_pages(page_data, min_checksum_rate,
                          dropout_map=result.unrecoverable,
                          allow_degraded=allow_degraded)
