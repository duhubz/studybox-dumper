"""Lead-in detection, byte framing, and packet parsing.

Layer above :mod:`studybox.clock`. A page begins with a long lead-in run of
*0* data bits that terminates in a single *1* bit; the data region then repeats
``0`` marker bit + 8 data bits MSB-first. The resulting byte stream is parsed
with the C5-anchored state machine defined by the StudyBox packet grammar:

* page header    ``C5 01 01 01 01 PP PP CS``   (fixed 8 bytes)
* segment header ``C5 TT TT X Y CS``            (``TT`` < 6)
* data packet    ``C5 LL D0..D(L-1) CS``        (``LL`` >= 1)
* control packet ``C5 00 BODY CS``              (``BODY`` & 0xF0 == 0xF0 ends page)
* padding/type 5: bytes are skipped until the next ``C5``

Every checksum is ``XOR(all preceding packet bytes incl. the leading C5)``, so
a valid packet XORs to zero over all of its bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import audio_in, clock, frontend

PAGE_MAGIC = 0xC5
MIN_LEADIN_BITS = 256
LEADIN_MIN_RUN = 64
GAP_HALF_CELLS = clock.GAP_HALF_CELLS


def _xor(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
    return value


@dataclass
class Packet:
    kind: str            # page_header | segment_header | data | control
    offset: int
    size: int
    checksum_ok: bool
    segment_type: int | None = None
    page_id: int | None = None
    checked: bool = True  # type-5 padding has no checksum byte
    structural_ok: bool = True  # set False when the packet shape is invalid
    body: bytes = b""


@dataclass
class DecodedPage:
    page_id: int | None
    terminator_bit: int
    lead_in_sample: int
    audio_sample: int
    packets: list[Packet] = field(default_factory=list)
    byte_length: int = 0
    desync: int = 0
    raw: bytes = b""
    observed_coverage: bool = True
    complete: bool = True
    measured_half: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    # Sub-sample timing diagnostics (only filled when ``retain_timing`` is set):
    # the region-relative half-index of each page transition and its fractional
    # grid residual (measured interval in half-cells minus its nearest integer).
    timing_half: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    timing_residual: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    @property
    def checksum_ok(self) -> int:
        return sum(1 for p in self.packets if p.checked and p.checksum_ok)

    @property
    def checksum_total(self) -> int:
        return sum(1 for p in self.packets if p.checked)

    @property
    def data128(self) -> list[Packet]:
        return [p for p in self.packets if p.kind == "data" and p.size == 131]


@dataclass
class DecodeResult:
    capture: dict
    sample_rate: int
    half_cell: float
    transitions: int
    pages: list[DecodedPage]
    malformed: int
    unrecoverable: list[dict] = field(default_factory=list)

    @property
    def checksum_ok(self) -> int:
        return sum(page.checksum_ok for page in self.pages)

    @property
    def checksum_total(self) -> int:
        return sum(page.checksum_total for page in self.pages)

    @property
    def checksum_rate(self) -> float:
        return self.checksum_ok / self.checksum_total if self.checksum_total else 0.0

    @property
    def data128_ok(self) -> int:
        return sum(1 for page in self.pages for p in page.data128 if p.checksum_ok)

    @property
    def data128_total(self) -> int:
        return sum(len(page.data128) for page in self.pages)


# --------------------------------------------------------------------------- #
# Bit reconstruction
# --------------------------------------------------------------------------- #

@dataclass
class _Region:
    bits: np.ndarray
    samples: np.ndarray  # absolute sample position of each cell k
    half: np.ndarray     # measured transition half-index per transition (base-relative)
    observed: int        # cells fully covered by observed transitions


def region_bits(half_index: np.ndarray, transition_samples: np.ndarray,
                base: int | None = None) -> _Region:
    """Build the MFM data-bit array and a per-cell sample map for one gap-free run.

    ``half_index`` is cumulative but may start at any parity. The caller should
    pass ``base`` = the half-index of the first lead-in transition so lead-in
    transitions sit on even indices; then cell ``k`` is the pair ``(2k, 2k+1)``
    and a data bit is a transition at ``2k+1``.
    """
    half = half_index - (half_index[0] if base is None else base)
    size = int(half.max() // 2) + 2
    bits = np.zeros(size, dtype=np.uint8)
    samples = np.full(size, np.nan, dtype=np.float64)

    for h, sample in zip(half, transition_samples):
        h = int(h)
        if h < 0:
            # A transition before the chosen lead-in base (e.g. residue from the
            # previous page when the longest lead-in run does not start the
            # segment) is not part of this page.
            continue
        if h >= 1 and h % 2 == 1:
            cell = (h - 1) // 2
            bits[cell] = 1
            samples[cell] = sample
        else:
            cell = h // 2
            if np.isnan(samples[cell]):
                samples[cell] = sample

    known = np.nonzero(~np.isnan(samples))[0]
    if known.size and known.size < samples.size:
        missing = np.nonzero(np.isnan(samples))[0]
        samples[missing] = np.interp(missing, known, samples[known])
    # The cell containing the last observed transition is trusted to the end of
    # its data half (a lone clock edge means the data bit is zero in valid MFM).
    # The final zero-filled allocation beyond that horizon is not signal and
    # must not be framed into bytes.
    horizon = int(half.max()) if half.size else -1
    observed = horizon // 2 + 1 if horizon >= 0 else 0
    return _Region(bits=bits, samples=samples, half=half, observed=observed)


def find_page_terminators(bits: np.ndarray, min_leadin: int = MIN_LEADIN_BITS
                          ) -> list[tuple[int, int]]:
    """Return ``(bit_index, zero_run)`` for each ``1`` after a long zero run."""
    result: list[tuple[int, int]] = []
    run = 0
    for i, bit in enumerate(bits):
        if bit:
            if run >= min_leadin:
                result.append((i, run))
            run = 0
        else:
            run += 1
    return result


def frame_bytes(bits: np.ndarray, start: int,
                end: int | None = None) -> list[tuple[int, int]]:
    """Frame ``(byte, bit_index)`` pairs from ``start`` using 0+8 MSB framing.

    Framing stops at the first non-zero marker bit, when fewer than nine bits
    remain before ``end`` (default: the end of ``bits``), or at ``end`` itself.

    A run of zero *bytes* in a payload is valid data (CHR/nametable pages are
    full of them) and is never treated as a lead-in. The caller bounds a page
    with the next true lead-in terminator, or the region end when there is none.
    """
    out: list[tuple[int, int]] = []
    limit = bits.size if end is None else min(end, bits.size)
    i = start
    while i + 9 <= limit:
        if bits[i] != 0:
            break
        value = 0
        for j in range(1, 9):
            value = (value << 1) | int(bits[i + j])
        out.append((value, i))
        i += 9
    return out


# --------------------------------------------------------------------------- #
# Packet parsing
# --------------------------------------------------------------------------- #

@dataclass
class ParseStatus:
    """How packet parsing for one page ended."""

    termination: str   # end_control | packet_boundary | truncated
    complete: bool     # reached a valid end control
    consumed: int      # byte index just past the last complete packet


def parse_page(data: bytes | bytearray | memoryview,
               start: int = 0, end: int | None = None
               ) -> tuple[list[Packet], int, ParseStatus]:
    """Parse the StudyBox packet grammar from a byte buffer.

    ``data`` is indexed in place (no per-byte Python list is built) and may be a
    ``bytes``/``bytearray``/``memoryview``. ``start``/``end`` bound the page
    within ``data``; packet offsets in the result are relative to ``start``.

    Returns ``(packets, desync, status)``. ``status.complete`` is true only when
    the page ended at a valid end control; anything else is a truncated page and
    must not be treated as intact.
    """
    n = len(data) if end is None else min(end, len(data))
    packets: list[Packet] = []
    desync = 0
    i = start
    termination = "none"

    if start + 8 <= n and data[start] == PAGE_MAGIC:
        head = bytes(data[start:start + 8])
        ok = (head[1:5] == bytes([1, 1, 1, 1]) and head[5] == head[6]
              and _xor(head[:7]) == head[7])
        packets.append(Packet("page_header", 0, 8, ok, page_id=head[5], body=head))
        i = start + 8
    elif i < n:
        desync += 1

    mode = "segment"
    while i < n:
        if mode == "segment":
            if data[i] != PAGE_MAGIC:
                i += 1
                desync += 1
                continue
            if i + 3 > n:
                termination = "truncated"
                break
            seg_type = data[i + 1]
            if seg_type == 5:
                # Padding: the parser enters resync mode as soon as it reads the
                # repeated type byte, so there is no checksum; packet bytes are
                # skipped until the next C5 starts a data packet.
                ok = data[i + 1] == data[i + 2]
                packets.append(Packet("segment_header", i - start, 3, ok,
                                      segment_type=5, checked=False,
                                      structural_ok=ok,
                                      body=bytes(data[i:i + 3])))
                i += 3
                while i < n and data[i] != PAGE_MAGIC:
                    i += 1
                mode = "data"
                continue
            if i + 6 > n:
                termination = "truncated"
                break
            seg = bytes(data[i:i + 6])
            if not (seg[1] == seg[2] and seg_type < 6):
                # Malformed segment header: resync on the next C5.
                i += 1
                desync += 1
                continue
            ok = _xor(seg[:5]) == seg[5]
            packets.append(Packet("segment_header", i - start, 6, ok,
                                  segment_type=seg_type, body=seg))
            i += 6
            mode = "data"
        else:
            if data[i] != PAGE_MAGIC:
                i += 1
                desync += 1
                continue
            if i + 2 > n:
                termination = "truncated"
                break
            length = data[i + 1]
            if length == 0:
                if i + 4 > n:
                    termination = "truncated"
                    break
                ctl = bytes(data[i:i + 4])
                ok = _xor(ctl[:3]) == ctl[3]
                packets.append(Packet("control", i - start, 4, ok, body=ctl))
                i += 4
                mode = "end" if (ctl[2] & 0xF0) == 0xF0 else "segment"
                if mode == "end":
                    # An end-shaped control with a bad checksum is retained for
                    # diagnostics but does not terminate the page cleanly: the
                    # bytes after it are untrusted, so completeness needs the
                    # checksum to pass too (#5).
                    termination = "end_control" if ok else "bad_end_control"
                    break
            else:
                pkt_end = i + 2 + length + 1
                if pkt_end > n:
                    termination = "truncated"
                    break
                pkt = bytes(data[i:pkt_end])
                ok = _xor(pkt[:2 + length]) == pkt[2 + length]
                packets.append(Packet("data", i - start, 2 + length + 1, ok,
                                      body=pkt[2:2 + length]))
                i = pkt_end
    if termination == "none":
        termination = "packet_boundary"
    return packets, desync, ParseStatus(termination, termination == "end_control",
                                        i - start)


# --------------------------------------------------------------------------- #
# Capture decode
# --------------------------------------------------------------------------- #

def extract_transitions(capture: audio_in.Capture,
                        window_seconds: float = 30.0,
                        overlap_seconds: float = 0.2,
                        discard_seconds: float = 0.06,
                        max_seconds: float | None = None,
                        **frontend_kwargs) -> tuple[np.ndarray, int]:
    """Stream the data channel and return absolute transition samples.

    ``discard_seconds`` is applied only at *artificial* window boundaries (where
    the neighbouring overlap can recover the interval). The physical start and
    end of the recording are preserved so a page at the end of the tape is not
    silently truncated by an edge-trim that no overlap can recover.
    """
    info = audio_in.sf.info(str(capture.data_path))
    sample_rate = int(info.samplerate)
    total_frames = int(info.frames)
    window = int(window_seconds * sample_rate)
    overlap = int(overlap_seconds * sample_rate)
    discard = int(discard_seconds * sample_rate)
    if max_seconds is None:
        max_frames = None
    else:
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be a finite positive number")
        max_frames = int(max_seconds * sample_rate)
    collected: list[np.ndarray] = []
    for start, block in audio_in.iter_windows(
            capture.data_path, capture.data_channel, window, overlap):
        if max_frames is not None and start >= max_frames:
            break
        transitions = frontend.detect_transitions(block, sample_rate, **frontend_kwargs)
        local = transitions.samples
        block_end = start + len(block)
        left_trim = 0 if start <= 0 else discard
        right_trim = 0 if block_end >= total_frames else discard
        keep = (local >= left_trim) & (local <= len(block) - right_trim)
        absolute = local[keep] + start
        if max_frames is not None:
            absolute = absolute[absolute < max_frames]
        collected.append(absolute)
    if not collected:
        return np.empty(0, dtype=np.float64), sample_rate
    samples = np.unique(np.round(np.concatenate(collected), 3))
    return samples, sample_rate


def decode_transitions(transitions: np.ndarray, sample_rate: float,
                       min_leadin_bits: int = MIN_LEADIN_BITS,
                       capture: dict | None = None,
                       retain_timing: bool = False) -> DecodeResult:
    """Decode an in-memory transition sequence into page records."""
    summary = capture or {}
    if transitions.size < 4:
        return DecodeResult(summary, sample_rate, 0.0, int(transitions.size), [], 0,
                            [{"reason": "too few transitions",
                              "transitions": int(transitions.size)}])

    grid = clock.recover_grid(transitions, sample_rate)
    # Only internal, over-long gaps split the transition stream into regions.
    # Keeping this list explicit lets each gap be classified against the pages
    # its left/right regions produced (a gap is benign only when both sides
    # decoded as pages) instead of treating every gap as expected structure.
    internal_breaks = [int(b) for b in grid.breaks
                       if 0 <= b < transitions.size - 1
                       and grid.units[b] > clock.MAX_HALF_CELLS]
    boundaries = [0, *[b + 1 for b in internal_breaks], int(transitions.size)]

    pages: list[DecodedPage] = []
    malformed = 0
    unrecoverable: list[dict] = []
    # Per-region outcome used to classify the gaps between regions: a region is
    # a "page" when it produced one, "undecodable" when it carries a lead-in run
    # but produced none, and "noise" otherwise (background tone has no lead-in).
    region_status: list[str] = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        produced = 0
        if hi - lo < 4:
            region_status.append("noise")
            continue
        half = grid.half_index[lo:hi]
        samples = transitions[lo:hi]
        units = grid.units[lo:hi - 1]
        # The page lead-in is a long run of 2-cell clock gaps, but a long zero
        # run inside a payload looks identical; pick the first candidate run
        # (earliest in the region) whose following bytes frame to a C5 page
        # header. This also fixes the half-cell parity, which the old longest-run
        # heuristic could take from a payload run.
        runs = sorted(clock.leadin_runs(units, min_length=LEADIN_MIN_RUN),
                      key=lambda run: run[0])
        base_candidates: list[int] = []
        for start, _ in runs:
            base = int(half[start])
            if base not in base_candidates:
                base_candidates.append(base)
        if not base_candidates:
            base_candidates = [int(half[0])]
        region: _Region | None = None
        starts: list[tuple[int, int]] = []
        for base in base_candidates:
            candidate = region_bits(half, samples, base)
            candidate_starts: list[tuple[int, int]] = []
            for terminator, zero_run in find_page_terminators(candidate.bits,
                                                              min_leadin_bits):
                probe = frame_bytes(candidate.bits, terminator + 1, candidate.observed)
                if len(probe) >= 8 and [v for v, _ in probe[:5]] == [PAGE_MAGIC, 1, 1, 1, 1]:
                    candidate_starts.append((terminator, zero_run))
            if candidate_starts:
                region = candidate
                starts = candidate_starts
                break
        if region is None:
            # A region that carries a long lead-in run but produced no page is
            # damaged signal, not background tone. Record it so a vanished page
            # cannot disappear while verification still reports a clean pass.
            if runs:
                unrecoverable.append({
                    "kind": "undecodable_region",
                    "sample": int(samples[0]) if samples.size else 0,
                    "transitions": int(hi - lo),
                })
                region_status.append("undecodable")
            else:
                region_status.append("noise")
            continue
        # Source-bit spans covered by packets that passed their own validation
        # (structure and, when present, checksum). Unchecked type-5 padding
        # counts only when its structure is intact. These spans exempt payload
        # zero runs from the discarded-page scan below; parsed-but-unvalidated
        # bytes never do.
        validated: list[tuple[int, int]] = []
        recognized_padding: list[tuple[int, int]] = []
        decoded_terminators: set[int] = set()
        # Known deferred false rejection: an exact header-shaped payload can
        # enter starts and truncate its containing packet before validation.
        # The reproduced case fails verification; resolving it needs provisional
        # boundary selection, not a broader discarded-candidate exemption.
        for index, (terminator, zero_run) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else region.bits.size
            end = min(end, region.observed)
            framed = frame_bytes(region.bits, terminator + 1, end)
            if len(framed) < 8:
                continue
            if [v for v, _ in framed[:5]] != [PAGE_MAGIC, 1, 1, 1, 1]:
                continue
            packets, desync, parse_status = parse_page(
                bytes(value for value, _ in framed))
            if not packets or packets[0].kind != "page_header":
                continue
            lead_cell = max(0, terminator - zero_run)
            malformed += desync
            # If framing stopped before the observed horizon and the page never
            # reached a valid end control, the tape ended mid-byte: the missing
            # bits are unrecoverable and must not be silently invented as zeros.
            observed_coverage = True
            if framed[-1][1] + 9 < end and not parse_status.complete:
                observed_coverage = False
                unrecoverable.append({
                    "kind": "unobserved_eof",
                    "page_id": packets[0].page_id,
                    "byte_offset": len(framed),
                    "sample": int(region.samples[max(0, min(
                        len(region.samples) - 1, end - 1))]),
                })
            # The frame loop stops early only on a non-zero marker bit, which is
            # a mid-page loss of byte alignment (dropout/glitch); record it.
            marker = framed[-1][1] + 9
            if marker + 9 <= end and int(region.bits[marker]) != 0:
                unrecoverable.append({
                    "kind": "mid_page_desync",
                    "page_id": packets[0].page_id,
                    "byte_offset": len(framed),
                    "sample": int(region.samples[min(len(region.samples) - 1,
                                                      marker // 2)]),
                })
            # Measured source transitions for this page, relative to the
            # terminator cell: the encoder round-trip compares against these.
            # Cells are terminator + 9 per byte (marker + 8 data bits), so the
            # last covered half-index is 2*(terminator + 9*len(framed)) + 1.
            span_end = 2 * (terminator + 9 * len(framed)) + 1
            mask = (region.half >= 2 * terminator) & (region.half <= span_end)
            measured = (region.half[mask] - 2 * terminator).astype(np.int64)
            timing_half = np.empty(0, dtype=np.int64)
            timing_residual = np.empty(0, dtype=np.float64)
            if retain_timing:
                selected = region.half[mask]
                selected_samples = samples[mask]
                if selected.size >= 2:
                    intervals = np.diff(selected_samples).astype(np.float64)
                    # grid.half_cell is in seconds; work in samples and let a
                    # page-local least-squares half-cell absorb speed drift so
                    # the residual reflects per-interval jitter, not scale.
                    h0 = grid.half_cell * sample_rate
                    counts = np.round(intervals / h0)
                    counts[counts < 1] = 1
                    h_local = intervals.sum() / counts.sum() if counts.sum() else h0
                    timing_half = selected[1:].astype(np.int64)
                    timing_residual = intervals / h_local - np.round(intervals / h_local)
            pages.append(DecodedPage(
                page_id=packets[0].page_id,
                terminator_bit=terminator,
                lead_in_sample=int(region.samples[lead_cell]),
                audio_sample=int(region.samples[terminator]),
                packets=packets,
                byte_length=len(framed),
                desync=desync,
                raw=bytes(value for value, _ in framed),
                observed_coverage=observed_coverage,
                complete=parse_status.complete,
                measured_half=measured,
                timing_half=timing_half,
                timing_residual=timing_residual,
            ))
            for packet in packets:
                if packet.structural_ok and (not packet.checked or packet.checksum_ok):
                    validated.append((
                        terminator + 1 + 9 * packet.offset,
                        terminator + 1 + 9 * (packet.offset + packet.size),
                    ))
            # Padding has no checksum: keep this separate from validated spans.
            # Only a complete, clean containing page earns the exemption. Bound
            # each skip at the very next C5, exactly as parse_page does, rather
            # than at a later successfully parsed packet after resynchronizing.
            if (parse_status.complete and observed_coverage and desync == 0
                    and all(p.structural_ok and (not p.checked or p.checksum_ok)
                            for p in packets)):
                raw = pages[-1].raw
                for packet in packets:
                    if packet.kind == "segment_header" and packet.segment_type == 5:
                        begin = packet.offset + packet.size
                        stop = raw.find(bytes([PAGE_MAGIC]), begin)
                        if stop >= begin:
                            recognized_padding.append((
                                terminator + 1 + 9 * begin,
                                terminator + 1 + 9 * stop,
                            ))
            decoded_terminators.add(terminator)
            produced += 1
        if produced == 0:
            # A region that carries a lead-in run but yielded no page is damaged
            # signal whether or not a base produced candidate starts; unify the
            # two paths so it is always diagnosed.
            unrecoverable.append({
                "kind": "undecodable_region",
                "sample": int(samples[0]) if samples.size else 0,
                "transitions": int(hi - lo),
            })
            region_status.append("undecodable")
            continue
        # Re-scan every lead-in candidate for page-like signal that was not
        # decoded as a page. In a gap-free region a damaged page can otherwise
        # be absorbed as the previous page's unvalidated tail, so retained-page
        # exactness alone does not prove capture coverage. Candidates inside a
        # validated packet span are payload zero runs, not pages.
        for terminator, _zero_run in find_page_terminators(region.bits,
                                                           min_leadin_bits):
            if terminator in decoded_terminators:
                continue
            if any(start <= terminator < end for start, end in validated):
                continue
            probe = frame_bytes(region.bits, terminator + 1, region.observed)
            marker_mismatches = 0
            if len(probe) < 8:
                # Diagnostic slots only: do not feed these values into parsing
                # or page output. A bad marker can stop framing even though the
                # header's data bits survive. Inspect all observed header slots,
                # including markers on the duplicated ID/checksum bytes.
                count = min(8, (region.observed - terminator - 1) // 9)
                if count < 5:
                    continue
                probe = []
                for slot in range(count):
                    pos = terminator + 1 + 9 * slot
                    marker_mismatches += int(region.bits[pos] != 0)
                    value = 0
                    for bit in region.bits[pos + 1:pos + 9]:
                        value = (value << 1) | int(bit)
                    probe.append((value, pos))
                if not marker_mismatches:
                    continue
            # Budget at most two wrong fixed values (C5 01 01 01 01) plus
            # nonzero observed header markers. Three or more damage indicators
            # are not detected (and are never guessed at or repaired).
            values = [value for value, _ in probe[:5]]
            mismatches = sum(1 for value, expected in
                             zip(values, [PAGE_MAGIC, 1, 1, 1, 1])
                             if value != expected)
            if mismatches + marker_mismatches > 2:
                continue
            header = bytes(value for value, _ in probe[:8])
            if any(start <= terminator and terminator + 1 + 9 * len(header) <= end
                   for start, end in recognized_padding):
                continue
            loss: dict = {
                "kind": "discarded_page",
                "sample": int(region.samples[terminator]),
                "transitions": int(hi - lo),
                "header": header.hex(),
                "magic_mismatches": mismatches,
                "marker_mismatches": marker_mismatches,
            }
            if len(header) >= 7 and header[5] == header[6]:
                loss["page_id"] = header[5]
            unrecoverable.append(loss)
        region_status.append("page")

    # Classify each gap against the regions on either side. A gap is a loss only
    # when it borders a lead-in-bearing region that failed to decode (a vanished
    # page); gaps inside background noise or between decoded pages are expected
    # tape structure. This keeps the real-tape noise gaps benign while a damaged
    # page still cannot disappear behind a clean-looking gap.
    gaps: list[dict] = []
    for gap_index, break_index in enumerate(internal_breaks):
        left = region_status[gap_index]
        right = region_status[gap_index + 1]
        gaps.append({
            "kind": "gap_loss" if "undecodable" in (left, right) else "gap",
            "sample": int(transitions[break_index]),
            "gap_half_cells": int(grid.units[break_index]),
            "left_page": left == "page",
            "right_page": right == "page",
        })

    return DecodeResult(summary, sample_rate, grid.half_cell, int(transitions.size),
                        pages, malformed, gaps + unrecoverable)


def decode_capture(capture: audio_in.Capture, min_leadin_bits: int = MIN_LEADIN_BITS,
                   window_seconds: float = 30.0,
                   max_seconds: float | None = None) -> DecodeResult:
    """Decode one side of a tape into page records with checksum statistics."""
    transitions, sample_rate = extract_transitions(
        capture, window_seconds=window_seconds, max_seconds=max_seconds)
    return decode_transitions(transitions, sample_rate, min_leadin_bits,
                              capture=capture.describe())
