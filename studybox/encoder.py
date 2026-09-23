"""Encode decoded pages back into MFM for the bit-level round-trip gate.

The decoder recovers page bytes from the measured flux transitions; the encoder
turns those bytes into the MFM half-cell pattern a correct tape must contain. If
the re-encoded transitions do not exactly reproduce the measured source
transitions, the decode is not bit-exact and must not be claimed as such.

The same encoder renders a page to a bipolar MFM waveform, which enables
real-hardware replay and an audio-level round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import framing


def bytes_to_bits(raw: bytes) -> list[int]:
    """0 marker bit + 8 data bits MSB-first, per byte."""
    bits: list[int] = []
    for byte in raw:
        bits.append(0)
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def mfm_half_cells(data_bits: list[int]) -> np.ndarray:
    """Return the half-cell indices that carry a flux transition.

    Standard MFM: a transition in the data half (``2k+1``) for a 1 bit and a
    clock transition in the first half (``2k``) when a 0 bit follows a 0 bit.
    """
    edges: list[int] = []
    previous = 0
    for k, bit in enumerate(data_bits):
        if bit == 0 and previous == 0:
            edges.append(2 * k)
        if bit:
            edges.append(2 * k + 1)
        previous = bit
    return np.asarray(edges, dtype=np.int64)


def encode_page_edges(page: framing.DecodedPage) -> np.ndarray:
    """Half-cell edges (relative to the terminator cell) for a decoded page."""
    return mfm_half_cells([1] + bytes_to_bits(page.raw))


@dataclass
class RoundTrip:
    page_id: int | None
    expected: int
    measured: int
    missing: int
    extra: int
    exact: bool
    payload_exact: bool = True
    payload_data_exact: bool = True
    payload_expected: int = 0
    payload_measured: int = 0
    payload_missing: int = 0
    payload_extra: int = 0
    clock_glitches: int = 0
    tail_bytes: int = 0
    tail_dropout: bool = False

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "expected_edges": self.expected,
            "measured_edges": self.measured,
            "missing": self.missing,
            "extra": self.extra,
            "exact": self.exact,
            "payload_exact": self.payload_exact,
            "payload_data_exact": self.payload_data_exact,
            "payload_expected_edges": self.payload_expected,
            "payload_measured_edges": self.payload_measured,
            "payload_missing": self.payload_missing,
            "payload_extra": self.payload_extra,
            "clock_glitches": self.clock_glitches,
            "tail_bytes": self.tail_bytes,
            "tail_dropout": self.tail_dropout,
        }


def packet_span(raw: bytes) -> int:
    """Return the byte count covered by parsed packets (the page's data span).

    Bytes after the last complete packet are the page's unchecksummed tail
    (type-5 padding or inter-page tone). When the page never reached a valid end
    control, the whole raw stream is treated as payload: unfinished packet bytes
    must not be silently excluded from the round-trip comparison.
    """
    packets, _, status = framing.parse_page(raw)
    if not packets or not status.complete:
        return len(raw)
    return max(packet.offset + packet.size for packet in packets)


def roundtrip_page(page: framing.DecodedPage) -> RoundTrip:
    """Compare the re-encoded MFM edges to the measured source transitions.

    ``exact`` is the strict bit-level comparison over the whole stored byte
    stream. ``payload_exact`` restricts the comparison to the checksummed packet
    span (``raw[:packet_span]``); tail/padding bytes are reported separately.
    ``missing`` are measured transitions the encoding does not predict; ``extra``
    are encoded transitions the tape does not contain. ``tail_dropout`` is a
    strict mismatch that lies entirely outside the packet span.
    """
    expected = set(int(value) for value in encode_page_edges(page))
    measured = set(int(value) for value in page.measured_half)
    missing = measured - expected
    extra = expected - measured
    # Transition-set equality is not enough: if the observed signal ends before
    # the last byte, the absent transitions are indistinguishable from a zero
    # bit, so exactness also requires observed coverage of every framed byte.
    exact = not missing and not extra and page.observed_coverage

    span = packet_span(page.raw)
    truncated = page.raw[:span]
    payload_expected = set(int(value) for value in mfm_half_cells([1] + bytes_to_bits(truncated)))
    cutoff = 18 * span + 1
    payload_measured = {int(value) for value in page.measured_half if int(value) <= cutoff}
    payload_missing = payload_measured - payload_expected
    payload_extra = payload_expected - payload_measured
    payload_exact = not payload_missing and not payload_extra
    # MFM data bits live on odd half-cells and clock bits on even ones. A
    # discrepancy on an even half-cell cannot change a decoded byte (and the XOR
    # checksums independently confirm the bytes), so it is a media clock glitch
    # rather than a decode error. Data-half discrepancies are real failures.
    payload_diff = payload_missing | payload_extra
    clock_glitches = sum(1 for edge in payload_diff if edge % 2 == 0)
    payload_data_exact = not any(edge % 2 == 1 for edge in payload_diff)
    tail_bytes = len(page.raw) - span
    tail_dropout = (not exact) and payload_exact and tail_bytes > 0

    return RoundTrip(
        page_id=page.page_id, expected=len(expected), measured=len(measured),
        missing=len(missing), extra=len(extra), exact=exact,
        payload_exact=payload_exact, payload_data_exact=payload_data_exact,
        payload_expected=len(payload_expected),
        payload_measured=len(payload_measured), payload_missing=len(payload_missing),
        payload_extra=len(payload_extra), clock_glitches=clock_glitches,
        tail_bytes=tail_bytes, tail_dropout=tail_dropout,
    )


def synthesize_page_audio(page: framing.DecodedPage, sample_rate: int,
                          half_cell: float, lead_in_half_cells: int = 0,
                          tail_half_cells: int = 0) -> np.ndarray:
    """Render a decoded page (plus lead-in/tail) as an MFM flux-derivative pulse train.

    The tape head differentiates flux, so a transition is a signed impulse that
    flips polarity. This is the ideal input the front-end expects; feeding it
    back through ``frontend -> clock -> framing`` is the audio-level round-trip.
    """
    page_edges = encode_page_edges(page)
    offset = 2 * lead_in_half_cells
    last = int(page_edges.max()) + 1 if page_edges.size else 0
    total_half = offset + last + 2 * tail_half_cells + 2
    flips = np.zeros(total_half, dtype=np.int64)
    lead_in = np.arange(0, offset, 2, dtype=np.int64)  # 0-bit clock tone
    edges = np.concatenate([lead_in, page_edges + offset])
    valid = edges[(edges >= 0) & (edges < total_half)]
    np.add.at(flips, valid, 1)
    level_per_half = np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0)
    # Cumulative fractional sample positions keep the requested bitrate: a fixed
    # integer samples-per-half-cell would round 1/(2*4800) s at 44.1 kHz to a
    # different rate. Rounding each half-cell boundary to the nearest sample
    # preserves the average half-cell duration.
    boundaries = np.round(
        np.arange(total_half + 1, dtype=np.float64) * half_cell * sample_rate
    ).astype(np.int64)
    counts = np.diff(boundaries)
    square = np.repeat(level_per_half, counts)
    return np.diff(square)


def bandlimit_pulses(signal: np.ndarray, sample_rate: int,
                     sigma_half_cells: float = 0.4) -> np.ndarray:
    """Shape ideal impulses into band-limited playback pulses.

    ``synthesize_page_audio`` emits one-sample impulses. Real playback pulses are
    band-limited; ideal impulses make the front-end's band-pass ring at the MFM
    minimum spacing, which no detector can reject without also dropping weak real
    transitions at dropout edges.
    """
    sigma = sigma_half_cells * (1.0 / (2.0 * 4800.0)) * sample_rate
    radius = int(round(4.0 * sigma))
    offsets = np.arange(-radius, radius + 1) / sigma
    kernel = np.exp(-0.5 * offsets * offsets)
    kernel /= kernel.sum()
    return np.convolve(signal, kernel, mode="same")


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    from . import container

    with open(path, "wb") as handle:
        handle.write(container.wav_bytes(samples, sample_rate))
