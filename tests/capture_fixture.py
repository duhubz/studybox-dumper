"""Reusable synthetic capture fixtures for the test suite.

Builds a realistic 44.1 kHz stereo capture where the left channel carries
narration and the right channel carries a 4800 bps MFM data signal. The data
signal is rendered by :func:`studybox.encoder.synthesize_page_audio`, so it
round-trips through the project's own front-end. This is a helper module, not a
test module (``unittest`` discovery matches ``test*.py``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from studybox import encoder, framing

DEFAULT_SAMPLE_RATE = 44100
HALF_CELL = 1.0 / (2.0 * 4800.0)


def xor(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
    return value


def page_header(page_id: int) -> bytes:
    head = bytes([0xC5, 1, 1, 1, 1, page_id, page_id])
    return head + bytes([xor(head)])


def segment_header(kind: int, x: int, y: int) -> bytes:
    head = bytes([0xC5, kind, kind, x, y])
    return head + bytes([xor(head)])


def data_packet(payload: bytes) -> bytes:
    head = bytes([0xC5, len(payload)]) + payload
    return head + bytes([xor(head)])


def control(body: int) -> bytes:
    head = bytes([0xC5, 0x00, body])
    return head + bytes([xor(head)])


def raw_page(page_id: int = 0, payload: bytes | None = None) -> bytes:
    """Return a complete, checksummed, end-terminated page."""
    if payload is None:
        payload = bytes(range(128))
    return (page_header(page_id) + segment_header(2, 1, 0x60)
            + data_packet(payload) + control(0xF5))


def decoded_page(page_id: int = 0, payload: bytes | None = None) -> framing.DecodedPage:
    """Build an in-memory page whose measured edges match its own encoding."""
    raw = raw_page(page_id, payload)
    page = framing.DecodedPage(page_id=page_id, terminator_bit=0,
                               lead_in_sample=0, audio_sample=0, raw=raw)
    page.measured_half = encoder.encode_page_edges(page)
    return page


def data_signal(page: framing.DecodedPage, sample_rate: int = DEFAULT_SAMPLE_RATE,
                lead_in_half_cells: int = 0, tail_half_cells: int = 0) -> np.ndarray:
    """Render a page (plus lead-in/tail) as an MFM pulse train at 44.1 kHz."""
    signal = encoder.synthesize_page_audio(
        page, sample_rate, HALF_CELL,
        lead_in_half_cells=lead_in_half_cells,
        tail_half_cells=tail_half_cells)
    return encoder.bandlimit_pulses(signal, sample_rate)


def continuous_pages_signal(raws: list[bytes], sample_rate: int,
                            lead_in_half_cells: int = 305,
                            seam_half_cells: int = 2) -> np.ndarray:
    """Render concatenated raw pages as one continuous MFM pulse train.

    Unlike concatenating independent ``data_signal`` renders, the half-cell
    level (polarity) and sample grid are shared across pages, so the seams are
    ordinary 2-cell MFM intervals rather than artificial gaps. ``lead_in_half_cells``
    clock transitions precede each page's terminator.
    """
    edges: list[np.ndarray] = []
    cursor = 0
    for raw in raws:
        page = framing.DecodedPage(page_id=raw[5] if len(raw) > 5 else None,
                                   terminator_bit=0, lead_in_sample=0,
                                   audio_sample=0, raw=raw)
        lead_in = np.arange(0, 2 * lead_in_half_cells, 2, dtype=np.int64)
        page_edges = encoder.encode_page_edges(page) + cursor + 2 * lead_in_half_cells
        edges.append(lead_in + cursor)
        edges.append(page_edges)
        cursor = int(page_edges.max()) + seam_half_cells
    all_edges = np.concatenate(edges)
    total_half = cursor + 2
    flips = np.zeros(total_half, dtype=np.int64)
    np.add.at(flips, all_edges[all_edges < total_half], 1)
    level = np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0)
    boundaries = np.round(
        np.arange(total_half + 1, dtype=np.float64) * HALF_CELL * sample_rate
    ).astype(np.int64)
    square = np.repeat(level, np.diff(boundaries))
    return encoder.bandlimit_pulses(np.diff(square), sample_rate)


def write_signal(path: str | Path, signal: np.ndarray,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 narration_amplitude: float = 0.05) -> Path:
    """Write a stereo WAV: narration on the left, a real data signal on the right."""
    signal = np.asarray(signal, dtype=np.float32)
    t = np.arange(signal.size, dtype=np.float64) / sample_rate
    narration = (narration_amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.column_stack([narration, signal])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sample_rate, format="WAV", subtype="PCM_16")
    return path


def write_capture(path: str | Path, page_id: int = 0, payload: bytes | None = None,
                  sample_rate: int = DEFAULT_SAMPLE_RATE,
                  lead_seconds: float = 0.25, tail_seconds: float = 0.25,
                  narration_amplitude: float = 0.05) -> Path:
    """Write a stereo WAV with narration left and a complete MFM page right."""
    page = decoded_page(page_id, payload)
    data = data_signal(page, sample_rate,
                       lead_in_half_cells=int(lead_seconds * 9600),
                       tail_half_cells=int(tail_seconds * 9600))
    lead = np.zeros(int(lead_seconds * sample_rate), dtype=np.float32)
    tail = np.zeros(int(tail_seconds * sample_rate), dtype=np.float32)
    signal = np.concatenate([lead, data.astype(np.float32), tail])
    return write_signal(path, signal, sample_rate, narration_amplitude)
