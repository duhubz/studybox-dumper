"""Analog front-end: recover flux-transition times from the data channel.

The StudyBox data track stores MFM as magnetic flux reversals. A tape head
differentiates flux, so each reversal appears as a peak in the playback signal.
The signal here is therefore a pulse train, not a square wave; zero-crossing
timing is unreliable because playback equalisation rings on every pulse.

This module band-limits the channel, finds the pulse peaks (both polarities)
with an adaptive threshold, rejects glitches closer than ~1.4 half-cells, and
refines each peak to a sub-sample position by parabolic interpolation, returning
transition positions in samples and seconds. The threshold is kept low because
dropout edges are weak; see :func:`detect_transitions`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfilt

# Nominal half-cell (half of one 4800 bps bit) in seconds, used only as a
# default glitch-rejection distance; the clock layer estimates the real value.
NOMINAL_HALF_CELL_SEC = 1.0 / (2.0 * 4800.0)


@dataclass(frozen=True)
class Transitions:
    """Transition positions measured from a data-channel capture."""

    samples: np.ndarray  # sub-sample positions, in samples
    sample_rate: float

    @property
    def times(self) -> np.ndarray:
        return self.samples / self.sample_rate

    def __len__(self) -> int:
        return int(self.samples.size)


def bandpass(signal: np.ndarray, sample_rate: float,
             low: float = 400.0, high: float = 8000.0, order: int = 2) -> np.ndarray:
    """Band-limit the pulse train, preserving transition shape."""
    nyquist = sample_rate / 2.0
    high = min(high, nyquist * 0.98)
    low = max(low, 1.0)
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfilt(sos, signal.astype(np.float64))


def _threshold_peaks(y: np.ndarray, envelope: np.ndarray, fraction: float,
                     min_distance: float) -> np.ndarray:
    """Refined peak positions accepted at ``fraction`` of the envelope."""
    # An exactly zero envelope is true silence: the moving-average power
    # underflowed. Disable detection there explicitly, so a zero threshold
    # cannot expose denormal filter ringing as peaks (the NaN the old sqrt
    # returned suppressed it only by accident).
    threshold = np.where(envelope > 0.0, envelope * fraction, np.inf)

    positive, _ = find_peaks(y, height=threshold, distance=3)
    negative, _ = find_peaks(-y, height=threshold, distance=3)
    positions = np.concatenate([positive, negative])
    amplitude = np.concatenate([y[positive], -y[negative]])
    order = np.argsort(positions)
    positions, amplitude = positions[order], amplitude[order]

    kept: list[tuple[int, float]] = []
    for position, value in zip(positions, amplitude):
        if kept and (position - kept[-1][0]) < min_distance:
            if value > kept[-1][1]:
                kept[-1] = (int(position), float(value))
        else:
            kept.append((int(position), float(value)))

    return np.array([_refine(y, position) for position, _ in kept],
                    dtype=np.float64)


def detect_transitions(signal: np.ndarray, sample_rate: float,
                       threshold_fraction: float = 0.20,
                       min_distance_sec: float = 1.4 * NOMINAL_HALF_CELL_SEC,
                       amp_window_sec: float = 0.02) -> Transitions:
    """Extract flux-transition positions from one window of data channel.

    The threshold is deliberately a low fraction of the local RMS envelope:
    dropout edges carry weak transitions, and a higher floor truncates pages
    there. Band-limiting ring lobes can sit at the MFM minimum spacing, so they
    cannot be separated from genuine weak edges by threshold or distance; real
    playback pulses are band-limited, and the synthesis tests shape their ideal
    impulse train to match (see ``tests/test_encoder.py``).
    """
    y = bandpass(signal, sample_rate)
    window = max(1, int(round(amp_window_sec * sample_rate)))
    # The moving average of a squared signal is mathematically nonnegative, but
    # floating-point cancellation can leave tiny negatives in silence, which
    # would make np.sqrt warn and return NaN. Clamp before the root.
    power = uniform_filter1d(y * y, window, mode="nearest")
    envelope = np.sqrt(np.maximum(power, 0.0))
    samples = _threshold_peaks(y, envelope, threshold_fraction,
                               min_distance_sec * sample_rate)
    return Transitions(samples=samples, sample_rate=float(sample_rate))


def _refine(signal: np.ndarray, index: int) -> float:
    """Parabolic sub-sample peak location around ``index``."""
    if index <= 0 or index >= signal.size - 1:
        return float(index)
    left, center, right = signal[index - 1], signal[index], signal[index + 1]
    denom = left - 2.0 * center + right
    if denom == 0.0:
        return float(index)
    offset = -0.5 * (right - left) / denom
    if abs(offset) > 1.0:
        return float(index)
    return index + offset
