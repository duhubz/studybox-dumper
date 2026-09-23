"""Half-cell estimation and MFM clock recovery.

MFM is run-length limited: the gap between flux transitions is 2, 3, or 4
half-cells. The lead-in tone (a long run of 0 data bits) produces transitions
every 2 half-cells, so the lower percentile of the interval distribution
estimates the shortest (2-half-cell) gap and thus the half-cell duration. A
first-order phase-locked loop then walks the transition list, quantising each
gap to an integer number of half-cells while tracking tape-speed drift,
producing the cumulative half-cell index of every transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_HALF_CELLS = 2
MAX_HALF_CELLS = 4
# Gaps longer than this are tape gaps/dropouts, not a valid MFM run length.
GAP_HALF_CELLS = 6
# Largest relative innovation accepted into the frequency estimate. Dropout
# edges and noise produce intervals whose implied half-cell is far from the
# running estimate; feeding those in drags the estimate down until it latches
# onto a wrong frequency. Gradual tape-speed drift stays well inside this band.
TRACK_TOLERANCE = 0.02


@dataclass(frozen=True)
class EdgeGrid:
    """Result of MFM clock recovery over a transition sequence."""

    half_index: np.ndarray   # cumulative half-cell index per transition
    units: np.ndarray        # half-cells between consecutive transitions
    half_cell: float         # final half-cell estimate, seconds
    start: int               # first transition index that was kept
    breaks: np.ndarray       # transition indices where a gap/dropout occurred

    @property
    def valid(self) -> np.ndarray:
        return (self.units >= MIN_HALF_CELLS) & (self.units <= MAX_HALF_CELLS)


def estimate_half_cell(intervals: np.ndarray, percentile: float = 10.0) -> float:
    """Estimate half-cell duration from raw inter-transition intervals."""
    if intervals.size == 0:
        raise ValueError("no intervals to estimate from")
    low = np.percentile(intervals, percentile)
    return float(low / MIN_HALF_CELLS)


def recover_grid(transition_samples: np.ndarray, sample_rate: float,
                 half_cell: float | None = None, gain: float = 0.005) -> EdgeGrid:
    """Quantise transitions onto a half-cell grid with a first-order PLL."""
    if transition_samples.size < 2:
        raise ValueError("need at least two transitions")
    seconds = np.diff(transition_samples) / sample_rate
    if half_cell is None or half_cell <= 0:
        half_cell = estimate_half_cell(seconds)

    # Quantise each gap against the *current* half-cell estimate, then let that
    # interval update the estimate. Quantising everything against the initial
    # value up front (the old behaviour) let a gradual speed drift accumulate
    # until the fixed estimate mis-counted whole intervals. Only in-band
    # innovations update the estimate, so noise cannot drag it away.
    h = float(half_cell)
    units = np.empty(seconds.size, dtype=np.int64)
    for i, gap in enumerate(seconds):
        count = int(round(gap / h))
        if count < 1:
            count = 1
        units[i] = count
        if MIN_HALF_CELLS <= count <= MAX_HALF_CELLS:
            target = gap / count
            if abs(target - h) <= h * TRACK_TOLERANCE:
                h += gain * (target - h)
    half_index = np.concatenate([[0], np.cumsum(units)])
    breaks = np.nonzero(units > MAX_HALF_CELLS)[0]

    return EdgeGrid(half_index=half_index, units=units, half_cell=h,
                    start=0, breaks=breaks)


def leadin_runs(units: np.ndarray, min_length: int = 24,
                value: int = MIN_HALF_CELLS) -> list[tuple[int, int]]:
    """Return ``(start, end)`` transition-index runs of ``value``-cell gaps.

    ``start`` is the index of the first transition in the run and ``end`` is the
    transition index one past the last one, so the terminator transition is at
    ``end``.
    """
    runs: list[tuple[int, int]] = []
    i = 0
    n = units.size
    while i < n:
        if units[i] != value:
            i += 1
            continue
        j = i
        while j < n and units[j] == value:
            j += 1
        if j - i >= min_length:
            runs.append((i, j))
        i = j
    return runs
