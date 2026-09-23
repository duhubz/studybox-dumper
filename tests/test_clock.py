"""Tests for half-cell estimation and MFM clock recovery."""

from __future__ import annotations

import unittest

import numpy as np

from studybox import clock


class ClockTest(unittest.TestCase):
    def test_estimate_half_cell_from_two_cell_cluster(self) -> None:
        intervals = np.array([2, 2, 2, 3, 4, 2, 3, 2, 4, 2], dtype=np.float64) * 100.0
        self.assertAlmostEqual(clock.estimate_half_cell(intervals), 100.0)

    def test_recover_grid_units(self) -> None:
        edges = np.array([0, 2, 5, 7, 11, 13, 16, 18], dtype=np.float64)
        grid = clock.recover_grid(edges, sample_rate=1.0)
        self.assertEqual(grid.units.tolist(), [2, 3, 2, 4, 2, 3, 2])
        self.assertEqual(grid.half_index.tolist(), [0, 2, 5, 7, 11, 13, 16, 18])
        self.assertTrue(grid.valid.all())

    def test_gap_is_a_break(self) -> None:
        edges = np.array([0, 2, 4, 6, 18, 20, 22], dtype=np.float64)
        grid = clock.recover_grid(edges, sample_rate=1.0)
        self.assertEqual(grid.breaks.tolist(), [3])

    def test_tracks_gradual_speed_drift(self) -> None:
        # A 30% slow drift across 6000 valid intervals: the PLL must follow it
        # and recover every interval count without inventing breaks (#5).
        pattern = np.array([2, 3, 2, 4, 2, 3, 2, 3], dtype=np.int64)
        counts = np.tile(pattern, 750)
        half = np.linspace(1.0, 1.3, counts.size)
        edges = np.concatenate([[0.0], np.cumsum(counts * half)])

        grid = clock.recover_grid(edges, sample_rate=1.0)

        self.assertTrue(np.array_equal(grid.units, counts))
        self.assertEqual(grid.breaks.tolist(), [])
        # The PLL lags a continuing ramp by ~rate/gain; close is enough.
        self.assertAlmostEqual(grid.half_cell, 1.3, delta=0.02)

    def test_outlier_innovations_are_rejected(self) -> None:
        # Dropout-edge noise implies half-cells far from the running estimate.
        # Accepting those updates drags the estimate down on real tapes until
        # it latches onto a wrong frequency; they must not move it.
        valid = np.tile([2.0, 3.0, 2.0, 4.0], 1000)
        noise = np.full(500, 2.7)  # counted as 3 cells, implied half-cell 0.9
        edges = np.concatenate([[0.0], np.cumsum(np.concatenate([valid, noise]))])

        grid = clock.recover_grid(edges, sample_rate=1.0)

        self.assertAlmostEqual(grid.half_cell, 1.0, delta=0.01)

    def test_leadin_runs(self) -> None:
        units = np.array([3, 2, 2, 2, 2, 3, 4, 2, 2, 3], dtype=np.int64)
        self.assertEqual(clock.leadin_runs(units, min_length=4), [(1, 5)])
        self.assertEqual(len(clock.leadin_runs(units, min_length=2)), 2)


if __name__ == "__main__":
    unittest.main()
