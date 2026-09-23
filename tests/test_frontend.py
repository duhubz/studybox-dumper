"""Synthetic pulse-train test for the analog front-end."""

from __future__ import annotations

import unittest
import warnings

import numpy as np

try:  # unittest discover -s tests puts tests/ on sys.path
    from . import capture_fixture
except ImportError:
    import capture_fixture

from studybox import frontend


def mfm_half_cells(data_bits: list[int]) -> list[int]:
    encoded: list[int] = []
    previous = 0
    for bit in data_bits:
        encoded.append(1 if (bit == 0 and previous == 0) else 0)
        encoded.append(bit)
        previous = bit
    return encoded


def pulse_train(data_bits: list[int], samples_per_half: int, sample_rate: int) -> np.ndarray:
    encoded = mfm_half_cells(data_bits)
    edges = [i for i, value in enumerate(encoded) if value]
    length = (len(encoded) + 12) * samples_per_half
    pulses = np.zeros(length, dtype=np.float64)
    polarity = 1.0
    for edge in edges:
        pulses[edge * samples_per_half] += polarity
        polarity = -polarity
    sigma = 1.2
    radius = int(4 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(pulses, kernel, mode="same"), np.array(edges) * samples_per_half


class FrontendTest(unittest.TestCase):
    def test_pulse_train_transitions_are_recovered(self) -> None:
        data_bits = [0] * 60 + [1] + [((0xA5 >> shift) & 1) for shift in range(7, -1, -1)]
        sample_rate = 96000
        samples_per_half = 10  # 4800 bps at 96 kHz
        signal, expected = pulse_train(data_bits, samples_per_half, sample_rate)

        transitions = frontend.detect_transitions(signal, sample_rate)
        self.assertGreaterEqual(len(transitions), len(expected) - 2)
        self.assertLessEqual(len(transitions), len(expected) + 2)

        recovered = transitions.samples
        for position in expected[:len(recovered)]:
            self.assertLess(float(np.min(np.abs(recovered - position))), 5.0)

    def test_interval_clusters_are_two_three_four_half_cells(self) -> None:
        data_bits = [0] * 200 + [1] + [
            bit for byte in (0x00, 0xFF, 0xA5, 0x5A, 0x33, 0xCC) for bit in
            [(byte >> shift) & 1 for shift in range(7, -1, -1)]
        ]
        sample_rate = 96000
        samples_per_half = 10
        signal, _ = pulse_train(data_bits, samples_per_half, sample_rate)
        transitions = frontend.detect_transitions(signal, sample_rate)

        half = np.median(np.diff(transitions.samples)) / 2.0
        units = np.round(np.diff(transitions.samples) / half)
        self.assertTrue(np.isin(units, [2, 3, 4]).mean() > 0.98)


class EnvelopeSilenceTest(unittest.TestCase):
    """The power envelope must be clamped before the square root (#D5)."""

    def test_pure_silence_is_finite_without_transitions(self) -> None:
        signal = np.zeros(96000, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            transitions = frontend.detect_transitions(signal, 96000)
        self.assertEqual(transitions.samples.size, 0)
        self.assertTrue(np.all(np.isfinite(transitions.samples)))

    def test_data_then_silence_warns_not_and_stays_quiet(self) -> None:
        # A loud pulse train followed by silence produced ~70k tiny negative
        # moving-average power samples; np.sqrt then warned and returned NaN.
        page = capture_fixture.decoded_page(1)
        data = capture_fixture.data_signal(page, 96000, lead_in_half_cells=9600,
                                           tail_half_cells=9600)
        lead = np.zeros(96000, dtype=np.float32)
        signal = np.concatenate([lead, data.astype(np.float32),
                                 np.zeros(96000, dtype=np.float32)])
        data_end = lead.size + data.size

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            transitions = frontend.detect_transitions(signal, 96000)

        self.assertTrue(np.all(np.isfinite(transitions.samples)))
        # Settled silence after the last data transition stays quiet.
        self.assertEqual(int(np.sum(transitions.samples > data_end)), 0)


if __name__ == "__main__":
    unittest.main()
