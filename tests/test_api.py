"""Tests for the shared decode/verify helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

try:  # unittest discover -s tests puts tests/ on sys.path
    from . import capture_fixture
except ImportError:
    import capture_fixture

from studybox import api, audio_in, container


class DecodeOutcomeTest(unittest.TestCase):
    def test_decode_file_bundles_capture_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = capture_fixture.write_capture(
                Path(tmp) / "cap.wav", sample_rate=96000)
            outcome = api.decode_file(path)

            self.assertIsInstance(outcome, api.DecodeOutcome)
            self.assertEqual(outcome.capture.data_channel, 1)
            self.assertGreater(len(outcome.pages), 0)
            self.assertEqual(outcome.pages, outcome.result.pages)
            self.assertEqual(outcome.checksum_rate, outcome.result.checksum_rate)
            self.assertEqual(outcome.checksum_total, outcome.result.checksum_total)

    def test_verify_file_uses_the_resolved_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = capture_fixture.write_capture(
                Path(tmp) / "cap.wav", sample_rate=96000)
            report = api.verify_file(path)
            self.assertTrue(report.passed, report.render())


class OutputAliasProtectionTest(unittest.TestCase):
    """A destination must never replace one of the capture's sources (#3)."""

    def _decode_capture(self, tmp: str):
        path = capture_fixture.write_capture(
            Path(tmp) / "cap.wav", sample_rate=96000)
        return api.decode_file(path)

    def test_write_studybox_rejects_data_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outcome = self._decode_capture(tmp)
            original = outcome.capture.data_path.read_bytes()

            with self.assertRaises(ValueError):
                api.write_studybox(outcome.capture.data_path, outcome.result,
                                   outcome.capture)

            self.assertEqual(outcome.capture.data_path.read_bytes(), original)

    def test_write_studybox_rejects_hardlink_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outcome = self._decode_capture(tmp)
            alias = Path(tmp) / "alias.wav"
            try:
                os.link(outcome.capture.data_path, alias)
            except OSError:
                self.skipTest("hardlinks unavailable")
            original = outcome.capture.data_path.read_bytes()

            with self.assertRaises(ValueError):
                api.write_studybox(alias, outcome.result, outcome.capture)

            self.assertEqual(outcome.capture.data_path.read_bytes(), original)

    def test_write_studybox_rejects_symlink_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outcome = self._decode_capture(tmp)
            alias = Path(tmp) / "alias.wav"
            try:
                alias.symlink_to(outcome.capture.data_path)
            except OSError:
                self.skipTest("symlinks unavailable")
            original = outcome.capture.data_path.read_bytes()

            with self.assertRaises(ValueError):
                api.write_studybox(alias, outcome.result, outcome.capture)

            self.assertEqual(outcome.capture.data_path.read_bytes(), original)

    def test_write_studybox_rejects_silence_export_over_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = capture_fixture.decoded_page(1)
            signal = capture_fixture.data_signal(
                page, 96000, lead_in_half_cells=2400, tail_half_cells=500)
            path = Path(tmp) / "mono.wav"
            path.write_bytes(container.wav_bytes(signal, 96000))
            capture = audio_in.resolve_file(path, 0)
            self.assertFalse(capture.has_narration)
            result = api.decode_file(capture).result
            original = path.read_bytes()

            with self.assertRaises(ValueError):
                api.write_studybox(path, result, capture, no_audio=True)

            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
