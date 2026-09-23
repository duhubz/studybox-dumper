"""Routing and narration tests for capture input."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:  # unittest discover -s tests puts tests/ on sys.path
    from . import capture_fixture
except ImportError:
    import capture_fixture

from studybox import api, audio_in, container, framing


class RoutingTest(unittest.TestCase):
    def _stereo(self, directory: str) -> Path:
        return capture_fixture.write_capture(Path(directory) / "cap.wav")

    def _mono(self, directory: str) -> Path:
        path = Path(directory) / "mono.wav"
        path.write_bytes(container.wav_bytes(np.zeros(4096, dtype=np.float32), 44100))
        return path

    def test_stereo_narration_is_opposite_data_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._stereo(tmp)
            capture = audio_in.resolve_file(path)
            self.assertEqual(capture.data_channel, 1)
            self.assertEqual(capture.audio_channel, 0)
            self.assertEqual(capture.audio_path, path)
            self.assertTrue(capture.has_narration)

            flipped = audio_in.resolve_file(path, data_channel=0)
            self.assertEqual(flipped.data_channel, 0)
            self.assertEqual(flipped.audio_channel, 1)

    def test_invalid_data_channel_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._stereo(tmp)
            for channel in (2, -1, 99):
                with self.assertRaises(ValueError):
                    audio_in.resolve_file(path, data_channel=channel)

    def test_negative_channel_helpers_fall_back_to_channel_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._stereo(tmp)

            np.testing.assert_array_equal(
                audio_in.read_channel(path, -1), audio_in.read_channel(path, 0))
            negative = np.concatenate([
                block for _, block in audio_in.iter_windows(
                    path, channel=-1, window_frames=1024)])
            zero = np.concatenate([
                block for _, block in audio_in.iter_windows(
                    path, channel=0, window_frames=1024)])
            np.testing.assert_array_equal(negative, zero)

    def test_paired_rate_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "casa-channel2.wav").write_bytes(
                container.wav_bytes(np.zeros(2048, dtype=np.float32), 44100))
            (root / "casa-channel1.wav").write_bytes(
                container.wav_bytes(np.zeros(2048, dtype=np.float32), 48000))
            with self.assertRaises(ValueError):
                audio_in.resolve_side(root, "A")

            # Equal rates resolve normally and expose both tracks.
            (root / "casa-channel1.wav").write_bytes(
                container.wav_bytes(np.zeros(2048, dtype=np.float32), 44100))
            capture = audio_in.resolve_side(root, "A")
            self.assertEqual(capture.data_channel, 0)
            self.assertTrue(capture.has_narration)

    def test_mono_file_with_stereo_side_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "casa.wav").write_bytes(
                container.wav_bytes(np.zeros(2048, dtype=np.float32), 44100))

            with self.assertRaisesRegex(ValueError, "has only 1 channel"):
                audio_in.resolve_side(root, "A")

    def test_missing_narration_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._mono(tmp)
            capture = audio_in.resolve_file(path, data_channel=0)
            self.assertFalse(capture.has_narration)
            self.assertIsNone(capture.audio_path)
            self.assertIsNone(capture.audio_channel)

            page = framing.DecodedPage(
                page_id=0, terminator_bit=0, lead_in_sample=0, audio_sample=0,
                raw=capture_fixture.raw_page(0))
            result = framing.DecodeResult({}, 44100, 0.0, 0, [page], 0)
            with self.assertRaises(ValueError):
                api.decode_to_container(result, capture)
            box = api.decode_to_container(result, capture, no_audio=True)
            self.assertTrue(box.audio)

    def test_write_studybox_without_narration_streams_silence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._mono(tmp)
            capture = audio_in.resolve_file(path, data_channel=0)
            page = framing.DecodedPage(page_id=0, terminator_bit=0,
                                       lead_in_sample=0, audio_sample=50,
                                       raw=capture_fixture.raw_page(0))
            result = framing.DecodeResult({}, 44100, 0.0, 0, [page], 0)
            out = Path(tmp) / "out.studybox"

            written, pages, audio_bytes = api.write_studybox(
                out, result, capture, no_audio=True)

            self.assertEqual(written, out)
            self.assertEqual(pages, 1)
            self.assertGreater(audio_bytes, 0)
            box = container.StudyBox.read(out)
            self.assertEqual(len(box.pages), 1)
            self.assertEqual(container.wav_frames(box.audio), 4096)


if __name__ == "__main__":
    unittest.main()
