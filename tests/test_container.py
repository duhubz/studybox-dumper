"""Round-trip tests for the Mesen2 ``.studybox`` container."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:  # unittest discover -s tests puts tests/ on sys.path
    from . import capture_fixture
except ImportError:
    import capture_fixture

from studybox import audio_in, container


class ContainerTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        wav = container.wav_bytes(np.zeros(10000, dtype=np.float32), 44100)
        head0 = bytes([0xC5, 1, 1, 1, 1, 0, 0, 0xC5])
        head1 = bytes([0xC5, 1, 1, 1, 1, 1, 1, 0xC5])
        pages = [
            container.Page(lead_in_offset=10, audio_offset=200, data=head0),
            container.Page(lead_in_offset=5000, audio_offset=6000,
                           data=head1 + b"\xaa" * 120),
        ]
        box = container.StudyBox(pages=pages, audio=wav)
        restored = container.StudyBox.from_bytes(box.to_bytes())

        self.assertEqual(len(restored.pages), 2)
        self.assertEqual(restored.audio, wav)
        for original, copy in zip(pages, restored.pages):
            self.assertEqual(original.lead_in_offset, copy.lead_in_offset)
            self.assertEqual(original.audio_offset, copy.audio_offset)
            self.assertEqual(original.data, copy.data)
        self.assertEqual(restored.pages[0].page_id, 0)
        self.assertEqual(restored.pages[1].page_id, 1)

    def test_rejects_oversized_input(self) -> None:
        with self.assertRaises(ValueError):
            container.StudyBox.from_bytes(b"STBX" + b"\x00" * 100, max_bytes=32)

    def test_rejects_bad_magic(self) -> None:
        with self.assertRaises(ValueError):
            container.StudyBox.from_bytes(b"NOPE" + b"\x00" * 20)

    def test_empty_studybox_cannot_be_serialized(self) -> None:
        wav = container.wav_bytes(np.zeros(100, dtype=np.float32), 44100)
        with self.assertRaisesRegex(ValueError, "no pages"):
            container.StudyBox(audio=wav).to_bytes()

    def test_rejects_truncated_chunk_headers(self) -> None:
        header = container.STBX_MAGIC + struct.pack(
            "<II", 4, container.VERSION)
        cases = (
            (container.PAGE_MAGIC + b"\x00" * 11, "PAGE"),
            (container.AUDI_MAGIC + b"\x00" * 7, "AUDI"),
        )
        for chunk, name in cases:
            with self.subTest(chunk=name), self.assertRaisesRegex(
                    ValueError, f"truncated {name} chunk header"):
                container.StudyBox.from_bytes(header + chunk)

    def test_rejects_out_of_order_pages(self) -> None:
        box = container.StudyBox(pages=[
            container.Page(0, 100, b"\xc5" * 8),
            container.Page(0, 50, b"\xc5" * 8),
        ])
        with self.assertRaises(ValueError):
            box.to_bytes()

    def test_rejects_nonzero_file_type(self) -> None:
        wav = container.wav_bytes(np.zeros(100, dtype=np.float32), 44100)
        page = bytes([0xC5, 1, 1, 1, 1, 0, 0, 0xC5])
        blob = b"".join([
            container.STBX_MAGIC,
            struct.pack("<II", 4, container.VERSION),
            container.PAGE_MAGIC,
            struct.pack("<III", len(page) + 8, 0, 50),
            page,
            container.AUDI_MAGIC,
            struct.pack("<II", len(wav) + 4, 1),
            wav,
        ])
        with self.assertRaises(ValueError):
            container.StudyBox.from_bytes(blob)

    def test_wav_is_valid_mono16(self) -> None:
        import io

        import soundfile as sf

        wav = container.wav_bytes(np.zeros(44100, dtype=np.float32), 44100)
        info = sf.info(io.BytesIO(wav))
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.samplerate, 44100)
        self.assertEqual(info.subtype, "PCM_16")

    def test_wav_frames_rejects_non_wav_and_wrong_shape(self) -> None:
        import io

        import soundfile as sf

        def encode(fmt: str, subtype: str | None, samples: np.ndarray) -> bytes:
            buffer = io.BytesIO()
            kwargs = {"format": fmt}
            if subtype:
                kwargs["subtype"] = subtype
            sf.write(buffer, samples, 44100, **kwargs)
            return buffer.getvalue()

        self.assertIsNone(container.wav_frames(
            encode("FLAC", None, np.zeros(100, dtype=np.float32))))
        self.assertIsNone(container.wav_frames(
            encode("WAV", "PCM_16", np.zeros((100, 2), dtype=np.float32))))
        self.assertIsNone(container.wav_frames(
            encode("WAV", "FLOAT", np.zeros(100, dtype=np.float32))))

    def test_rejects_mislabeled_flac_payload(self) -> None:
        import io

        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(100, dtype=np.float32), 44100, format="FLAC")
        page = bytes([0xC5, 1, 1, 1, 1, 0, 0, 0xC5])
        blob = b"".join([
            container.STBX_MAGIC,
            struct.pack("<II", 4, container.VERSION),
            container.PAGE_MAGIC,
            struct.pack("<III", len(page) + 8, 0, 50),
            page,
            container.AUDI_MAGIC,
            struct.pack("<II", len(buffer.getvalue()) + 4, container.FILE_TYPE_WAV),
            buffer.getvalue(),
        ])
        with self.assertRaises(ValueError):
            container.StudyBox.from_bytes(blob)


class StreamingWriterTest(unittest.TestCase):
    def test_empty_page_list_is_rejected_before_touching_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.studybox"
            original = b"keep this destination"
            path.write_bytes(original)

            with self.assertRaisesRegex(ValueError, "no pages"):
                container.write_streaming(path, [], frames=100, silence=True)

            self.assertEqual(path.read_bytes(), original)

    def test_audi_length_overflow_never_opens_or_changes_destination(self) -> None:
        real_stat = Path.stat
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.studybox"
            original = b"existing destination must survive AUDI overflow"
            # The first size fits uint32 itself, but not after AUDI's type field.
            for wav_size in ((1 << 32) - 4, 1 << 32):
                with self.subTest(wav_size=wav_size):
                    path.write_bytes(original)

                    def fake_stat(source, *args, **kwargs):
                        if source.name == "audio.wav":
                            return mock.Mock(st_size=wav_size)
                        return real_stat(source, *args, **kwargs)

                    with mock.patch.object(Path, "stat", autospec=True,
                                           side_effect=fake_stat), \
                            mock.patch.object(container, "open", wraps=open,
                                              create=True) as output_open:
                        with self.assertRaises(struct.error):
                            container.write_streaming(
                                path, [container.Page(0, 0, b"test")],
                                frames=100, silence=True)
                        output_open.assert_not_called()
                    self.assertEqual(path.read_bytes(), original)

    def test_invalid_page_metadata_never_opens_or_changes_destination(self) -> None:
        cases = {
            "audio_order": [container.Page(0, 80), container.Page(0, 40)],
            "lead_order": [container.Page(30, 40), container.Page(20, 80)],
            "audio_before_lead": [container.Page(80, 40)],
            "negative": [container.Page(-1, 40)],
            "non_integer": [container.Page(0.5, 40)],
            "uint32_overflow": [container.Page(0, 1 << 32)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.studybox"
            original = b"existing destination must survive invalid input"
            for label, pages in cases.items():
                with self.subTest(case=label):
                    path.write_bytes(original)
                    with mock.patch.object(container, "open", wraps=open,
                                           create=True) as output_open, \
                            mock.patch.object(container.tempfile, "TemporaryDirectory",
                                              side_effect=AssertionError(
                                                  "invalid metadata reached audio preparation")):
                        with self.assertRaises((ValueError, struct.error)):
                            container.write_streaming(path, pages, frames=1 << 32,
                                                      silence=True)
                        output_open.assert_not_called()
                    self.assertEqual(path.read_bytes(), original)

    def test_streaming_equals_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cap_path = capture_fixture.write_capture(
                Path(tmp) / "cap.wav", sample_rate=96000)
            capture = audio_in.resolve_file(cap_path)
            info = audio_in.sf.info(str(capture.audio_path))
            narration = audio_in.read_channel(capture.audio_path, capture.audio_channel)

            page0 = bytes([0xC5, 1, 1, 1, 1, 0, 0, 0xC5])
            page1 = bytes([0xC5, 1, 1, 1, 1, 1, 1, 0xC5]) + b"\xaa" * 16
            pages = [container.Page(0, 50, page0),
                     container.Page(500, 800, page1)]

            expected = container.StudyBox(
                pages=pages,
                audio=container.wav_bytes(narration, int(info.samplerate))).to_bytes()
            out = Path(tmp) / "stream.studybox"
            container.write_streaming(out, pages, capture,
                                      int(info.samplerate), int(info.frames))

            self.assertEqual(out.read_bytes(), expected)

    def test_streaming_rejects_output_aliasing_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cap_path = capture_fixture.write_capture(
                Path(tmp) / "cap.wav", sample_rate=96000)
            capture = audio_in.resolve_file(cap_path)
            info = audio_in.sf.info(str(capture.audio_path))
            page = bytes([0xC5, 1, 1, 1, 1, 0, 0, 0xC5])
            pages = [container.Page(0, 50, page)]
            original = cap_path.read_bytes()

            with self.assertRaises(ValueError):
                container.write_streaming(cap_path, pages, capture,
                                          int(info.samplerate), int(info.frames))

            self.assertEqual(cap_path.read_bytes(), original)

    def test_streaming_silence_equals_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample_rate, frames = 44100, 5000
            page0 = bytes([0xC5, 1, 1, 1, 1, 0, 0, 0xC5])
            pages = [container.Page(0, 50, page0)]
            zero_wav = container.wav_bytes(np.zeros(frames, dtype=np.float32),
                                           sample_rate)
            expected = container.StudyBox(pages=pages, audio=zero_wav).to_bytes()

            out = Path(tmp) / "silent.studybox"
            size = container.write_streaming(out, pages, sample_rate=sample_rate,
                                             frames=frames, silence=True)

            self.assertEqual(out.read_bytes(), expected)
            self.assertEqual(size, len(zero_wav))


if __name__ == "__main__":
    unittest.main()
