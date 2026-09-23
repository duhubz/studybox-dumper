"""CLI round-trip gate tests."""

from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

try:  # unittest discover -s tests puts tests/ on sys.path
    from . import capture_fixture
except ImportError:
    import capture_fixture

from studybox import api, cli, container, encoder, framing

SAMPLE_RATE = 96000


def write_raw_page_capture(path, raw: bytes, sample_rate: int = SAMPLE_RATE):
    """Write a stereo capture whose data channel carries ``raw`` as one page."""
    page = framing.DecodedPage(page_id=raw[5] if len(raw) > 5 else None,
                               terminator_bit=0, lead_in_sample=0,
                               audio_sample=0, raw=raw)
    page.measured_half = encoder.encode_page_edges(page)
    data = encoder.synthesize_page_audio(
        page, sample_rate, capture_fixture.HALF_CELL,
        lead_in_half_cells=2400, tail_half_cells=500)
    signal = np.concatenate([np.zeros(24000, dtype=np.float32),
                             data.astype(np.float32),
                             np.zeros(24000, dtype=np.float32)])
    return capture_fixture.write_signal(path, signal, sample_rate)


class RoundtripCLITest(unittest.TestCase):
    def test_zero_exit_on_clean_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = capture_fixture.write_capture(
                Path(tmp) / "clean.wav", sample_rate=SAMPLE_RATE)
            json_path = Path(tmp) / "roundtrip.json"

            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                # The clamped power envelope must not warn on silent regions.
                warnings.simplefilter("error", RuntimeWarning)
                code = cli.main(["roundtrip", str(path), "--json", str(json_path)])

            self.assertEqual(code, 0)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["strict"])
            self.assertTrue(payload["exact"])
            self.assertTrue(payload["complete"])
            self.assertTrue(payload["checksums_ok"])
            self.assertTrue(payload["verify_passed"])
            self.assertFalse(any(item["kind"] == "discarded_page"
                                 for item in payload["capture_losses"]))

    def test_damaged_multi_page_capture_fails_strict(self) -> None:
        raws = [capture_fixture.raw_page(page_id) for page_id in (1, 2, 3)]
        raws[1] = bytes([0xC4]) + raws[1][1:]  # middle page magic damaged
        signal = capture_fixture.continuous_pages_signal(raws, SAMPLE_RATE)
        with tempfile.TemporaryDirectory() as tmp:
            path = capture_fixture.write_signal(
                Path(tmp) / "damaged.wav", signal, SAMPLE_RATE)
            json_path = Path(tmp) / "roundtrip.json"

            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(["roundtrip", str(path), "--json", str(json_path)])

            self.assertEqual(code, 1)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            # Retained pages still round-trip exactly, but the capture lost a
            # page: strict success must be refused by independent verification.
            self.assertTrue(payload["exact"])
            self.assertFalse(payload["strict"])
            self.assertFalse(payload["verify_passed"])
            self.assertTrue(any(item["kind"] == "discarded_page"
                                for item in payload["capture_losses"]))

    def test_malformed_type5_structure_fails_strict(self) -> None:
        raw = (capture_fixture.page_header(0)
               + bytes([0xC5, 0x05, 0x04])  # type-5 header bytes disagree
               + bytes([0xAA] * 20)
               + capture_fixture.control(0xF5))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_raw_page_capture(Path(tmp) / "badtype5.wav", raw)
            json_path = Path(tmp) / "roundtrip.json"

            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(["roundtrip", str(path), "--json", str(json_path)])

            self.assertEqual(code, 1)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["exact"])
            self.assertFalse(payload["strict"])
            self.assertFalse(payload["verify_passed"])

    def test_parser_desync_fails_strict(self) -> None:
        raw = (capture_fixture.page_header(0)
               + capture_fixture.segment_header(2, 1, 0x60)
               + capture_fixture.data_packet(bytes(range(16)))
               + b"\x00\x00"  # junk bytes inside the parsed packet span
               + capture_fixture.control(0xF5))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_raw_page_capture(Path(tmp) / "desync.wav", raw)
            json_path = Path(tmp) / "roundtrip.json"

            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(["roundtrip", str(path), "--json", str(json_path)])

            self.assertEqual(code, 1)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["exact"])
            self.assertFalse(payload["strict"])
            self.assertFalse(payload["verify_passed"])

    def test_nonzero_exit_on_incomplete_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = (capture_fixture.page_header(0)
                   + capture_fixture.segment_header(2, 1, 0x60)
                   + capture_fixture.data_packet(bytes(range(16))))  # no end control
            page = framing.DecodedPage(page_id=0, terminator_bit=0, lead_in_sample=0,
                                       audio_sample=0, raw=raw)
            page.measured_half = encoder.encode_page_edges(page)
            data = encoder.synthesize_page_audio(
                page, SAMPLE_RATE, capture_fixture.HALF_CELL,
                lead_in_half_cells=2400, tail_half_cells=500)
            signal = np.concatenate([
                np.zeros(24000, dtype=np.float32), data.astype(np.float32),
                np.zeros(24000, dtype=np.float32)])
            path = capture_fixture.write_signal(
                Path(tmp) / "bad.wav", signal, SAMPLE_RATE)
            json_path = Path(tmp) / "roundtrip.json"

            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                # The clamped power envelope must not warn on silent regions.
                warnings.simplefilter("error", RuntimeWarning)
                code = cli.main(["roundtrip", str(path), "--json", str(json_path)])

            self.assertEqual(code, 1)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["strict"])
            self.assertFalse(payload["complete"])


class OutputAliasCLITest(unittest.TestCase):
    """CLI outputs must not overwrite their own inputs (#3)."""

    def _capture(self, tmp: str) -> Path:
        return capture_fixture.write_capture(
            Path(tmp) / "cap.wav", sample_rate=SAMPLE_RATE)

    def test_decode_rejects_json_over_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._capture(tmp)
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["decode", str(path), "--json", str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(path.read_bytes(), original)

    def test_decode_rejects_container_over_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._capture(tmp)
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["decode", str(path), "--studybox", str(path),
                                 "--no-audio"])
            self.assertEqual(code, 2)
            self.assertEqual(path.read_bytes(), original)

    def test_decode_rejects_sidecar_over_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._capture(tmp)
            out = Path(tmp) / "out.studybox"
            with contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["decode", str(path), "--json", str(out),
                                 "--studybox", str(out)])
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())

    def test_verify_rejects_json_over_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._capture(tmp)
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["verify", str(path), "--json", str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(path.read_bytes(), original)

    def test_roundtrip_rejects_json_over_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._capture(tmp)
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["roundtrip", str(path), "--json", str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(path.read_bytes(), original)

    def test_verify_rejects_json_over_studybox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._capture(tmp)
            outcome = api.decode_file(path)
            box_path = Path(tmp) / "out.studybox"
            api.write_studybox(box_path, outcome.result, outcome.capture)
            original = box_path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["verify", "--studybox", str(box_path),
                                 "--json", str(box_path)])
            self.assertEqual(code, 2)
            self.assertEqual(box_path.read_bytes(), original)


class DecodeErrorCLITest(unittest.TestCase):
    def test_missing_capture_is_a_clean_cli_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.wav"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli.main(["decode", str(missing), "--quiet"])

            self.assertEqual(code, 2)
            self.assertIn("decode:", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

    def test_zero_page_decode_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "silent.wav"
            output = Path(tmp) / "empty.studybox"
            sidecar = Path(tmp) / "empty.json"
            sf.write(capture, np.zeros((44100, 2), dtype=np.float32), 44100,
                     format="WAV")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                code = cli.main(["decode", str(capture), "--quiet",
                                 "--studybox", str(output), "--json", str(sidecar)])

            self.assertEqual(code, 2)
            self.assertIn("no pages decoded", stderr.getvalue())
            self.assertFalse(output.exists())
            self.assertFalse(sidecar.exists())


class VerifyMalformedContainerCLITest(unittest.TestCase):
    def test_truncated_chunk_header_returns_clean_cli_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truncated.studybox"
            path.write_bytes(
                container.STBX_MAGIC
                + struct.pack("<II", 4, container.VERSION)
                + container.PAGE_MAGIC
                + b"\x00" * 11)
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                code = cli.main(["verify", "--studybox", str(path)])

            self.assertEqual(code, 2)
            self.assertIn("truncated PAGE chunk header", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
