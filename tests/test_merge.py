"""Tests for A/B page recovery (``merge``) over synthetic containers."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:  # unittest discover -s tests puts tests/ on sys.path
    from . import capture_fixture
except ImportError:
    import capture_fixture

from studybox import cli, container, merge, verify

FRAMES = 100000


def box_with(raws: list[bytes]) -> container.StudyBox:
    audio = container.wav_bytes(np.zeros(FRAMES, dtype=np.float32), 44100)
    pages = [
        container.Page(lead_in_offset=index * 1000 + 10,
                       audio_offset=index * 1000 + 500, data=raw)
        for index, raw in enumerate(raws)
    ]
    return container.StudyBox(pages=pages, audio=audio)


def page_with_bad_data_checksum(page_id: int, payload: bytes) -> bytes:
    raw = bytearray(capture_fixture.raw_page(page_id, payload))
    checksum = 8 + 6 + 2 + len(payload)
    raw[checksum] ^= 0x01
    return bytes(raw)


class MergeChoiceTest(unittest.TestCase):
    def test_identical_pages_keep_base(self) -> None:
        raw = capture_fixture.raw_page(0)
        outcome = merge.merge_boxes(box_with([raw]), [box_with([raw])], ["a", "b"])

        self.assertEqual(outcome.box.pages[0].data, raw)
        self.assertEqual(outcome.provenance["repaired"], 0)
        self.assertEqual(outcome.provenance["pages"][0]["resolution"], "identical")

    def test_prefix_dropout_uses_complete_side(self) -> None:
        full = capture_fixture.raw_page(1, bytes(range(128)))
        cut = full[:56]  # cut inside the data packet
        outcome = merge.merge_boxes(box_with([cut]), [box_with([full])], ["a", "b"])

        self.assertEqual(outcome.box.pages[0].data, full)
        self.assertEqual(outcome.provenance["repaired"], 1)
        self.assertIn("prefix_dropout", outcome.provenance["pages"][0]["resolution"])
        self.assertEqual(outcome.provenance["pages"][0]["source"], "b")

    def test_invalid_longer_prefix_is_left_open(self) -> None:
        full = bytearray(capture_fixture.raw_page(1, bytes(range(128))))
        full[-1] ^= 0x01  # invalidate the end-control checksum
        cut = bytes(full[:56])
        damaged_longer = bytes(full)

        outcome = merge.merge_boxes(
            box_with([cut]), [box_with([damaged_longer])], ["a", "b"])

        self.assertEqual(outcome.box.pages[0].data, cut)
        self.assertEqual(outcome.provenance["repaired"], 0)
        self.assertEqual(outcome.provenance["open_conflicts"], 1)
        entry = outcome.provenance["pages"][0]
        self.assertTrue(entry["open"])
        self.assertIn("invalid", entry["resolution"])
        self.assertEqual(len(entry["variants"]), 2)

    def test_structurally_invalid_padding_is_not_fully_valid(self) -> None:
        raw = (capture_fixture.page_header(1) + bytes([0xC5, 0x05, 0x04])
               + bytes([0xAA] * 8) + capture_fixture.control(0xF5))

        stats = merge._stats(raw)

        self.assertTrue(stats["complete"])
        self.assertEqual(stats["structural_errors"], 1)
        self.assertFalse(merge._fully_valid(stats))

    def test_single_checksum_clean_variant_wins(self) -> None:
        payload = bytes(range(64))
        clean = capture_fixture.raw_page(2, payload)
        dirty = page_with_bad_data_checksum(2, payload)
        outcome = merge.merge_boxes(box_with([dirty]), [box_with([clean])], ["a", "b"])

        self.assertEqual(outcome.box.pages[0].data, clean)
        self.assertIn("checksum_repair", outcome.provenance["pages"][0]["resolution"])

    def test_length_truncation_uses_complete_variant(self) -> None:
        payloads = [bytes([value]) * 128 for value in range(4)]
        long_raw = (capture_fixture.page_header(3)
                    + capture_fixture.segment_header(2, 1, 0x60))
        for payload in payloads:
            long_raw += capture_fixture.data_packet(payload)
        long_raw += capture_fixture.control(0xF5)
        short_raw = (capture_fixture.page_header(3)
                     + capture_fixture.segment_header(2, 1, 0x60))
        for payload in payloads[:3]:
            short_raw += capture_fixture.data_packet(payload)
        short_raw += capture_fixture.control(0xF5)
        outcome = merge.merge_boxes(box_with([short_raw]), [box_with([long_raw])],
                                    ["a", "b"])

        self.assertEqual(outcome.box.pages[0].data, long_raw)
        self.assertIn("length_truncation",
                      outcome.provenance["pages"][0]["resolution"])

    def test_third_clean_variant_keeps_length_truncation_conflict_open(self) -> None:
        long_raw = (capture_fixture.page_header(3)
                    + capture_fixture.segment_header(2, 1, 0x60))
        for payload in [bytes([value]) * 128 for value in range(4)]:
            long_raw += capture_fixture.data_packet(payload)
        long_raw += capture_fixture.control(0xF5)

        short_raw = (capture_fixture.page_header(3)
                     + capture_fixture.segment_header(2, 1, 0x60))
        for payload in [bytes([value]) * 128 for value in range(3)]:
            short_raw += capture_fixture.data_packet(payload)
        short_raw += capture_fixture.control(0xF5)

        middle_raw = (capture_fixture.page_header(3)
                      + capture_fixture.segment_header(2, 1, 0x60))
        for value in (0x22, 0x33, 0x44):
            middle_raw += capture_fixture.data_packet(bytes([value]) * 160)
        middle_raw += capture_fixture.control(0xF5)

        outcome = merge.merge_boxes(
            box_with([short_raw]), [box_with([long_raw]), box_with([middle_raw])],
            ["base", "long", "other-clean"])

        self.assertEqual(outcome.box.pages[0].data, short_raw)
        self.assertEqual(outcome.provenance["open_conflicts"], 1)
        self.assertTrue(outcome.provenance["pages"][0]["open"])

    def test_irreducible_conflict_keeps_base(self) -> None:
        base_raw = capture_fixture.raw_page(4, bytes(range(64)))
        other_raw = capture_fixture.raw_page(4, bytes(range(64, 128)))
        outcome = merge.merge_boxes(box_with([base_raw]), [box_with([other_raw])],
                                    ["a", "b"])

        self.assertEqual(outcome.box.pages[0].data, base_raw)
        self.assertEqual(outcome.provenance["open_conflicts"], 1)
        entry = outcome.provenance["pages"][0]
        self.assertTrue(entry["open"])
        self.assertEqual(entry["resolution"], "checksum_preserving_ambiguity")

    def test_conflicting_clean_donors_keep_damaged_base_open(self) -> None:
        base_raw = page_with_bad_data_checksum(4, b"\x00" * 128)
        left = capture_fixture.raw_page(4, b"\x11" * 128)
        right = capture_fixture.raw_page(4, b"\xee" * 128)

        outcome = merge.merge_boxes(
            box_with([base_raw]), [box_with([left]), box_with([right])],
            ["base", "left", "right"])

        self.assertEqual(outcome.box.pages[0].data, base_raw)
        self.assertEqual(outcome.provenance["open_conflicts"], 1)
        entry = outcome.provenance["pages"][0]
        self.assertTrue(entry["open"])
        self.assertEqual(entry["source"], "base")
        variants = entry["variants"]
        self.assertEqual(len(variants), 3)
        self.assertEqual(sum(item["checksum_clean"] for item in variants), 2)
        self.assertTrue(all("fully_valid" in item for item in variants))

    def test_page_count_mismatch_is_rejected(self) -> None:
        base = box_with([capture_fixture.raw_page(0)])
        other = box_with([capture_fixture.raw_page(0), capture_fixture.raw_page(1)])
        with self.assertRaises(ValueError):
            merge.merge_boxes(base, [other], ["a", "b"])

    def test_merged_container_verifies(self) -> None:
        full = capture_fixture.raw_page(5)
        outcome = merge.merge_boxes(box_with([full[:100]]), [box_with([full])],
                                    ["a", "b"])
        self.assertTrue(verify.verify_studybox(outcome.box).passed)

    def test_offsets_come_from_the_base(self) -> None:
        full = capture_fixture.raw_page(6)
        base = box_with([full[:100]])
        other = box_with([full])
        other.pages[0].lead_in_offset = 999
        other.pages[0].audio_offset = 5000
        outcome = merge.merge_boxes(base, [other], ["a", "b"])

        self.assertEqual(outcome.box.pages[0].lead_in_offset, 10)
        self.assertEqual(outcome.box.pages[0].audio_offset, 500)


class MergeCLITest(unittest.TestCase):
    def test_merge_writes_container_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            full = capture_fixture.raw_page(7)
            base_path = Path(tmp) / "a.studybox"
            other_path = Path(tmp) / "b.studybox"
            box_with([full[:100]]).write(base_path)
            box_with([full]).write(other_path)
            out = Path(tmp) / "merged.studybox"
            provenance = Path(tmp) / "merged.json"

            with contextlib.redirect_stdout(io.StringIO()):
                code = cli.main(["merge", str(base_path), str(other_path),
                                 "--studybox", str(out), "--json", str(provenance)])

            self.assertEqual(code, 0)
            self.assertEqual(container.StudyBox.read(out).pages[0].data, full)
            payload = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertEqual(payload["base"], "a.studybox")
            self.assertEqual(payload["sources"], ["a.studybox", "b.studybox"])
            self.assertEqual(payload["repaired"], 1)
            self.assertEqual(payload["open_conflicts"], 0)

    def test_merge_reports_page_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "a.studybox"
            other_path = Path(tmp) / "b.studybox"
            box_with([capture_fixture.raw_page(0)]).write(base_path)
            box_with([capture_fixture.raw_page(0),
                      capture_fixture.raw_page(1)]).write(other_path)
            out = Path(tmp) / "merged.studybox"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = cli.main(["merge", str(base_path), str(other_path),
                                 "--studybox", str(out)])

            self.assertEqual(code, 2)
            self.assertIn("page counts differ", stderr.getvalue())
            self.assertFalse(out.exists())

    def test_merge_rejects_output_over_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            full = capture_fixture.raw_page(8)
            base_path = Path(tmp) / "a.studybox"
            other_path = Path(tmp) / "b.studybox"
            box_with([full]).write(base_path)
            box_with([full]).write(other_path)
            original = base_path.read_bytes()

            with contextlib.redirect_stderr(io.StringIO()):
                code = cli.main(["merge", str(base_path), str(other_path),
                                 "--studybox", str(base_path)])

            self.assertEqual(code, 2)
            self.assertEqual(base_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
