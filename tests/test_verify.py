"""Tests for the independent decode gate and reports."""

from __future__ import annotations

import unittest

import numpy as np

from studybox import container, framing, report, verify


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


def valid_page(page_id: int) -> bytes:
    return (page_header(page_id) + segment_header(2, 1, 0x60)
            + data_packet(bytes(range(128))) + control(0xF5))


class VerifyTest(unittest.TestCase):
    def test_valid_pages_pass(self) -> None:
        gate = verify.evaluate_pages([valid_page(0), valid_page(1)])
        self.assertTrue(gate.passed, gate.render())
        self.assertGreater(gate.checksum_total, 0)
        self.assertEqual(gate.checksum_ok, gate.checksum_total)

    def test_malformed_header_fails(self) -> None:
        data = bytearray(valid_page(0))
        data[1] = 0x02  # break the required 01 bytes
        gate = verify.evaluate_pages([bytes(data)])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("c5_page_headers", failed)

    def test_corrupt_checksum_lowers_rate(self) -> None:
        data = bytearray(valid_page(2))
        data[-1] ^= 0xFF
        gate = verify.evaluate_pages([bytes(data)])
        self.assertFalse(gate.passed)
        self.assertLess(gate.checksum_rate, 1.0)
        self.assertTrue(gate.dropout_map)

    def test_dropout_fails_by_default(self) -> None:
        gate = verify.evaluate_pages(
            [valid_page(0)],
            dropout_map=[{"kind": "mid_page_desync", "page_id": 0, "byte_offset": 1}])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("no_dropouts", failed)

    def test_subthreshold_checksum_fails_by_default(self) -> None:
        data = bytearray(valid_page(2))
        data[-1] ^= 0xFF
        gate = verify.evaluate_pages([bytes(data)])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("all_checksums", failed)

    def test_invalid_end_control_fails_completeness(self) -> None:
        data = bytearray(valid_page(2))
        data[-1] ^= 0xFF  # end-shaped control with a broken checksum
        gate = verify.evaluate_pages([bytes(data)])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("page_complete", failed)

    def test_allow_degraded_permits_tolerance(self) -> None:
        data = bytearray(valid_page(2))
        data[13] ^= 0xFF  # segment-header checksum; the page still ends validly
        gate = verify.evaluate_pages([bytes(data)], min_checksum_rate=0.75,
                                     allow_degraded=True)
        self.assertTrue(gate.passed, gate.render())
        self.assertTrue(gate.degraded)
        self.assertIn("all_checksums",
                      [c.name for c in gate.checks if not c.passed])

    def test_rate_floor_is_mandatory_under_tolerance(self) -> None:
        data = bytearray(valid_page(2))
        data[13] ^= 0xFF    # segment-header checksum
        data[144] ^= 0xFF   # data-packet checksum -> 2/4 = 0.5
        gate = verify.evaluate_pages([bytes(data)], min_checksum_rate=0.75,
                                     allow_degraded=True)
        self.assertFalse(gate.passed)
        self.assertLess(gate.checksum_rate, 0.75)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("packet_checksum_rate", failed)

    def test_min_checksum_rate_must_be_in_range(self) -> None:
        for value in (-0.01, 1.01, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                verify.evaluate_pages([valid_page(0)], min_checksum_rate=value)

    def test_container_verify_keeps_waived_checks_waived(self) -> None:
        data = bytearray(valid_page(0))
        data[13] ^= 0xFF  # 3/4 checksums, above the 0.75 floor
        wav = container.wav_bytes(np.zeros(1000, dtype="float32"), 44100)
        box = container.StudyBox(
            pages=[container.Page(0, 100, bytes(data))], audio=wav)
        gate = verify.verify_studybox(box, min_checksum_rate=0.75,
                                      allow_degraded=True)
        self.assertTrue(gate.passed, gate.render())
        self.assertTrue(gate.degraded)
        self.assertIn("all_checksums",
                      [c.name for c in gate.checks if not c.passed])

    def test_malformed_type5_padding_fails(self) -> None:
        # Repeated type bytes must match; 0x05 != 0x04 is a structural error even
        # though type-5 padding has no checksum byte (#6).
        data = page_header(0) + bytes([0xC5, 0x05, 0x04]) + control(0xF5)
        gate = verify.evaluate_pages([data])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("packet_structure", failed)

    def test_header_only_page_fails_completeness(self) -> None:
        gate = verify.evaluate_pages([page_header(0)])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("page_complete", failed)

    def test_truncated_data_packet_fails(self) -> None:
        data = page_header(0) + segment_header(2, 1, 0x60) + bytes([0xC5, 4, 1, 2])
        gate = verify.evaluate_pages([data])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("page_complete", failed)

    def test_truncated_control_packet_fails(self) -> None:
        data = page_header(0) + control(0x05)[:2]  # C5 00, no body/checksum
        gate = verify.evaluate_pages([data])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("page_complete", failed)

    def test_trailing_bytes_after_packets_are_not_desync(self) -> None:
        gate = verify.evaluate_pages([valid_page(0) + b"\xaa\xaa"])
        self.assertTrue(gate.passed, gate.render())

    def test_desync_inside_packet_span_fails(self) -> None:
        data = (page_header(0) + b"\xaa\xaa" + segment_header(2, 1, 0x60)
                + data_packet(bytes(range(128))) + control(0xF5))
        gate = verify.evaluate_pages([data])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("packet_parse_clean", failed)

    def test_container_gate_checks_offsets(self) -> None:
        wav = container.wav_bytes(np.zeros(1000, dtype="float32"), 44100)
        good = container.StudyBox(
            pages=[container.Page(0, 100, valid_page(0)),
                   container.Page(200, 300, valid_page(1))], audio=wav)
        gate = verify.verify_studybox(good)
        self.assertTrue(gate.passed, gate.render())

        bad = container.StudyBox(
            pages=[container.Page(0, 100, valid_page(0)),
                   container.Page(0, 50, valid_page(1))], audio=wav)
        gate = verify.verify_studybox(bad)
        self.assertFalse(gate.passed)

    def test_offsets_require_one_pair_per_page(self) -> None:
        gate = verify.evaluate_pages([valid_page(0)], offsets=[])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("offsets_valid", failed)

    def test_negative_or_transposed_offsets_fail(self) -> None:
        for offsets in ([(0, 100), (-2, -1)], [(100, 50)]):
            gate = verify.evaluate_pages([valid_page(0)] * len(offsets),
                                         offsets=offsets)
            self.assertFalse(gate.passed, offsets)
            failed = [c.name for c in gate.checks if not c.passed]
            self.assertIn("offsets_valid", failed)

    def test_out_of_order_offsets_fail(self) -> None:
        gate = verify.evaluate_pages([valid_page(0), valid_page(1)],
                                     offsets=[(0, 200), (50, 150)])
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("offsets_valid", failed)

    def test_oversized_page_rejected(self) -> None:
        with self.assertRaises(ValueError):
            verify.evaluate_pages([valid_page(0)], max_page_bytes=8)
        with self.assertRaises(ValueError):
            verify.evaluate_pages([valid_page(0), valid_page(1)], max_total_bytes=16)

    def test_missing_audio_fails(self) -> None:
        box = container.StudyBox(
            pages=[container.Page(0, 50, valid_page(0))], audio=b"")
        gate = verify.verify_studybox(box)
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("audio_present", failed)

    def test_offset_beyond_audio_fails(self) -> None:
        wav = container.wav_bytes(np.zeros(100, dtype="float32"), 44100)
        box = container.StudyBox(
            pages=[container.Page(0, 200, valid_page(0))], audio=wav)
        gate = verify.verify_studybox(box)
        self.assertFalse(gate.passed)
        failed = [c.name for c in gate.checks if not c.passed]
        self.assertIn("offsets_within_audio", failed)


class ReportTest(unittest.TestCase):
    def test_unchecked_padding_is_excluded_from_checksum_breakdown(self) -> None:
        raw = (page_header(0) + bytes([0xC5, 0x05, 0x05])
               + bytes([0xAA] * 8) + control(0xF5))
        packets, _, _ = framing.parse_page(raw)
        page = framing.DecodedPage(
            page_id=0, terminator_bit=0, lead_in_sample=0, audio_sample=0,
            packets=packets, raw=raw)

        payload = report.page_dict(page)

        self.assertEqual(payload["checksum_total"], 2)
        self.assertNotIn("segment_header", payload["packets"])

    def test_render_text_includes_totals(self) -> None:
        payload = {
            "capture": {"name": "casan"},
            "half_cell_us": 101.5,
            "totals": {"pages": 2, "checksum_ok": 10, "checksum_total": 10,
                        "checksum_rate": 1.0, "data128_ok": 4, "data128_total": 4},
            "pages": [{"page_id": 0, "lead_in_sample": 1, "audio_sample": 2,
                        "byte_length": 3, "checksum_ok": 5, "checksum_total": 5,
                        "data128_ok": 2, "data128_total": 2}],
            "dropouts": [],
        }
        text = report.render_text(payload)
        self.assertIn("casan", text)
        self.assertIn("page 0", text)


if __name__ == "__main__":
    unittest.main()
