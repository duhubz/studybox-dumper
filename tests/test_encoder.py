"""Round-trip tests for the MFM encoder."""

from __future__ import annotations

import unittest

import numpy as np

from studybox import encoder, framing, frontend


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


def make_page(page_id: int) -> framing.DecodedPage:
    raw = (page_header(page_id) + segment_header(2, 1, 0x60)
           + data_packet(bytes(range(128))) + control(0xF5))
    page = framing.DecodedPage(page_id=page_id, terminator_bit=0,
                               lead_in_sample=0, audio_sample=0, raw=raw)
    page.measured_half = encoder.encode_page_edges(page)
    return page


class EncoderTest(unittest.TestCase):
    def test_edge_roundtrip_is_exact(self) -> None:
        page = make_page(7)
        trip = encoder.roundtrip_page(page)
        self.assertTrue(trip.exact, trip)
        self.assertEqual(trip.missing, 0)
        self.assertEqual(trip.extra, 0)

    def test_missing_transition_is_detected(self) -> None:
        page = make_page(1)
        page.measured_half = page.measured_half[:-1]  # drop the last transition
        trip = encoder.roundtrip_page(page)
        self.assertFalse(trip.exact)
        self.assertGreater(trip.extra, 0)

    def test_tail_dropout_is_not_a_payload_failure(self) -> None:
        page = make_page(3)
        payload_measured = encoder.encode_page_edges(page)
        page.raw = page.raw + b"\xaa\xaa"  # unchecksummed tail
        page.measured_half = payload_measured  # tape has no transitions in the tail
        trip = encoder.roundtrip_page(page)
        self.assertFalse(trip.exact)
        self.assertTrue(trip.payload_exact, trip)
        self.assertTrue(trip.tail_dropout, trip)
        self.assertEqual(trip.tail_bytes, 2)

    def test_missing_payload_transition_fails_payload(self) -> None:
        page = make_page(4)
        present = [int(v) for v in page.measured_half]
        data_edge = next(e for e in present if e % 2 == 1)
        page.measured_half = np.array([e for e in present if e != data_edge])
        trip = encoder.roundtrip_page(page)
        self.assertFalse(trip.exact)
        self.assertFalse(trip.payload_exact, trip)
        self.assertFalse(trip.payload_data_exact, trip)
        self.assertFalse(trip.tail_dropout, trip)

    def test_clock_glitch_is_not_a_data_failure(self) -> None:
        page = make_page(5)
        present = {int(v) for v in page.measured_half}
        clock = next(e for e in range(0, 18 * len(page.raw), 2) if e not in present)
        page.measured_half = np.array(sorted(present | {clock}))
        trip = encoder.roundtrip_page(page)
        self.assertFalse(trip.payload_exact, trip)
        self.assertTrue(trip.payload_data_exact, trip)
        self.assertGreaterEqual(trip.clock_glitches, 1)

    def _frontend_roundtrip(self, sample_rate: int) -> None:
        page = make_page(9)
        half_cell = 1.0 / (2.0 * 4800.0)
        signal = encoder.synthesize_page_audio(
            page, sample_rate, half_cell, lead_in_half_cells=400, tail_half_cells=50)
        signal = encoder.bandlimit_pulses(signal, sample_rate)

        transitions = frontend.detect_transitions(signal, sample_rate)
        result = framing.decode_transitions(transitions.samples, sample_rate)

        self.assertEqual(len(result.pages), 1)
        decoded = result.pages[0]
        self.assertEqual(decoded.page_id, 9)
        self.assertEqual(decoded.raw, page.raw)
        self.assertTrue(encoder.roundtrip_page(decoded).exact)

    def test_synthesized_audio_roundtrips_through_frontend(self) -> None:
        self._frontend_roundtrip(96000)

    def test_synthesized_audio_roundtrips_through_frontend_at_44100(self) -> None:
        # Genuine synthesis -> front-end -> clock -> framing at a fractional
        # sample rate, where a fixed integer samples-per-half-cell would shift
        # the bitrate to 4410 bps and break the decode.
        self._frontend_roundtrip(44100)

    def test_synthesized_audio_roundtrips_through_frontend_at_48000(self) -> None:
        self._frontend_roundtrip(48000)

    def test_truncated_eof_is_not_exact(self) -> None:
        page = make_page(1)
        data_bits = [0] * 300 + [1] + encoder.bytes_to_bits(page.raw)
        edges = encoder.mfm_half_cells(data_bits)[:-1]  # final transition lost

        result = framing.decode_transitions(edges.astype(np.float64), sample_rate=1.0)

        self.assertEqual(len(result.pages), 1)
        decoded = result.pages[0]
        self.assertFalse(decoded.observed_coverage)
        self.assertFalse(encoder.roundtrip_page(decoded).exact)

    def test_synthesis_roundtrips_at_44100(self) -> None:
        page = make_page(9)
        sample_rate = 44100
        half_cell = 1.0 / (2.0 * 4800.0)
        signal = encoder.synthesize_page_audio(
            page, sample_rate, half_cell, lead_in_half_cells=400, tail_half_cells=50)

        # The pulse train's non-zero samples are its flux transitions. They must
        # land on the exact half-cell grid at 44.1 kHz (where a half-cell is a
        # fractional 4.59375 samples); a fixed integer repeat would shift the
        # bitrate to 4410 bps and break this decode.
        transitions = np.nonzero(signal)[0].astype(np.float64)
        result = framing.decode_transitions(transitions, sample_rate)

        self.assertEqual(len(result.pages), 1)
        decoded = result.pages[0]
        self.assertEqual(decoded.page_id, 9)
        self.assertEqual(decoded.raw, page.raw)
        self.assertTrue(encoder.roundtrip_page(decoded).exact)

    def test_synthesis_preserves_bitrate_at_44100(self) -> None:
        page = make_page(9)
        sample_rate = 44100
        half_cell = 1.0 / (2.0 * 4800.0)
        edges = encoder.encode_page_edges(page)
        lead_in, tail = 400, 50
        total_half = 2 * lead_in + int(edges.max()) + 1 + 2 * tail + 2
        signal = encoder.synthesize_page_audio(
            page, sample_rate, half_cell, lead_in_half_cells=lead_in,
            tail_half_cells=tail)
        expected = round(total_half * half_cell * sample_rate) - 1
        self.assertEqual(signal.size, expected)


if __name__ == "__main__":
    unittest.main()
