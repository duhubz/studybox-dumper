"""Synthetic MFM round-trip and framing/checksum tests."""

from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

try:  # unittest discover -s tests puts tests/ on sys.path
    from . import capture_fixture
except ImportError:
    import capture_fixture

from studybox import api, audio_in, clock, encoder, framing, verify


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


def mfm_encode(data_bits: list[int]) -> np.ndarray:
    """Return transition edge indices for standard MFM of ``data_bits``."""
    encoded: list[int] = []
    previous = 0
    for bit in data_bits:
        clock_bit = 1 if (bit == 0 and previous == 0) else 0
        encoded.append(clock_bit)
        encoded.append(bit)
        previous = bit
    return np.array([i for i, value in enumerate(encoded) if value], dtype=np.float64)


def bytes_to_bits(stream: bytes) -> list[int]:
    bits: list[int] = []
    for byte in stream:
        bits.append(0)
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


class MFMRoundTripTest(unittest.TestCase):
    def test_clock_recovers_every_data_bit(self) -> None:
        stream = page_header(3) + segment_header(2, 1, 0x60) \
            + data_packet(bytes(range(128))) + control(0x05) \
            + segment_header(4, 2, 0x10) + data_packet(b"\xab" * 128) + control(0xF5)
        data_bits = [0] * 300 + [1] + bytes_to_bits(stream)

        edges = mfm_encode(data_bits)
        grid = clock.recover_grid(edges, sample_rate=1.0)
        region = framing.region_bits(grid.half_index, edges)
        recovered = region.bits.tolist()

        self.assertGreaterEqual(len(recovered), len(data_bits))
        self.assertEqual(recovered[:len(data_bits)], data_bits)

    def test_page_framing_and_checksums(self) -> None:
        stream = page_header(3) + segment_header(2, 1, 0x60) \
            + data_packet(bytes(range(128))) + control(0x05) \
            + segment_header(4, 2, 0x10) + data_packet(b"\xab" * 128) + control(0xF5)
        data_bits = [0] * 300 + [1] + bytes_to_bits(stream)
        edges = mfm_encode(data_bits)
        grid = clock.recover_grid(edges, sample_rate=1.0)
        region = framing.region_bits(grid.half_index, edges)

        terminators = framing.find_page_terminators(region.bits, min_leadin=256)
        self.assertEqual(len(terminators), 1)
        terminator, zero_run = terminators[0]
        self.assertEqual(zero_run, 300)

        framed = framing.frame_bytes(region.bits, terminator + 1)
        packets, desync, _ = framing.parse_page(bytes(value for value, _ in framed))

        self.assertEqual(desync, 0)
        self.assertEqual(packets[0].kind, "page_header")
        self.assertEqual(packets[0].page_id, 3)
        self.assertTrue(all(packet.checksum_ok for packet in packets))
        data128 = [p for p in packets if p.kind == "data"]
        self.assertEqual(len(data128), 2)
        self.assertTrue(all(p.size == 131 for p in data128))

    def test_corrupt_checksum_is_detected(self) -> None:
        stream = bytearray(page_header(5) + segment_header(2, 1, 0x60)
                           + data_packet(b"\x01\x02\x03") + control(0xF5))
        stream[-1] ^= 0xFF  # break the final control checksum
        data_bits = [0] * 300 + [1] + bytes_to_bits(bytes(stream))
        edges = mfm_encode(data_bits)
        grid = clock.recover_grid(edges, sample_rate=1.0)
        region = framing.region_bits(grid.half_index, edges)
        terminator, _ = framing.find_page_terminators(region.bits, 256)[0]
        packets, _, _ = framing.parse_page(
            bytes(v for v, _ in framing.frame_bytes(region.bits, terminator + 1)))
        self.assertEqual(packets[-1].kind, "control")
        self.assertFalse(packets[-1].checksum_ok)

    def test_type5_padding_has_no_checksum(self) -> None:
        stream = page_header(0) + bytes([0xC5, 0x05, 0x05]) + bytes([0xAA] * 20) \
            + control(0xF5)
        data_bits = [0] * 300 + [1] + bytes_to_bits(stream)
        edges = mfm_encode(data_bits)
        grid = clock.recover_grid(edges, sample_rate=1.0)
        region = framing.region_bits(grid.half_index, edges)
        terminator, _ = framing.find_page_terminators(region.bits, 256)[0]
        packets, desync, _ = framing.parse_page(
            bytes(v for v, _ in framing.frame_bytes(region.bits, terminator + 1)))
        self.assertEqual(desync, 0)
        padding = [p for p in packets if p.kind == "segment_header" and p.segment_type == 5]
        self.assertEqual(len(padding), 1)
        self.assertFalse(padding[0].checked)


class ParseCursorsTest(unittest.TestCase):
    def test_parse_page_accepts_bytes(self) -> None:
        stream = page_header(3) + segment_header(2, 1, 0x60) \
            + data_packet(bytes(range(8))) + control(0xF5)
        reference = framing.parse_page(stream)[0]
        for buffer in (stream, bytearray(stream), memoryview(stream)):
            packets, desync, status = framing.parse_page(buffer)
            self.assertEqual(desync, 0)
            self.assertTrue(status.complete)
            self.assertEqual([p.offset for p in packets],
                             [p.offset for p in reference])
            self.assertEqual(packets[-1].body, control(0xF5))

    def test_parse_page_start_cursor(self) -> None:
        prefix = b"\x00\x01"
        stream = prefix + page_header(3) + segment_header(2, 1, 0x60) \
            + data_packet(bytes(range(8))) + control(0xF5)
        packets, _, status = framing.parse_page(stream, start=len(prefix))
        self.assertTrue(status.complete)
        self.assertEqual(packets[0].offset, 0)
        self.assertEqual(packets[0].page_id, 3)


class EndControlCompletenessTest(unittest.TestCase):
    """An end-shaped control only completes the page when its checksum is valid (#5)."""

    def test_checksum_invalid_end_control_is_incomplete(self) -> None:
        stream = bytearray(page_header(0) + segment_header(2, 1, 0x60)
                           + data_packet(b"\x01\x02\x03") + control(0xF5))
        stream[-1] ^= 0xFF  # break the end control's checksum
        packets, _, status = framing.parse_page(bytes(stream))
        self.assertFalse(status.complete)
        self.assertEqual(status.termination, "bad_end_control")
        self.assertEqual(packets[-1].kind, "control")
        self.assertFalse(packets[-1].checksum_ok)

    def test_valid_end_control_is_complete(self) -> None:
        stream = page_header(0) + segment_header(2, 1, 0x60) \
            + data_packet(b"\x01\x02\x03") + control(0xF5)
        _, _, status = framing.parse_page(stream)
        self.assertTrue(status.complete)
        self.assertEqual(status.termination, "end_control")


class ByteFramingTest(unittest.TestCase):
    def test_region_bits_ignores_transitions_before_base(self) -> None:
        half_index = np.array([0, 2, 3, 5, 100, 101, 103], dtype=np.int64)
        samples = np.arange(7, dtype=np.float64)
        region = framing.region_bits(half_index, samples, base=100)
        self.assertEqual(region.bits.tolist(), [1, 1, 0])

    def test_nonzero_marker_stops_framing(self) -> None:
        bits = np.zeros(40, dtype=np.uint8)
        bits[0] = 1  # first marker bit is not zero
        self.assertEqual(framing.frame_bytes(bits, 0), [])

    def test_zero_bytes_are_payload_not_a_leadin(self) -> None:
        bits = np.zeros(40 * 9, dtype=np.uint8)
        framed = framing.frame_bytes(bits, 0)
        self.assertEqual(len(framed), 40)
        self.assertTrue(all(value == 0 for value, _ in framed))

    def test_end_bound_limits_framing(self) -> None:
        bits = np.zeros(40 * 9, dtype=np.uint8)
        framed = framing.frame_bytes(bits, 0, 10 * 9)
        self.assertEqual(len(framed), 10)


class ZeroRunRegressionTest(unittest.TestCase):
    """A payload with >=32 zero bytes must round-trip whole (Phase 7A bug)."""

    def test_payload_with_long_zero_run_framed_fully(self) -> None:
        payload = b"\x00" * 64 + bytes(range(64))
        stream = page_header(9) + segment_header(2, 1, 0x60) \
            + data_packet(payload) + control(0xF5)
        data_bits = [0] * 300 + [1] + bytes_to_bits(stream)
        edges = mfm_encode(data_bits)
        grid = clock.recover_grid(edges, sample_rate=1.0)
        region = framing.region_bits(grid.half_index, edges)

        terminator, _ = framing.find_page_terminators(region.bits, 256)[0]
        framed = framing.frame_bytes(region.bits, terminator + 1)
        packets, desync, status = framing.parse_page(
            bytes(value for value, _ in framed))

        self.assertEqual(desync, 0)
        self.assertTrue(status.complete)
        data = [packet for packet in packets if packet.kind == "data"]
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].body, payload)
        self.assertTrue(data[0].checksum_ok)

    def test_decode_transitions_recovers_long_zero_run(self) -> None:
        payload = b"\x00" * 64 + bytes(range(64))
        stream = page_header(9) + segment_header(2, 1, 0x60) \
            + data_packet(payload) + control(0xF5)
        data_bits = [0] * 300 + [1] + bytes_to_bits(stream)
        edges = mfm_encode(data_bits)

        result = framing.decode_transitions(edges, sample_rate=1.0)

        self.assertEqual(len(result.pages), 1)
        page = result.pages[0]
        self.assertEqual(page.page_id, 9)
        self.assertEqual(page.raw, stream)
        data = [packet for packet in page.packets if packet.kind == "data"]
        self.assertEqual(data[0].body, payload)
        self.assertTrue(data[0].checksum_ok)
        # A payload zero run is not a discarded page.
        self.assertFalse(any(item["kind"] == "discarded_page"
                             for item in result.unrecoverable))

    def test_false_terminator_in_payload_does_not_split_page(self) -> None:
        # 64 zero bytes create a >=256-bit zero run with a following 1 bit,
        # which looks like a lead-in; the true page must still be one page.
        payload = b"\x00" * 64 + b"\x80" + bytes(63)
        stream = page_header(4) + segment_header(2, 1, 0x60) \
            + data_packet(payload) + control(0xF5)
        data_bits = [0] * 300 + [1] + bytes_to_bits(stream)
        edges = mfm_encode(data_bits)

        result = framing.decode_transitions(edges, sample_rate=1.0)

        self.assertEqual(len(result.pages), 1)
        self.assertEqual(result.pages[0].raw, stream)
        self.assertTrue(all(p.checksum_ok for p in result.pages[0].packets))
        # The false lead-in sits inside a validated packet span and must not be
        # reported as a discarded page.
        self.assertFalse(any(item["kind"] == "discarded_page"
                             for item in result.unrecoverable))

    def test_exact_header_in_payload_is_a_loud_failure(self) -> None:
        # Deferred false rejection (README "Known false-rejection case"): a
        # payload zero run ending in a 1 bit followed by a complete page header
        # is byte-identical to a real page boundary. Pin the current behaviour:
        # the true page is split and verification FAILS loudly instead of
        # silently accepting a mangled page. If boundary selection is ever
        # fixed, replace this with an assertion that the page survives intact.
        embedded = page_header(7)
        payload = b"\x00" * 64 + b"\x01" + embedded + bytes(range(63))
        stream = page_header(4) + segment_header(2, 1, 0x60) \
            + data_packet(payload) + control(0xF5)
        data_bits = [0] * 300 + [1] + bytes_to_bits(stream)
        edges = mfm_encode(data_bits)

        result = framing.decode_transitions(edges, sample_rate=1.0)

        self.assertEqual([page.page_id for page in result.pages], [4, 7])
        self.assertTrue(all(not page.complete for page in result.pages))
        self.assertNotIn(stream, [page.raw for page in result.pages])
        self.assertTrue(result.unrecoverable)
        self.assertFalse(verify.verify_decode_result(result).passed)


class GapFreeDiscardedPageTest(unittest.TestCase):
    """A damaged page inside a gap-free region must be diagnosed (#1).

    In a continuous signal a damaged page used to be absorbed as the previous
    page's unvalidated tail: neighbours decoded, verification passed, and the
    retained pages round-tripped exactly. The scan for page-like lead-in
    candidates outside validated packet spans must make that impossible.
    """

    @staticmethod
    def _pages() -> list[bytes]:
        return [capture_fixture.raw_page(page_id) for page_id in (1, 2, 3)]

    @staticmethod
    def _edges(raws: list[bytes], leadin_bits: int = 305) -> np.ndarray:
        data_bits: list[int] = []
        for raw in raws:
            data_bits += [0] * leadin_bits + [1] + bytes_to_bits(raw)
        return mfm_encode(data_bits)

    def _decode(self, raws: list[bytes], leadin_bits: int = 305):
        return framing.decode_transitions(self._edges(raws, leadin_bits),
                                          sample_rate=1.0)

    def _assert_discarded(self, result, page_id: int, mismatches: int) -> None:
        losses = [item for item in result.unrecoverable
                  if item["kind"] == "discarded_page"]
        self.assertEqual(len(losses), 1, result.unrecoverable)
        self.assertEqual(losses[0].get("page_id"), page_id)
        self.assertEqual(losses[0]["magic_mismatches"], mismatches)
        self.assertFalse(verify.verify_decode_result(result).passed)
        self.assertTrue(all(encoder.roundtrip_page(page).exact
                            for page in result.pages))

    def test_clean_continuous_pages_have_no_losses(self) -> None:
        result = self._decode(self._pages())
        self.assertEqual([page.page_id for page in result.pages], [1, 2, 3])
        self.assertEqual(result.unrecoverable, [])
        self.assertTrue(verify.verify_decode_result(result).passed)

    def test_marker_and_value_damage_share_diagnostic_budget(self) -> None:
        for wrong_values in (1, 2):
            with self.subTest(wrong_values=wrong_values):
                raws = self._pages()
                raw = bytearray(raws[0])
                for slot in range(1, wrong_values + 1):
                    raw[slot] = 0
                raws[0] = bytes(raw)
                bits = []
                for raw in raws:
                    bits += [0] * 305 + [1] + bytes_to_bits(raw)
                bits[306] = 1
                result = framing.decode_transitions(mfm_encode(bits), 1.0)
                losses = [x for x in result.unrecoverable if x['kind'] == 'discarded_page']
                self.assertEqual(len(losses), 1 if wrong_values == 1 else 0)
                if losses:
                    self.assertEqual(losses[0]['magic_mismatches'], 1)
                    self.assertEqual(losses[0]['marker_mismatches'], 1)
                    self.assertFalse(verify.verify_decode_result(result).passed)

    def test_corrupt_first_magic_is_diagnosed(self) -> None:
        raws = self._pages()
        raws[0] = bytes([0xC4]) + raws[0][1:]
        result = self._decode(raws)
        self.assertEqual([page.page_id for page in result.pages], [2, 3])
        self._assert_discarded(result, page_id=1, mismatches=1)

    def test_corrupt_middle_magic_is_diagnosed(self) -> None:
        raws = self._pages()
        raws[1] = bytes([0xC4]) + raws[1][1:]
        result = self._decode(raws)
        self.assertEqual([page.page_id for page in result.pages], [1, 3])
        self._assert_discarded(result, page_id=2, mismatches=1)

    def test_corrupt_last_magic_is_diagnosed(self) -> None:
        raws = self._pages()
        raws[2] = bytes([0xC4]) + raws[2][1:]
        result = self._decode(raws)
        self.assertEqual([page.page_id for page in result.pages], [1, 2])
        self._assert_discarded(result, page_id=3, mismatches=1)

    def test_one_corrupt_fixed_header_byte_is_diagnosed(self) -> None:
        raws = self._pages()
        damaged = bytearray(raws[1])
        damaged[2] ^= 0x01
        raws[1] = bytes(damaged)
        result = self._decode(raws)
        self.assertEqual([page.page_id for page in result.pages], [1, 3])
        self._assert_discarded(result, page_id=2, mismatches=1)

    def test_two_corrupt_fixed_header_bytes_are_diagnosed(self) -> None:
        raws = self._pages()
        damaged = bytearray(raws[1])
        damaged[2] ^= 0x01
        damaged[3] ^= 0x01
        raws[1] = bytes(damaged)
        result = self._decode(raws)
        self.assertEqual([page.page_id for page in result.pages], [1, 3])
        self._assert_discarded(result, page_id=2, mismatches=2)

    def test_three_corrupt_fixed_header_bytes_are_not_detected(self) -> None:
        # Documented limitation: the scan tolerates at most two damaged fixed
        # bytes, so heavier fixed-byte damage is not diagnosed (and never
        # guessed at). This test pins the limitation rather than claiming it.
        raws = self._pages()
        damaged = bytearray(raws[1])
        damaged[1] ^= 0x01
        damaged[2] ^= 0x01
        damaged[3] ^= 0x01
        raws[1] = bytes(damaged)
        result = self._decode(raws)
        self.assertEqual([page.page_id for page in result.pages], [1, 3])
        self.assertFalse(any(item["kind"] == "discarded_page"
                             for item in result.unrecoverable))

    def test_nearby_leadin_alignment_is_diagnosed(self) -> None:
        raws = self._pages()
        raws[1] = bytes([0xC4]) + raws[1][1:]
        result = self._decode(raws, leadin_bits=300)
        self.assertEqual([page.page_id for page in result.pages], [1, 3])
        self._assert_discarded(result, page_id=2, mismatches=1)


class GapFreeFrontendLossTest(unittest.TestCase):
    """The gap-free loss diagnosis holds through the real front end (#1)."""

    def test_marker_damage_is_diagnosed_in_memory_and_through_frontend(self) -> None:
        from studybox import frontend

        raws = [capture_fixture.raw_page(i) for i in (1, 2, 3)]
        for damaged in range(3):
            for slot in (0, 3, 7):
                with self.subTest(page=damaged, slot=slot):
                    bits = []
                    for index, raw in enumerate(raws):
                        start = len(bits) + 306
                        bits += [0] * 305 + [1] + bytes_to_bits(raw)
                        if index == damaged:
                            bits[start + 9 * slot] = 1
                    edges = mfm_encode(bits)
                    results = [framing.decode_transitions(edges, 1.0)]
                    # One shared MFM grid/polarity; no per-page render seams.
                    for rate in (44100, 48000, 96000):
                        flips = np.zeros(len(bits) * 2 + 20, dtype=np.int64)
                        flips[edges.astype(np.int64)] = 1
                        levels = 1.0 - 2.0 * (np.cumsum(flips) % 2)
                        boundaries = np.round(np.arange(len(flips) + 1)
                                              * rate * capture_fixture.HALF_CELL).astype(int)
                        signal = encoder.bandlimit_pulses(
                            np.diff(np.repeat(levels, np.diff(boundaries))), rate)
                        transitions = frontend.detect_transitions(signal, rate)
                        results.append(framing.decode_transitions(transitions.samples, rate))
                    for result in results:
                        self.assertEqual([p.page_id for p in result.pages],
                                         [i for i in (1, 2, 3) if i != damaged + 1])
                        losses = [x for x in result.unrecoverable
                                  if x['kind'] == 'discarded_page']
                        self.assertEqual(len(losses), 1)
                        self.assertEqual(losses[0]['marker_mismatches'], 1)
                        self.assertEqual(losses[0]['header'], raws[damaged][:8].hex())
                        self.assertFalse(verify.verify_decode_result(result).passed)

    def _damaged_pages(self) -> list[bytes]:
        raws = [capture_fixture.raw_page(page_id) for page_id in (1, 2, 3)]
        raws[1] = bytes([0xC4]) + raws[1][1:]
        return raws

    def _decode(self, raws: list[bytes], sample_rate: int):
        signal = capture_fixture.continuous_pages_signal(raws, sample_rate)
        with tempfile.TemporaryDirectory() as tmp:
            path = capture_fixture.write_signal(
                Path(tmp) / "gapfree.wav", signal, sample_rate)
            return api.decode_file(path, channel=audio_in.DATA_CHANNEL).result

    def _check_damage(self, sample_rate: int) -> None:
        result = self._decode(self._damaged_pages(), sample_rate)
        self.assertEqual([page.page_id for page in result.pages], [1, 3])
        losses = [item for item in result.unrecoverable
                  if item["kind"] == "discarded_page"]
        self.assertEqual(len(losses), 1, result.unrecoverable)
        self.assertEqual(losses[0].get("page_id"), 2)
        # A shared-polarity continuous render must not create artificial gaps.
        self.assertFalse(any(item["kind"] in ("gap", "gap_loss")
                             for item in result.unrecoverable))
        self.assertFalse(verify.verify_decode_result(result).passed)
        self.assertTrue(all(encoder.roundtrip_page(page).exact
                            for page in result.pages))

    def test_damage_diagnosed_at_44100(self) -> None:
        self._check_damage(44100)

    def test_damage_diagnosed_at_48000(self) -> None:
        self._check_damage(48000)

    def test_damage_diagnosed_at_96000(self) -> None:
        self._check_damage(96000)

    def test_clean_continuous_frontend_has_no_losses(self) -> None:
        raws = [capture_fixture.raw_page(page_id) for page_id in (1, 2, 3)]
        result = self._decode(raws, 44100)
        self.assertEqual([page.page_id for page in result.pages], [1, 2, 3])
        self.assertEqual(result.unrecoverable, [])
        self.assertTrue(verify.verify_decode_result(result).passed)


class RecognizedPaddingTest(unittest.TestCase):
    """Only clean containing pages exempt header-shaped type-5 padding."""

    def _decode(self, damage=None, crossing=False):
        near_header = bytes([1, 0xC4, 1, 1, 1, 1])
        if not crossing:
            near_header += bytes([7, 7, 0]) + bytes(40)
        raw = bytearray(page_header(1) + bytes([0xC5, 5, 5])
                        + bytes(64) + near_header + control(0xF5))
        if damage == "checksum":
            raw[7] ^= 1
        elif damage == "structure":
            raw[10] = 4
        elif damage == "incomplete":
            raw[-1] ^= 1
        elif damage == "desync":
            raw[8:8] = b"\xaa"
        bits = [0] * 305 + [1] + bytes_to_bits(bytes(raw))
        return framing.decode_transitions(mfm_encode(bits), 1.0), bytes(raw)

    def test_clean_type5_padding_is_not_a_discarded_page(self) -> None:
        result, raw = self._decode()
        self.assertEqual([p.raw for p in result.pages], [raw])
        self.assertEqual(result.unrecoverable, [])
        self.assertTrue(verify.verify_decode_result(result).passed)
        self.assertTrue(encoder.roundtrip_page(result.pages[0]).exact)

    def test_invalid_containing_page_does_not_exempt_padding(self) -> None:
        for damage in ("checksum", "structure", "incomplete", "desync"):
            with self.subTest(damage=damage):
                result, _ = self._decode(damage)
                self.assertIn("discarded_page", [x['kind'] for x in result.unrecoverable])
                self.assertFalse(verify.verify_decode_result(result).passed)

    def test_padding_exemption_does_not_cross_next_c5(self) -> None:
        result, raw = self._decode(crossing=True)
        self.assertEqual(result.pages[0].raw, raw)
        self.assertTrue(result.pages[0].complete)
        self.assertEqual(result.pages[0].checksum_ok, result.pages[0].checksum_total)
        self.assertIn("discarded_page", [x['kind'] for x in result.unrecoverable])


class SecondsLimitTest(unittest.TestCase):
    """``--seconds`` must bound the transitions that enter the decoder (#10)."""

    def _capture(self, directory: str) -> audio_in.Capture:
        path = capture_fixture.write_capture(
            Path(directory) / "capture.wav", lead_seconds=0.1, tail_seconds=0.1)
        return audio_in.resolve_file(path)

    def test_transitions_do_not_exceed_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = self._capture(tmp)
            sample_rate = int(audio_in.sf.info(str(capture.data_path)).samplerate)
            transitions, rate = framing.extract_transitions(capture, max_seconds=0.25)
            self.assertEqual(rate, sample_rate)
            self.assertGreater(transitions.size, 0)
            self.assertLess(float(transitions.max()), 0.25 * sample_rate)

    def test_nonpositive_seconds_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            capture = self._capture(tmp)
            for value in (0, 0.0, -1.0, float("inf"), float("nan")):
                with self.assertRaises(ValueError):
                    framing.extract_transitions(capture, max_seconds=value)


class ObservedHorizonTest(unittest.TestCase):
    """A page ending mid-byte must not decode invented zero bits (#1)."""

    def test_missing_final_transition_is_unrecoverable(self) -> None:
        stream = page_header(3) + segment_header(2, 1, 0x60) \
            + data_packet(bytes([1, 2, 3])) + control(0xF5)
        data_bits = [0] * 300 + [1] + bytes_to_bits(stream)
        edges = mfm_encode(data_bits)[:-1]  # the tape's final transition is gone

        result = framing.decode_transitions(edges, sample_rate=1.0)

        self.assertEqual(len(result.pages), 1)
        page = result.pages[0]
        self.assertFalse(page.observed_coverage)
        self.assertTrue(
            any(item.get("kind") == "unobserved_eof" for item in result.unrecoverable))
        # The truncated final byte must not be invented from the zero horizon.
        self.assertEqual(page.raw, stream[:-1])


class CaptureEndExtractionTest(unittest.TestCase):
    """The physical end of the recording must survive edge trimming (#3)."""

    def test_no_trailing_silence_preserves_final_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = capture_fixture.write_capture(
                Path(tmp) / "no_tail.wav", page_id=9, sample_rate=96000,
                tail_seconds=0.0)
            capture = audio_in.resolve_file(path)
            info = audio_in.sf.info(str(capture.data_path))

            with warnings.catch_warnings():
                # The clamped power envelope must not warn in the silent lead-in.
                warnings.simplefilter("error", RuntimeWarning)
                transitions, sample_rate = framing.extract_transitions(capture)

            self.assertEqual(sample_rate, 96000)
            # The final transition lies inside the trailing 60 ms that an
            # unconditional trim used to discard: the physical endpoint survives.
            self.assertGreater(float(transitions.max()),
                               info.frames - 0.06 * sample_rate)
            result = framing.decode_transitions(transitions, sample_rate)
            self.assertEqual([page.page_id for page in result.pages], [9])
            self.assertEqual(result.pages[0].raw, capture_fixture.raw_page(9))
            self.assertTrue(result.pages[0].complete)


class UndecodableRegionTest(unittest.TestCase):
    """A lead-in-bearing region that yields no page must not vanish (#1)."""

    def _two_page_transitions(self, corrupt_second: bool) -> np.ndarray:
        first = page_header(1) + segment_header(2, 1, 0x60) \
            + data_packet(bytes(range(128))) + control(0xF5)
        second = page_header(2) + segment_header(2, 1, 0x60) \
            + data_packet(bytes(range(128))) + control(0xF5)
        if corrupt_second:
            second = bytes([0xC4]) + second[1:]  # header no longer a page start
        edges_first = mfm_encode([0] * 300 + [1] + bytes_to_bits(first))
        edges_second = mfm_encode([0] * 300 + [1] + bytes_to_bits(second))
        edges_second = edges_second + (edges_first.max() + 500)  # inter-page gap
        return np.concatenate([edges_first, edges_second])

    def test_undecodable_region_is_reported_as_loss(self) -> None:
        result = framing.decode_transitions(self._two_page_transitions(True),
                                            sample_rate=1.0)
        self.assertEqual([page.page_id for page in result.pages], [1])
        kinds = [item["kind"] for item in result.unrecoverable]
        self.assertIn("undecodable_region", kinds)
        self.assertIn("gap_loss", kinds)
        self.assertNotIn("gap", kinds)
        self.assertFalse(verify.verify_decode_result(result).passed)

    def test_gap_between_decoded_pages_is_benign(self) -> None:
        result = framing.decode_transitions(self._two_page_transitions(False),
                                            sample_rate=1.0)
        self.assertEqual([page.page_id for page in result.pages], [1, 2])
        kinds = [item["kind"] for item in result.unrecoverable]
        self.assertEqual(kinds, ["gap"])
        self.assertTrue(verify.verify_decode_result(result).passed,
                        verify.verify_decode_result(result).render())


if __name__ == "__main__":
    unittest.main()
