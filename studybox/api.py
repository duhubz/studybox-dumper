"""Shared decode/verify helpers used by the CLI and GUI.

These wrap :mod:`studybox.audio_in`, :mod:`studybox.framing`,
:mod:`studybox.container`, and :mod:`studybox.verify` so front-ends do not each
re-implement target resolution and container assembly. Behavior is intentionally
identical to the previous CLI-private helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio_in, container, framing, paths, verify

CaptureLike = str | Path | audio_in.Capture


def capture_inputs(capture: audio_in.Capture) -> list[tuple[str, Path | None]]:
    """The ``(label, path)`` source files a decode depends on."""
    return [("data capture", capture.data_path),
            ("narration", capture.audio_path)]


def require_distinct_outputs(outputs: list[tuple[str, Path | None]],
                             capture: audio_in.Capture) -> None:
    """Reject outputs that alias the capture's sources or each other."""
    paths.require_distinct(outputs, capture_inputs(capture))


@dataclass
class DecodeOutcome:
    """A decode plus the resolved capture it came from.

    Front-ends need both the page records and the capture (to write a
    self-contained container), so :func:`decode_file` returns them together.
    The common :class:`~studybox.framing.DecodeResult` attributes are exposed
    as passthrough properties for convenience.
    """

    result: framing.DecodeResult
    capture: audio_in.Capture

    @property
    def pages(self) -> list[framing.DecodedPage]:
        return self.result.pages

    @property
    def sample_rate(self) -> int:
        return self.result.sample_rate

    @property
    def half_cell(self) -> float:
        return self.result.half_cell

    @property
    def transitions(self) -> int:
        return self.result.transitions

    @property
    def malformed(self) -> int:
        return self.result.malformed

    @property
    def unrecoverable(self) -> list[dict]:
        return self.result.unrecoverable

    @property
    def checksum_ok(self) -> int:
        return self.result.checksum_ok

    @property
    def checksum_total(self) -> int:
        return self.result.checksum_total

    @property
    def checksum_rate(self) -> float:
        return self.result.checksum_rate

    @property
    def data128_ok(self) -> int:
        return self.result.data128_ok

    @property
    def data128_total(self) -> int:
        return self.result.data128_total


def resolve_target(target: CaptureLike, side: str | None = None,
                   channel: int = audio_in.DATA_CHANNEL) -> audio_in.Capture:
    """Resolve a directory side or an explicit capture file to a ``Capture``."""
    if isinstance(target, audio_in.Capture):
        return target
    path = Path(target)
    if path.is_dir():
        return audio_in.resolve_side(path, side or "A")
    return audio_in.resolve_file(path, channel)


def decode_file(target: CaptureLike, side: str | None = None,
                channel: int = audio_in.DATA_CHANNEL,
                seconds: float | None = None,
                window_seconds: float = 30.0) -> DecodeOutcome:
    """Decode a capture (or resolved ``Capture``) into a ``DecodeOutcome``."""
    capture = resolve_target(target, side, channel)
    result = framing.decode_capture(capture, window_seconds=window_seconds,
                                    max_seconds=seconds)
    return DecodeOutcome(result=result, capture=capture)


def verify_file(target: CaptureLike, side: str | None = None,
                channel: int = audio_in.DATA_CHANNEL,
                seconds: float | None = None,
                min_checksum_rate: float = verify.DEFAULT_MIN_CHECKSUM_RATE,
                allow_degraded: bool = False) -> verify.VerifyReport:
    """Run the independent decode gate over a capture file/directory."""
    outcome = decode_file(target, side, channel, seconds)
    return verify.verify_decode_result(outcome.result, min_checksum_rate,
                                       allow_degraded)


def write_studybox(path: str | Path, result: framing.DecodeResult,
                   capture: audio_in.Capture, no_audio: bool = False
                   ) -> tuple[Path, int, int]:
    """Write a self-contained ``.studybox``.

    When narration is present it is streamed to disk in bounded blocks instead
    of being materialised; otherwise ``no_audio`` embeds silence. Returns
    ``(path, page_count, embedded_audio_bytes)``.
    """
    if not result.pages:
        raise ValueError("decode found no pages; cannot write a .studybox")
    # Silence export never touches the narration, but the page bytes still came
    # from the data capture: replacing it would destroy the source recording.
    require_distinct_outputs([("container output", path)], capture)
    pages = [container.Page(page.lead_in_sample, page.audio_sample, page.raw)
             for page in result.pages]
    if capture.has_narration:
        info = audio_in.sf.info(str(capture.audio_path))
        wav_size = container.write_streaming(path, pages, capture,
                                             int(info.samplerate), int(info.frames))
        return Path(path), len(pages), wav_size
    if not no_audio:
        raise ValueError(
            "capture has no narration track; pass no_audio/--no-audio to embed "
            "silence instead of the data signal")
    data_info = audio_in.sf.info(str(capture.data_path))
    wav_size = container.write_streaming(
        path, pages, sample_rate=int(data_info.samplerate),
        frames=int(data_info.frames), silence=True)
    return Path(path), len(pages), wav_size


def decode_to_container(result: framing.DecodeResult,
                        capture: audio_in.Capture,
                        no_audio: bool = False) -> container.StudyBox:
    """Build a self-contained ``.studybox`` from a decode plus its capture.

    A missing narration track is an error unless ``no_audio`` is set, in which
    case a silent track at the data sample rate/length keeps page offsets valid.
    """
    if not result.pages:
        raise ValueError("decode found no pages; cannot build a .studybox")
    if capture.has_narration:
        info = audio_in.sf.info(str(capture.audio_path))
        audio = audio_in.read_channel(capture.audio_path, capture.audio_channel)
        sample_rate = int(info.samplerate)
    elif no_audio:
        data_info = audio_in.sf.info(str(capture.data_path))
        sample_rate = int(data_info.samplerate)
        audio = np.zeros(int(data_info.frames), dtype=np.float32)
    else:
        raise ValueError(
            "capture has no narration track; pass no_audio/--no-audio to embed "
            "silence instead of the data signal")
    pages = [container.Page(page.lead_in_sample, page.audio_sample, page.raw)
             for page in result.pages]
    return container.StudyBox(
        pages=pages, audio=container.wav_bytes(audio, sample_rate))
