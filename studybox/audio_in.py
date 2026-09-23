"""Capture input: locate and stream the data (channel 2) and audio (channel 1).

StudyBox tapes are stereo: the left/narration track carries audio and the
right/data track carries the 4800 bps MFM signal. Paired per-channel
extractions are also accepted.

When a capture is given as a directory, one common label convention is
recognised: trimmed (``casan``/``casbn``) and untrimmed (``casa``/``casb``)
stereo sides. Explicit file paths and channels are supported for captures that
do not follow it. Mixdowns (``-channel1,2``) are never data-bearing and are
skipped.

FLAC/WAV/OGG are read through :mod:`soundfile` (libsndfile). Long sides are read
in windows so a multi-hour 24-bit file never has to be held in memory at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

MIXDOWN_MARKER = "-channel1,2"
AUDIO_SUFFIXES = (".flac", ".wav", ".ogg")
DATA_CHANNEL = 1
AUDIO_CHANNEL = 0
SIDE_STEMS = {"a": ("casan", "casa"), "b": ("casbn", "casb")}


@dataclass(frozen=True)
class Capture:
    """A resolved stereo/paired capture for one side of a tape.

    ``audio_path`` is ``None`` when no narration track could be found; callers
    must treat that explicitly rather than falling back to the data signal.
    """

    name: str
    data_path: Path
    data_channel: int
    audio_path: Path | None
    audio_channel: int | None
    trimmed: bool

    @property
    def has_narration(self) -> bool:
        return self.audio_path is not None

    def info(self) -> sf._SoundFileInfo:  # type: ignore[name-defined]
        return sf.info(str(self.data_path))

    def describe(self) -> dict:
        data_info = sf.info(str(self.data_path))
        audio: dict | None = None
        if self.audio_path is not None:
            audio_info = sf.info(str(self.audio_path))
            audio = {
                "path": str(self.audio_path),
                "channel": self.audio_channel,
                "sample_rate": audio_info.samplerate,
                "channels": audio_info.channels,
                "frames": audio_info.frames,
                "subtype": audio_info.subtype,
            }
        return {
            "name": self.name,
            "trimmed": self.trimmed,
            "data": {
                "path": str(self.data_path),
                "channel": self.data_channel,
                "sample_rate": data_info.samplerate,
                "channels": data_info.channels,
                "frames": data_info.frames,
                "subtype": data_info.subtype,
            },
            "audio": audio,
        }


def _first_existing(directory: Path, stem: str) -> Path | None:
    for suffix in AUDIO_SUFFIXES:
        path = directory / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def _is_mixdown(path: Path) -> bool:
    return MIXDOWN_MARKER in path.name


def resolve_side(directory: str | Path, side: str) -> Capture:
    """Resolve the best capture pair for side ``A``/``B`` of a labelled dump.

    Precedence is trimmed (``casan``/``casbn``) then untrimmed
    (``casa``/``casb``). A stereo file routes data=right/audio=left; a paired
    ``-channel2``/``-channel1`` extraction routes each as mono.

    Paired files must share a sample rate and time origin, because data-track
    sample offsets are copied unchanged into the embedded narration WAV. A rate
    mismatch raises rather than silently shifting playback timing.

    This convention is a convenience for one common naming scheme; for any other
    layout pass an explicit file and data channel to :func:`resolve_file`.
    """
    directory = Path(directory)
    stems = SIDE_STEMS.get(side.lower())
    if stems is None:
        raise ValueError(f"side must be 'A' or 'B', not {side!r}")

    mono_stereo: tuple[str, Path, int] | None = None
    for stem in stems:
        stereo = _first_existing(directory, stem)
        if stereo is not None and not _is_mixdown(stereo):
            info = sf.info(str(stereo))
            if info.channels >= 2:
                return Capture(stem, stereo, DATA_CHANNEL, stereo, AUDIO_CHANNEL,
                               trimmed=stem.endswith("n"))
            if mono_stereo is None:
                mono_stereo = (stem, stereo, info.channels)

    for stem in stems:
        data = _first_existing(directory, f"{stem}-channel2")
        if data is not None:
            audio = _first_existing(directory, f"{stem}-channel1")
            if audio is not None:
                data_rate = sf.info(str(data)).samplerate
                audio_rate = sf.info(str(audio)).samplerate
                if data_rate != audio_rate:
                    raise ValueError(
                        f"paired data/narration sample rates differ "
                        f"({data_rate} vs {audio_rate} Hz); resample both to a "
                        f"shared rate with a shared time origin before decoding")
            return Capture(stem, data, 0, audio, 0 if audio is not None else None,
                           trimmed=stem.endswith("n"))

    if mono_stereo is not None:
        stem, path, channels = mono_stereo
        raise ValueError(
            f"{path} is named as stereo side {stem} but has only "
            f"{channels} channel(s); use a stereo file or paired channel files")
    raise FileNotFoundError(f"no data-bearing capture for side {side} in {directory}")


def resolve_file(path: str | Path, data_channel: int = DATA_CHANNEL) -> Capture:
    """Resolve an explicit capture file, selecting ``data_channel`` for data.

    For a multi-channel file the narration is the channel opposite the selected
    data channel (so ``--data-channel 0`` routes narration to channel 1 and vice
    versa). A single-channel file has no narration and is represented as such.
    An out-of-range ``data_channel`` raises instead of silently reading channel
    0.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    info = sf.info(str(path))
    if not 0 <= data_channel < info.channels:
        raise ValueError(
            f"data channel {data_channel} is out of range for a "
            f"{info.channels}-channel file")
    if info.channels < 2:
        return Capture(path.stem, path, data_channel, None, None, trimmed=False)
    narration = 1 - data_channel if data_channel in (0, 1) else AUDIO_CHANNEL
    return Capture(path.stem, path, data_channel, path, narration, trimmed=False)


def read_channel(path: str | Path, channel: int = DATA_CHANNEL,
                 start: int = 0, frames: int = -1) -> np.ndarray:
    """Read one channel as float32 mono; falls back to channel 0 if out of range."""
    with sf.SoundFile(str(path)) as handle:
        channel = channel if 0 <= channel < handle.channels else 0
        handle.seek(max(0, start))
        block = handle.read(frames, dtype="float32", always_2d=True)
        return np.ascontiguousarray(block[:, channel])


def iter_windows(path: str | Path, channel: int = DATA_CHANNEL,
                 window_frames: int = 1 << 20, overlap_frames: int = 0
                 ) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start_frame, samples)`` windows for a long capture."""
    with sf.SoundFile(str(path)) as handle:
        channel = channel if 0 <= channel < handle.channels else 0
        step = window_frames - overlap_frames
        if step <= 0:
            raise ValueError("overlap_frames must be smaller than window_frames")
        start = 0
        while start < handle.frames:
            handle.seek(start)
            block = handle.read(window_frames, dtype="float32", always_2d=True)
            if block.size == 0:
                break
            yield start, np.ascontiguousarray(block[:, channel])
            if len(block) < window_frames:
                break
            start += step
