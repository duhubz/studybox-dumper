"""Read and write the Mesen2 ``.studybox`` container.

Matches ``Core/NES/Loaders/StudyBoxLoader.cpp``:

* ``STBX`` + uint32 length (4) + uint32 version (0x100)
* repeated ``PAGE`` chunks: uint32 ``pageSize`` (payload + 8), uint32
  ``leadInOffset``, uint32 ``audioOffset``, then ``pageSize - 8`` data bytes.
  Offsets are sample indices into the AUDI audio; they must be monotonic.
* a final ``AUDI`` chunk: uint32 size, uint32 file type (0 = WAV), then the
  embedded WAV file. ``size`` counts the 4-byte type field plus the WAV.
"""

from __future__ import annotations

import io
import struct
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

STREAM_BLOCK = 1 << 20

STBX_MAGIC = b"STBX"
PAGE_MAGIC = b"PAGE"
AUDI_MAGIC = b"AUDI"
VERSION = 0x100
FILE_TYPE_WAV = 0
DEFAULT_MAX_CONTAINER_BYTES = 512 << 20  # guard before reading untrusted files


@dataclass
class Page:
    lead_in_offset: int
    audio_offset: int
    data: bytes = b""

    @property
    def page_id(self) -> int | None:
        return self.data[5] if len(self.data) > 5 else None

    @property
    def valid(self) -> bool:
        return self.audio_offset >= self.lead_in_offset and len(self.data) >= 8


@dataclass
class StudyBox:
    pages: list[Page] = field(default_factory=list)
    audio: bytes = b""            # embedded WAV file
    file_type: int = FILE_TYPE_WAV

    # ------------------------------------------------------------------ read
    @classmethod
    def from_bytes(cls, blob: bytes,
                   max_bytes: int = DEFAULT_MAX_CONTAINER_BYTES) -> StudyBox:
        if len(blob) > max_bytes:
            raise ValueError(
                f"studybox input is {len(blob)} bytes, over the "
                f"{max_bytes}-byte limit")
        if len(blob) < 12 or blob[:4] != STBX_MAGIC:
            raise ValueError("not a studybox file (missing STBX)")
        length, version = struct.unpack_from("<II", blob, 4)
        if length != 4 or version != VERSION:
            raise ValueError(f"unsupported studybox version/length: {version:#x}/{length}")
        offset = 12
        pages: list[Page] = []
        audio = b""
        file_type = FILE_TYPE_WAV
        prev_audio = 0
        prev_lead = 0
        while offset + 4 <= len(blob):
            tag = blob[offset:offset + 4]
            offset += 4
            if tag == PAGE_MAGIC:
                if offset + 12 > len(blob):
                    raise ValueError("truncated PAGE chunk header")
                page_size, lead_in, audio_off = struct.unpack_from("<III", blob, offset)
                offset += 12
                if audio_off < lead_in:
                    raise ValueError("PAGE audio offset precedes lead-in offset")
                if audio_off < prev_audio or lead_in < prev_lead:
                    raise ValueError("PAGE chunks are not in tape order")
                prev_audio, prev_lead = audio_off, lead_in
                payload = page_size - 8
                if payload < 0 or offset + payload > len(blob):
                    raise ValueError("PAGE chunk size out of range")
                pages.append(Page(lead_in, audio_off, blob[offset:offset + payload]))
                offset += payload
            elif tag == AUDI_MAGIC:
                if offset + 8 > len(blob):
                    raise ValueError("truncated AUDI chunk header")
                audio_size, file_type = struct.unpack_from("<II", blob, offset)
                offset += 8
                payload = audio_size - 4
                if payload < 0 or offset + payload > len(blob):
                    raise ValueError("AUDI chunk size out of range")
                audio = blob[offset:offset + payload]
                break
            else:
                raise ValueError(f"unsupported chunk tag {tag!r}")
        if not pages:
            raise ValueError("studybox file has no pages")
        if not audio:
            raise ValueError("studybox file has no AUDI chunk")
        if file_type != FILE_TYPE_WAV:
            raise ValueError(f"unsupported AUDI file type {file_type}")
        frames = wav_frames(audio)
        if frames is None:
            raise ValueError("AUDI payload is not a readable WAV")
        for page in pages:
            if page.audio_offset > frames or page.lead_in_offset > frames:
                raise ValueError("PAGE offset beyond embedded audio frame count")
        return cls(pages=pages, audio=audio, file_type=file_type)

    @classmethod
    def read(cls, path: str | Path,
             max_bytes: int = DEFAULT_MAX_CONTAINER_BYTES) -> StudyBox:
        path = Path(path)
        size = path.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"{path} is {size} bytes, over the {max_bytes}-byte limit")
        return cls.from_bytes(path.read_bytes(), max_bytes=max_bytes)

    # ----------------------------------------------------------------- write
    def to_bytes(self) -> bytes:
        if not self.pages:
            raise ValueError("cannot write a studybox file with no pages")
        chunks: list[bytes] = [STBX_MAGIC, struct.pack("<II", 4, VERSION)]
        chunks.extend(_iter_page_chunks(self.pages))
        chunks.append(AUDI_MAGIC)
        chunks.append(struct.pack("<II", len(self.audio) + 4, self.file_type))
        chunks.append(self.audio)
        return b"".join(chunks)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.to_bytes())
        return path


def _iter_page_chunks(pages: list[Page]):
    """Yield the validated ``PAGE`` chunk bytes for ``pages`` in tape order."""
    prev_audio = 0
    prev_lead = 0
    for page in pages:
        if page.audio_offset < page.lead_in_offset:
            raise ValueError("PAGE audio offset precedes lead-in offset")
        if page.audio_offset < prev_audio or page.lead_in_offset < prev_lead:
            raise ValueError("PAGE chunks must be in tape order")
        prev_audio, prev_lead = page.audio_offset, page.lead_in_offset
        yield PAGE_MAGIC
        yield struct.pack("<III", len(page.data) + 8,
                          page.lead_in_offset, page.audio_offset)
        yield page.data


def write_streaming(path: str | Path, pages: list[Page], capture=None,
                    sample_rate: int = 44100, frames: int = 0,
                    silence: bool = False) -> int:
    """Write a ``.studybox`` without holding the audio in memory.

    Pages are small and passed in memory; the ``AUDI`` WAV is streamed to a
    temporary file in bounded blocks and then copied into the container. With
    ``silence`` set, ``frames`` of zero PCM-16 are written instead of reading
    ``capture`` (for ``--no-audio`` on a long recording). Output is
    byte-identical to ``StudyBox(pages, audio_bytes).to_bytes()`` for the same
    audio. Returns the embedded WAV size in bytes.
    """
    from . import audio_in, paths

    if not pages:
        raise ValueError("cannot write a studybox file with no pages")
    if not silence and (capture is None or not capture.has_narration):
        raise ValueError("capture has no narration track to stream")
    if capture is not None:
        # Direct callers must not replace the capture they are streaming from;
        # api.write_studybox guards its own path too, including silence export.
        paths.require_distinct(
            [("container output", path)],
            [("data capture", capture.data_path),
             ("narration", capture.audio_path)])
    frames = int(frames)
    for page in pages:
        if page.lead_in_offset > frames or page.audio_offset > frames:
            raise ValueError("PAGE offset beyond narration frame count")
    # Exhaust lazy ordering/uint32 serialization checks before any destination
    # is opened. Page payloads are already in memory; this only retains their
    # references and the small serialized chunk headers.
    page_chunks = list(_iter_page_chunks(pages))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        remaining = frames
        with sf.SoundFile(str(wav_path), mode="w", samplerate=int(sample_rate),
                          channels=1, format="WAV", subtype="PCM_16") as out:
            if silence:
                zero = np.zeros(STREAM_BLOCK, dtype=np.float32)
                while remaining > 0:
                    count = min(STREAM_BLOCK, remaining)
                    out.write(zero[:count])
                    remaining -= count
            else:
                for _, block in audio_in.iter_windows(
                        capture.audio_path, capture.audio_channel,
                        window_frames=STREAM_BLOCK, overlap_frames=0):
                    if remaining <= 0:
                        break
                    chunk = block[:remaining]
                    out.write(chunk)
                    remaining -= len(chunk)
        if remaining != 0:
            raise ValueError(f"narration ended {remaining} frame(s) early")
        wav_size = wav_path.stat().st_size
        # Validate the uint32 AUDI length before truncating an existing output.
        audi_header = struct.pack("<II", wav_size + 4, FILE_TYPE_WAV)
        with open(path, "wb") as handle:
            handle.write(STBX_MAGIC)
            handle.write(struct.pack("<II", 4, VERSION))
            for chunk in page_chunks:
                handle.write(chunk)
            handle.write(AUDI_MAGIC)
            handle.write(audi_header)
            with open(wav_path, "rb") as src:
                while True:
                    block = src.read(STREAM_BLOCK)
                    if not block:
                        break
                    handle.write(block)
    return wav_size


def wav_bytes(samples: np.ndarray, sample_rate: int = 44100) -> bytes:
    """Encode mono float samples as a 16-bit PCM WAV (what Mesen's WavReader reads)."""
    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(samples, dtype=np.float32), sample_rate,
             format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def wav_frames(audio: bytes) -> int | None:
    """Return the frame count of an embedded mono PCM-16 WAV, or ``None``.

    Mesen's audio reader expects a plain mono 16-bit PCM WAV. Any other payload
    that libsndfile can merely open (for example a FLAC stream mislabeled with
    ``FILE_TYPE_WAV``) is rejected rather than treated as valid audio.
    """
    if not audio:
        return None
    try:
        info = sf.info(io.BytesIO(audio))
    except Exception:
        return None
    if (info.format != "WAV" or info.subtype != "PCM_16"
            or info.channels != 1):
        return None
    return int(info.frames)
