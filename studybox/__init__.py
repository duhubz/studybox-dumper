"""StudyBox cassette decoder.

Converts StudyBox cassette captures (stereo WAV/FLAC, narration on the left
channel and 4800 bps MFM data on the right) into Mesen2-playable ``.studybox``
containers.

Layering:

- :mod:`studybox.audio_in`  - find and stream the data/audio channels.
- :mod:`studybox.frontend`  - analog front-end: transitions from the data track.
- :mod:`studybox.clock`     - half-cell estimation and MFM clock recovery.
- :mod:`studybox.framing`   - lead-in detection, byte framing, packets/checksums.
- :mod:`studybox.container` - ``.studybox`` read/write matching Mesen2.
"""

__version__ = "0.2.2"

from .container import Page, StudyBox

__all__ = ["Page", "StudyBox", "__version__"]
