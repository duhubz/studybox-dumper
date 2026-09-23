"""Output/input path-alias protection shared by writers and front-ends.

Writing a container or a JSON sidecar over one of its own inputs destroys the
recording (or the report it was built from). These helpers reject aliases --
identical resolved paths, symlinks, and existing hardlinks -- before any
destination is opened. The policy lives here so the CLI, GUI, and streaming
writer cannot drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path

PathLike = str | os.PathLike


def _resolve(path: PathLike) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def aliases(first: PathLike, second: PathLike) -> bool:
    """True when two paths name the same file, including hardlinks/symlinks."""
    first_path, second_path = Path(first), Path(second)
    try:
        if first_path.exists() and second_path.exists() and first_path.samefile(second_path):
            return True
    except OSError:
        pass
    return _resolve(first_path) == _resolve(second_path)


def require_distinct(outputs: list[tuple[str, PathLike | None]],
                     inputs: list[tuple[str, PathLike | None]]) -> None:
    """Reject outputs that alias an input or another output.

    ``outputs`` and ``inputs`` are ``(label, path)`` pairs; ``None`` paths are
    skipped. Raises ``ValueError`` naming both sides of the collision.
    """
    output_pairs = [(label, Path(path)) for label, path in outputs
                    if path is not None]
    for label, output in output_pairs:
        for input_label, source in inputs:
            if source is None:
                continue
            if aliases(output, source):
                raise ValueError(
                    f"refusing to write {label} {output}: it is the same file "
                    f"as {input_label} {source}")
    for index, (label, output) in enumerate(output_pairs):
        for other_label, other in output_pairs[index + 1:]:
            if aliases(output, other):
                raise ValueError(
                    f"refusing to write {label} {output}: it is the same file "
                    f"as {other_label} {other}")
