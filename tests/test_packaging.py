"""Release-archive layout regression: symlinks must survive zipping (#4).

The release workflow packages a POSIX PyInstaller onedir build with ``zip -y``
because plain ``zip`` dereferences file symlinks and does not preserve
directory links; extracted archives must then be smoke-tested. This test
packages a small file+directory symlink fixture with the same commands and
extracts it with ``unzip``, which restores stored symlinks. It is skipped
where the tools are absent (Windows and minimal environments); the release
workflow exercises it on Linux and macOS.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


def _run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


@unittest.skipUnless(shutil.which("zip") and shutil.which("unzip"),
                     "zip/unzip unavailable")
class SymlinkArchiveTest(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        bundle = root / "dist" / "studybox"
        version = bundle / "Frameworks" / "Python.framework" / "Versions" / "3.12"
        version.mkdir(parents=True)
        launcher = bundle / "studybox"
        launcher.write_text("#!/bin/sh\necho studybox\n", encoding="utf-8")
        launcher.chmod(0o755)
        (bundle / "libsample.so.1.0").write_bytes(b"\x7fELF")
        (bundle / "libsample.so.1").symlink_to("libsample.so.1.0")
        (bundle / "Frameworks" / "Python.framework" / "Versions" / "Current"
         ).symlink_to("3.12")
        return bundle

    def test_zip_y_preserves_file_and_directory_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture(root)
            _run(["zip", "-y", "-r", "bundle.zip", "dist"], root)
            extracted = root / "extracted"
            extracted.mkdir()
            _run(["unzip", "-q", str(root / "bundle.zip"), "-d", str(extracted)],
                 root)

            bundle = extracted / "dist" / "studybox"
            file_link = bundle / "libsample.so.1"
            dir_link = (bundle / "Frameworks" / "Python.framework" / "Versions"
                        / "Current")
            self.assertTrue(file_link.is_symlink())
            self.assertEqual(os.readlink(file_link), "libsample.so.1.0")
            self.assertTrue(dir_link.is_symlink())
            self.assertEqual(os.readlink(dir_link), "3.12")
            self.assertTrue((bundle / "studybox").stat().st_mode & stat.S_IXUSR)


if __name__ == "__main__":
    unittest.main()
