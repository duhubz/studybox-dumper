"""PyInstaller entry point for the CLI.

PyInstaller executes the entry script as ``__main__`` with no parent package, so
it must use absolute imports rather than ``studybox``'s relative ones.
"""

from studybox.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
