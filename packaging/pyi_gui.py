"""PyInstaller entry point for the GUI (absolute imports only)."""

from studybox.gui import main

if __name__ == "__main__":
    raise SystemExit(main())
