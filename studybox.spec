# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: onedir builds for the CLI and the Tkinter GUI.

Build both entry points into a single ``studybox`` directory (one COLLECT), and
add a macOS ``.app`` bundle for the GUI. Intended for PyInstaller 6.x:

    pip install ".[build]"
    pyinstaller studybox.spec
"""

import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# pyinstaller-hooks-contrib's soundfile hook ships libsndfile from
# ``_soundfile_data``; collect it explicitly too so a build without that hook
# still finds the library. ``soundfile`` itself is a module, so the data lives
# in the sibling ``_soundfile_data`` package.
soundfile_datas = (collect_data_files("soundfile")
                   or collect_data_files("_soundfile_data"))
soundfile_binaries = (collect_dynamic_libs("soundfile")
                      or collect_dynamic_libs("_soundfile_data"))

datas = [("LICENSE", "."), ("NOTICE", ".")] + soundfile_datas
gui_datas = datas + [("packaging/icon.png", ".")]
binaries = soundfile_binaries + collect_dynamic_libs("scipy")

cli = Analysis(
    ["packaging/pyi_cli.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    excludes=["tkinter"],
    noarchive=False,
)
cli_pyz = PYZ(cli.pure)

gui = Analysis(
    ["packaging/pyi_gui.py"],
    pathex=["."],
    binaries=binaries,
    datas=gui_datas,
    hiddenimports=["tkinter"],
    excludes=[],
    noarchive=False,
)
gui_pyz = PYZ(gui.pure)

# The GUI gets platform-native icons; the CLI keeps its normal executable icon.
exe_suffix = ".exe" if sys.platform == "win32" else ""
gui_exe_icon = "packaging/icon.ico" if sys.platform == "win32" else None
cli_exe = EXE(
    cli_pyz,
    cli.scripts,
    [],
    exclude_binaries=True,
    name=f"studybox{exe_suffix}",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
gui_exe = EXE(
    gui_pyz,
    gui.scripts,
    [],
    exclude_binaries=True,
    name=f"studybox-gui{exe_suffix}",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=gui_exe_icon,
    disable_windowed_traceback=True,
)

COLLECT(
    cli_exe,
    cli.binaries,
    cli.datas,
    gui_exe,
    gui.binaries,
    gui.datas,
    strip=False,
    upx=False,
    name="studybox",
)

if sys.platform == "darwin":
    # ``gui_exe`` excludes binaries/data, so the .app must be given the GUI's
    # collected runtime TOCs explicitly; otherwise the bundle only contains the
    # executable and its pure-Python modules (#7).
    BUNDLE(
        gui_exe,
        gui.binaries,
        gui.datas,
        name="StudyBox Dumper.app",
        icon="packaging/icon.icns",
        bundle_identifier="org.studybox.dumper",
        info_plist={"NSHighResolutionCapable": True},
    )
