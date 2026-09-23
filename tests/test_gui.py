"""GUI smoke test (skipped when tkinter or a display is unavailable)."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock


def _tkinter_importable() -> bool:
    try:
        import tkinter  # noqa: F401
    except Exception:
        return False
    return True


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


class _FakeVar:
    """Stand-in for a Tk variable (only ``get`` is used by the app)."""

    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class _FakeListbox:
    """Stand-in for the merge listbox (only ``get(0, "end")`` is used)."""

    def __init__(self, items: list[str]) -> None:
        self.items = list(items)

    def get(self, first: int, last: str | None = None) -> list[str]:
        return list(self.items[first:])


@unittest.skipUnless(_display_available(), "no display or tkinter unavailable")
class GuiTest(unittest.TestCase):
    def test_app_builds(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        from studybox import gui

        root = tk.Tk()
        try:
            app = gui.StudyBoxApp(root)
            self.assertEqual(app.channel_var.get(), "1")
            self.assertIsNotNone(app._icon_image)
            self.assertIsInstance(app.capture_button, ttk.Button)
            self.assertIsInstance(app.channel_box, ttk.Combobox)
            self.assertEqual(str(app.channel_box.cget("state")), "readonly")
            self.assertIsInstance(app.seconds_entry, ttk.Entry)
            self.assertIsInstance(app.no_audio_check, ttk.Checkbutton)
            self.assertFalse(app.write_json_var.get())
            self.assertFalse(app.no_audio_var.get())
            self.assertGreaterEqual(len(app._tooltips), 6)
            self.assertIsNotNone(app.decode_button)
            self.assertIsNotNone(app.verify_button)
            self.assertIsInstance(app.notebook, ttk.Notebook)
            self.assertEqual(app.notebook.index("end"), 2)
            self.assertIsNotNone(app.merge_button)
            self.assertIsNotNone(app.merge_verify_button)
            self.assertFalse(app.merge_json_var.get())
        finally:
            root.destroy()

    def test_tooltip_window_shows_and_hides(self) -> None:
        import tkinter as tk

        from studybox import gui

        root = tk.Tk()
        try:
            app = gui.StudyBoxApp(root)
            root.update_idletasks()
            tooltip = app._tooltips[0]

            tooltip._show()
            self.assertIsNotNone(tooltip._window)
            self.assertTrue(tooltip._window.winfo_exists())
            tooltip._hide(None)

            self.assertIsNone(tooltip._window)
        finally:
            root.destroy()


@unittest.skipUnless(_tkinter_importable(), "tkinter unavailable")
class GuiSnapshotTest(unittest.TestCase):
    """A worker must use the inputs captured when the job was started (#8)."""

    def _app(self):
        from studybox import gui

        app = object.__new__(gui.StudyBoxApp)
        app.capture_var = _FakeVar("/tmp/first.wav")
        app.channel_var = _FakeVar("1")
        app.seconds_var = _FakeVar("")
        app.output_var = _FakeVar("")
        app.no_audio_var = _FakeVar(False)
        app.write_json_var = _FakeVar(False)
        self.captured: dict = {}

        def fake_run(work, label):
            self.captured["work"] = work
            self.captured["label"] = label

        app._run_async = fake_run
        return app

    @staticmethod
    def _fake_outcome():
        from types import SimpleNamespace

        return SimpleNamespace(result=object(), capture=object())

    def test_file_picker_sets_file_target_and_default_output(self) -> None:
        from studybox import gui

        app = self._app()
        with mock.patch.object(gui.filedialog, "askopenfilename",
                               return_value="/tmp/side-a capture.wav"):
            app._browse_capture()

        self.assertEqual(app.capture_var.get(), "/tmp/side-a capture.wav")
        self.assertEqual(app.output_var.get(), "/tmp/side-a capture.studybox")

    def test_decode_rejects_folder_input(self) -> None:
        from studybox import gui

        app = self._app()
        app.capture_var.value = "/tmp/capture-folder"
        with mock.patch.object(gui.Path, "is_dir", return_value=True):
            app.decode()
            with self.assertRaisesRegex(ValueError, "select a capture file"):
                self.captured["work"]()

    def test_decode_snapshots_inputs(self) -> None:
        from studybox import gui

        app = self._app()
        app.seconds_var.value = "12.5"
        app.no_audio_var.value = True
        calls: dict = {}

        def fake_decode(target, channel=None, seconds=None):
            calls["decode"] = (target, channel, seconds)
            return self._fake_outcome()

        def fake_write(out, result, capture, no_audio=False):
            calls["output"] = out
            calls["no_audio"] = no_audio
            return (out, 3, 4)

        with mock.patch.object(gui.api, "decode_file", side_effect=fake_decode), \
                mock.patch.object(gui.api, "require_distinct_outputs"), \
                mock.patch.object(gui.api, "write_studybox", side_effect=fake_write), \
                mock.patch.object(gui.report, "render_text", return_value="TEXT"), \
                mock.patch.object(gui.report, "decode_report", return_value={}):
            app.decode()
            app.capture_var.value = "/tmp/second.wav"      # typed while running
            app.channel_var.value = "7"
            app.seconds_var.value = "2"
            app.no_audio_var.value = False
            app.output_var.value = "/tmp/other.studybox"
            text = self.captured["work"]()

        self.assertEqual(calls["decode"], ("/tmp/first.wav", 1, 12.5))
        self.assertEqual(calls["output"], "/tmp/first.studybox")
        self.assertTrue(calls["no_audio"])
        self.assertIn("TEXT", text)

    def test_decode_writes_json_when_selected(self) -> None:
        from studybox import gui

        app = self._app()
        app.write_json_var = _FakeVar(True)
        payload = {"tool": "studybox.decode"}

        with mock.patch.object(gui.api, "decode_file",
                               side_effect=lambda *a, **k: self._fake_outcome()), \
                mock.patch.object(gui.api, "require_distinct_outputs") as guard, \
                mock.patch.object(gui.api, "write_studybox",
                                  return_value=("/tmp/first.studybox", 1, 2)), \
                mock.patch.object(gui.report, "decode_report", return_value=payload), \
                mock.patch.object(gui.report, "render_text", return_value="TEXT"), \
                mock.patch.object(gui.report, "write_json") as write_json:
            app.decode()
            self.captured["work"]()

        write_json.assert_called_once_with(Path("/tmp/first.json"), payload)
        # Both destinations were preflighted together before any write.
        outputs = guard.call_args.args[0]
        self.assertEqual([label for label, _ in outputs],
                         ["container output", "JSON sidecar"])

    def test_verify_snapshots_inputs(self) -> None:
        from studybox import gui

        app = self._app()
        app.seconds_var.value = "7"
        calls: dict = {}

        def fake_decode(target, channel=None, seconds=None):
            calls["decode"] = (target, channel, seconds)
            return self._fake_outcome()

        with mock.patch.object(gui.api, "decode_file", side_effect=fake_decode), \
                mock.patch.object(gui.verify, "verify_decode_result",
                                  return_value=mock.Mock(render=lambda: "VERIFY")):
            app.verify()
            app.capture_var.value = "/tmp/second.wav"
            app.channel_var.value = "7"
            app.seconds_var.value = ""
            text = self.captured["work"]()

        self.assertEqual(calls["decode"], ("/tmp/first.wav", 1, 7.0))
        self.assertIn("VERIFY", text)

    def test_merge_snapshots_inputs(self) -> None:
        from types import SimpleNamespace

        from studybox import gui

        app = object.__new__(gui.StudyBoxApp)
        app.merge_base_var = _FakeVar("/tmp/base.studybox")
        app.merge_others = _FakeListbox(["/tmp/other.studybox"])
        app.merge_output_var = _FakeVar("/tmp/merged.studybox")
        app.merge_json_var = _FakeVar(False)
        self.captured = {}

        def fake_run(work, label):
            self.captured["work"] = work
            self.captured["label"] = label

        app._run_async = fake_run
        calls: dict = {}

        def fake_merge(base, others, **kwargs):
            calls["merge"] = (base, list(others))
            box = SimpleNamespace(pages=[1, 2], write=lambda path: Path(path))
            return SimpleNamespace(box=box, provenance={
                "repaired": 1, "open_conflicts": 0, "page_count": 2})

        with mock.patch.object(gui.merge, "merge_files", side_effect=fake_merge), \
                mock.patch.object(gui.paths, "require_distinct"), \
                mock.patch.object(gui.verify, "verify_studybox",
                                  return_value=mock.Mock(render=lambda: "VERIFY")):
            app.merge()
            app.merge_base_var.value = "/tmp/late-base.studybox"
            app.merge_others.items = ["/tmp/late.studybox"]
            app.merge_output_var.value = "/tmp/late.studybox"
            text = self.captured["work"]()

        self.assertEqual(calls["merge"], ("/tmp/base.studybox", ["/tmp/other.studybox"]))
        self.assertEqual(self.captured["label"], "merge")
        self.assertIn("1 repaired", text)
        self.assertIn("VERIFY", text)

    def test_merge_requires_an_other_recording(self) -> None:
        from studybox import gui

        app = object.__new__(gui.StudyBoxApp)
        app.merge_base_var = _FakeVar("/tmp/base.studybox")
        app.merge_others = _FakeListbox([])
        app.merge_output_var = _FakeVar("/tmp/merged.studybox")
        app.merge_json_var = _FakeVar(False)
        self.captured = {}

        def fake_run(work, label):
            self.captured["work"] = work
            self.captured["label"] = label

        app._run_async = fake_run
        app.merge()

        with self.assertRaisesRegex(ValueError, "at least one other recording"):
            self.captured["work"]()


if __name__ == "__main__":
    unittest.main()
