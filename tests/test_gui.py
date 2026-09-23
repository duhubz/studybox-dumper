"""GUI smoke test (skipped when tkinter or a display is unavailable)."""

from __future__ import annotations

import tempfile
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
            self.assertIsInstance(app.log_scrollbar, ttk.Scrollbar)
            self.assertIsInstance(app.clear_log_button, ttk.Button)
            self.assertFalse(app.write_json_var.get())
            self.assertFalse(app.no_audio_var.get())
            self.assertGreaterEqual(len(app._tooltips), 6)
            self.assertIsNotNone(app.decode_button)
            self.assertEqual(app.decode_button.cget("text"), "Decode + Verify")
            self.assertIsInstance(app.notebook, ttk.Notebook)
            self.assertEqual(app.notebook.index("end"), 2)
            self.assertIsNotNone(app.merge_button)
            self.assertEqual(app.merge_button.cget("text"), "Merge + Verify")
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
        app._launch_directory = Path.cwd()
        app.capture_var = _FakeVar("/tmp/first.wav")
        app.channel_var = _FakeVar("1")
        app.seconds_var = _FakeVar("")
        app.output_var = _FakeVar("")
        app.no_audio_var = _FakeVar(False)
        app.write_json_var = _FakeVar(False)
        app._suggested_output = None
        app._suggested_merge_output = None
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
        app._launch_directory = Path("/tmp/gui-app")
        with mock.patch.object(gui.filedialog, "askopenfilename",
                               return_value="/tmp/side-a capture.wav"):
            app._browse_capture()

        self.assertEqual(app.capture_var.get(), "/tmp/side-a capture.wav")
        self.assertEqual(app.output_var.get(),
                         "/tmp/gui-app/output/side-a capture.studybox")

        app._set_capture_target("/tmp/side-b.wav")
        self.assertEqual(app.output_var.get(), "/tmp/gui-app/output/side-b.studybox")

        app.output_var.set("/tmp/custom.studybox")
        app._set_capture_target("/tmp/side-c.wav")
        self.assertEqual(app.output_var.get(), "/tmp/custom.studybox")

    def test_merge_base_picker_updates_default_output(self) -> None:
        from studybox import gui

        app = object.__new__(gui.StudyBoxApp)
        app._launch_directory = Path("/tmp/gui-app")
        app.merge_base_var = _FakeVar("")
        app.merge_output_var = _FakeVar("")
        app._suggested_merge_output = None

        with mock.patch.object(gui.filedialog, "askopenfilename",
                               side_effect=["/tmp/side-a.studybox",
                                            "/tmp/side-b.studybox"]):
            app._browse_merge_base()
            self.assertEqual(app.merge_output_var.get(),
                             "/tmp/gui-app/output/side-a-merged.studybox")

            app._browse_merge_base()
        self.assertEqual(app.merge_output_var.get(),
                         "/tmp/gui-app/output/side-b-merged.studybox")

        app.merge_output_var.set("/tmp/custom-merged.studybox")
        app.merge_base_var.set("/tmp/side-c.studybox")
        app._sync_default_merge_output()
        self.assertEqual(app.merge_output_var.get(), "/tmp/custom-merged.studybox")

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
            calls["outcome"] = self._fake_outcome()
            return calls["outcome"]

        def fake_write(out, result, capture, no_audio=False):
            calls["output"] = out
            calls["no_audio"] = no_audio
            return (out, 3, 4)

        with tempfile.TemporaryDirectory() as working_dir:
            app._launch_directory = Path(working_dir)
            with mock.patch.object(gui.api, "decode_file", side_effect=fake_decode), \
                    mock.patch.object(gui.api, "require_distinct_outputs"), \
                    mock.patch.object(gui.api, "write_studybox", side_effect=fake_write), \
                    mock.patch.object(gui.report, "render_text", return_value="TEXT"), \
                    mock.patch.object(gui.report, "decode_report", return_value={}), \
                    mock.patch.object(gui.verify, "verify_decode_result",
                                      return_value=mock.Mock(
                                          passed=True, render=lambda: "VERIFY")) as verify:
                app.decode()
                app.capture_var.value = "/tmp/second.wav"  # typed while running
                app.channel_var.value = "7"
                app.seconds_var.value = "2"
                app.no_audio_var.value = False
                app.output_var.value = "/tmp/other.studybox"
                response = self.captured["work"]()

            self.assertEqual(calls["output"],
                             str(Path(working_dir) / "output" / "first.studybox"))
            self.assertTrue((Path(working_dir) / "output").is_dir())
            verify.assert_called_once_with(calls["outcome"].result)

        self.assertEqual(calls["decode"], ("/tmp/first.wav", 1, 12.5))
        self.assertTrue(calls["no_audio"])
        self.assertEqual(response.kind, "decode-pass")
        self.assertIn("Dump looks good", response.text)
        self.assertIn("VERIFY", response.text)
        self.assertIn("TEXT", response.text)
        self.assertIn("Dump looks good", response.text.splitlines()[-1])
        self.assertIn("All strict verification checks passed", response.popup_text)

    def test_decode_writes_json_when_selected(self) -> None:
        from studybox import gui

        app = self._app()
        app.output_var = _FakeVar("/tmp/first.studybox")
        app.write_json_var = _FakeVar(True)
        payload = {"tool": "studybox.decode"}

        with mock.patch.object(gui.api, "decode_file",
                               side_effect=lambda *a, **k: self._fake_outcome()), \
                mock.patch.object(gui.api, "require_distinct_outputs") as guard, \
                mock.patch.object(gui.api, "write_studybox",
                                  return_value=("/tmp/first.studybox", 1, 2)), \
                mock.patch.object(gui.report, "decode_report", return_value=payload), \
                mock.patch.object(gui.report, "render_text", return_value="TEXT"), \
                mock.patch.object(gui.verify, "verify_decode_result",
                                  return_value=mock.Mock(
                                      passed=False, render=lambda: "VERIFY")), \
                 mock.patch.object(gui.report, "write_json") as write_json:
            app.decode()
            response = self.captured["work"]()

        write_json.assert_called_once_with(Path("/tmp/first.json"), payload)
        self.assertEqual(response.kind, "decode-fail")
        self.assertIn("Dump has problems", response.text)
        self.assertIn("still saved to", response.popup_text)
        # Both destinations were preflighted together before any write.
        outputs = guard.call_args.args[0]
        self.assertEqual([label for label, _ in outputs],
                         ["container output", "JSON sidecar"])

    def test_decode_verdict_shows_a_plain_language_popup(self) -> None:
        from studybox import gui

        cases = [
            ("decode-pass", "showinfo", "Dump looks good", "Dump looks good"),
            ("decode-fail", "showwarning", "Dump has problems", "Dump has problems"),
            ("merge-pass", "showinfo", "Merged dump looks good",
             "Merged dump looks good"),
            ("merge-fail", "showwarning", "Merged dump needs review",
             "Merged dump needs review"),
        ]
        for kind, dialog, title, status in cases:
            with self.subTest(kind=kind):
                app = object.__new__(gui.StudyBoxApp)
                app._messages = gui.queue.Queue()
                app._messages.put(gui.WorkerMessage(
                    kind, "log", "popup details", title))
                app._log = mock.Mock()
                app.root = mock.Mock()
                app.status_var = _FakeVar()

                with mock.patch.object(gui.messagebox, dialog) as show_dialog:
                    app._poll()

                show_dialog.assert_called_once_with(
                    title, "popup details", parent=app.root)
                self.assertEqual(app.status_var.get(), status)

    def test_finished_task_status_is_capitalized(self) -> None:
        from studybox import gui

        app = object.__new__(gui.StudyBoxApp)
        app._messages = gui.queue.Queue()
        app._messages.put(gui.WorkerMessage("ok", "task output"))
        app._log = mock.Mock()
        app.root = mock.Mock()
        app.status_var = _FakeVar()

        app._poll()

        self.assertEqual(app.status_var.get(), "Done")

    def test_clear_log_empties_text_widget(self) -> None:
        from studybox import gui

        app = object.__new__(gui.StudyBoxApp)
        app.log = mock.Mock()

        app._clear_log()

        self.assertEqual(app.log.configure.call_args_list, [
            mock.call(state="normal"), mock.call(state="disabled")])
        app.log.delete.assert_called_once_with("1.0", "end")

    def test_merge_snapshots_inputs(self) -> None:
        from types import SimpleNamespace

        from studybox import gui

        app = object.__new__(gui.StudyBoxApp)
        app.merge_base_var = _FakeVar("/tmp/base.studybox")
        app.merge_others = _FakeListbox(["/tmp/other.studybox"])
        app.merge_output_var = _FakeVar("/tmp/merged.studybox")
        app._suggested_merge_output = None
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
                                  return_value=mock.Mock(
                                      passed=True, render=lambda: "VERIFY")):
            app.merge()
            app.merge_base_var.value = "/tmp/late-base.studybox"
            app.merge_others.items = ["/tmp/late.studybox"]
            app.merge_output_var.value = "/tmp/late.studybox"
            response = self.captured["work"]()

        self.assertEqual(calls["merge"], ("/tmp/base.studybox", ["/tmp/other.studybox"]))
        self.assertEqual(self.captured["label"], "merge")
        self.assertEqual(response.kind, "merge-pass")
        self.assertIn("1 repaired", response.text)
        self.assertIn("VERIFY", response.text)
        self.assertIn("Merged dump looks good", response.text)

    def test_merge_open_conflicts_require_review_even_if_verify_passes(self) -> None:
        from types import SimpleNamespace

        from studybox import gui

        app = object.__new__(gui.StudyBoxApp)
        app.merge_base_var = _FakeVar("/tmp/base.studybox")
        app.merge_others = _FakeListbox(["/tmp/other.studybox"])
        app.merge_output_var = _FakeVar("/tmp/merged.studybox")
        app._suggested_merge_output = None
        app.merge_json_var = _FakeVar(False)
        self.captured = {}

        def fake_run(work, label):
            self.captured["work"] = work
            self.captured["label"] = label

        app._run_async = fake_run
        box = SimpleNamespace(pages=[1], write=lambda path: Path(path))
        outcome = SimpleNamespace(box=box, provenance={
            "repaired": 0, "open_conflicts": 1, "page_count": 1})

        with mock.patch.object(gui.merge, "merge_files", return_value=outcome), \
                mock.patch.object(gui.paths, "require_distinct"), \
                mock.patch.object(gui.verify, "verify_studybox",
                                  return_value=mock.Mock(
                                      passed=True, render=lambda: "verify: PASS")):
            app.merge()
            response = self.captured["work"]()

        self.assertEqual(response.kind, "merge-fail")
        self.assertEqual(response.popup_title, "Merged dump needs review")
        self.assertIn("1 unresolved merge conflict(s) remain", response.popup_text)

    def test_merge_requires_an_other_recording(self) -> None:
        from studybox import gui

        app = object.__new__(gui.StudyBoxApp)
        app.merge_base_var = _FakeVar("/tmp/base.studybox")
        app.merge_others = _FakeListbox([])
        app.merge_output_var = _FakeVar("/tmp/merged.studybox")
        app._suggested_merge_output = None
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
