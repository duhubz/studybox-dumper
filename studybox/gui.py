"""Tkinter front-end for the common decode, verify, and merge workflows.

The heavy work runs on a worker thread so the window stays responsive; the
worker posts its result back to the Tk event loop through a queue. The GUI is a
thin shell over :mod:`studybox.api` and :mod:`studybox.merge`; the CLI offers
additional commands and options.
"""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from . import api, audio_in, merge, paths, report, verify


def _window_icon_path() -> Path | None:
    if getattr(sys, "frozen", False):
        candidates: list[Path] = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "icon.png")
        executable_dir = Path(sys.executable).resolve().parent
        candidates.extend([
            executable_dir / "icon.png",
            executable_dir.parent / "Resources" / "icon.png",
            executable_dir.parent / "Frameworks" / "icon.png",
        ])
    else:
        candidates = [Path(__file__).resolve().parents[1]
                      / "packaging" / "icon.png"]
    return next((path for path in candidates if path.is_file()), None)


class ToolTip:
    """Show a short explanation while the pointer rests over a widget."""

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._after_id: str | None = None
        self._window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: tk.Event) -> None:
        self._after_id = self.widget.after(450, self._show)

    def _show(self) -> None:
        self._after_id = None
        if self._window is not None or not self.widget.winfo_exists():
            return
        self._window = tk.Toplevel(self.widget)
        self._window.wm_overrideredirect(True)
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._window.wm_geometry(f"+{x}+{y}")
        tk.Label(self._window, text=self.text, justify="left", anchor="w",
                 background="#fff8c4", relief="solid", borderwidth=1,
                 padx=6, pady=4).pack()

    def _hide(self, _event: tk.Event) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._window is not None:
            self._window.destroy()
            self._window = None


@dataclass(frozen=True)
class WorkerMessage:
    """A worker result and optional verification popup details."""

    kind: str
    text: str
    popup_text: str | None = None
    popup_title: str | None = None


class StudyBoxApp:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._launch_directory = Path.cwd()
        self.root.title("StudyBox Dumper")
        self._icon_image: tk.PhotoImage | None = None
        icon_path = _window_icon_path()
        if icon_path is not None:
            try:
                self._icon_image = tk.PhotoImage(master=root, file=str(icon_path))
                root.iconphoto(True, self._icon_image)
            except tk.TclError:
                self._icon_image = None
        self._messages: queue.Queue[WorkerMessage] = queue.Queue()
        self._worker: threading.Thread | None = None

        self.capture_var = tk.StringVar()
        self.channel_var = tk.StringVar(value=str(audio_in.DATA_CHANNEL))
        self.seconds_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.no_audio_var = tk.BooleanVar(value=False)
        self.write_json_var = tk.BooleanVar(value=False)
        self._suggested_output: str | None = None
        self.merge_base_var = tk.StringVar()
        self.merge_output_var = tk.StringVar()
        self._suggested_merge_output: str | None = None
        self.merge_json_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self._tooltips: list[ToolTip] = []

        self._build()
        self._poll()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        self.notebook = ttk.Notebook(frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        decode_tab = ttk.Frame(self.notebook, padding=8)
        decode_tab.columnconfigure(1, weight=1)
        self.notebook.add(decode_tab, text="Decode")
        self._build_decode_tab(decode_tab)

        merge_tab = ttk.Frame(self.notebook, padding=8)
        merge_tab.columnconfigure(1, weight=1)
        self.notebook.add(merge_tab, text="Merge")
        self._build_merge_tab(merge_tab)

        log_heading = ttk.Frame(frame)
        log_heading.grid(row=1, column=0, sticky="ew", pady=(8, 2))
        ttk.Label(log_heading, text="Dump Log").pack(side="left")
        self.clear_log_button = ttk.Button(
            log_heading, text="Clear Log", command=self._clear_log)
        self.clear_log_button.pack(side="right")

        log_frame = ttk.Frame(frame)
        log_frame.grid(row=2, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=14, width=80, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        self.log_scrollbar = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log.yview)
        self.log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=self.log_scrollbar.set)

        ttk.Label(frame, textvariable=self.status_var, anchor="w").grid(
            row=3, column=0, sticky="ew", pady=(6, 0))

    def _build_decode_tab(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Capture file").grid(row=0, column=0, sticky="w")
        capture_entry = ttk.Entry(frame, textvariable=self.capture_var)
        capture_entry.grid(row=0, column=1, sticky="ew", padx=4)
        capture_entry.bind("<Return>", self._sync_default_output)
        capture_entry.bind("<FocusOut>", self._sync_default_output)
        self.capture_button = ttk.Button(
            frame, text="Browse file...", command=self._browse_capture)
        self.capture_button.grid(
            row=0, column=2, padx=(0, 4))

        ttk.Label(frame, text="Data channel").grid(
            row=1, column=0, sticky="w", pady=4)
        self.channel_box = ttk.Combobox(
            frame, textvariable=self.channel_var, state="readonly", width=4,
            values=[str(index) for index in range(8)])
        self.channel_box.grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(frame, text="Seconds (optional)").grid(
            row=2, column=0, sticky="w")
        self.seconds_entry = ttk.Entry(frame, textvariable=self.seconds_var, width=12)
        self.seconds_entry.grid(row=2, column=1, sticky="w", padx=4)
        self.no_audio_check = ttk.Checkbutton(
            frame, text="Embed silence when narration is missing",
            variable=self.no_audio_var)
        self.no_audio_check.grid(row=2, column=2, columnspan=2, sticky="w")

        ttk.Label(frame, text="Output .studybox").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.output_var).grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=4)
        ttk.Button(frame, text="Browse...", command=self._browse_output).grid(
            row=3, column=3)

        self.write_json_check = ttk.Checkbutton(
            frame, text="Write JSON sidecar", variable=self.write_json_var)
        self.write_json_check.grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, columnspan=4, sticky="ew", pady=8)
        self.decode_button = ttk.Button(
            buttons, text="Decode + Verify", command=self.decode)
        self.decode_button.pack(side="left")

        self._add_tooltip(capture_entry,
                          "Choose an audio capture file. Its filename can be anything.")
        self._add_tooltip(
            self.channel_box,
            "Data channel for an explicit multichannel file; "
            "standard stereo uses channel 1.")
        self._add_tooltip(
            self.seconds_entry,
            "Optional diagnostic limit. The decoder estimates clock and page "
            "boundaries from only this portion, so pages near the cutoff may "
            "be incomplete.")
        self._add_tooltip(self.no_audio_check,
                          "For mono captures without narration, embed silence instead of failing. "
                          "This does not recover narration audio.")
        self._add_tooltip(self.write_json_check,
                          "The JSON sidecar records page, checksum, and loss diagnostics.")
        self._add_tooltip(
            self.decode_button,
            "Decode the capture, save a .studybox, and automatically check the decoded pages.")

    def _build_merge_tab(self, frame: ttk.Frame) -> None:
        ttk.Label(frame, text="Base .studybox").grid(row=0, column=0, sticky="w")
        base_entry = ttk.Entry(frame, textvariable=self.merge_base_var)
        base_entry.grid(row=0, column=1, sticky="ew", padx=4)
        base_entry.bind("<Return>", self._sync_default_merge_output)
        base_entry.bind("<FocusOut>", self._sync_default_merge_output)
        ttk.Button(frame, text="Browse...",
                   command=self._browse_merge_base).grid(row=0, column=2)

        ttk.Label(frame, text="Other recordings").grid(
            row=1, column=0, sticky="nw", pady=4)
        self.merge_others = tk.Listbox(frame, height=4, selectmode="extended")
        self.merge_others.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        other_buttons = ttk.Frame(frame)
        other_buttons.grid(row=1, column=2, sticky="n")
        ttk.Button(other_buttons, text="Add...",
                   command=self._add_merge_others).pack(fill="x")
        ttk.Button(other_buttons, text="Remove",
                   command=self._remove_merge_others).pack(fill="x", pady=(4, 0))

        ttk.Label(frame, text="Output .studybox").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.merge_output_var).grid(
            row=2, column=1, sticky="ew", padx=4)
        ttk.Button(frame, text="Browse...",
                   command=self._browse_merge_output).grid(row=2, column=2)

        ttk.Checkbutton(frame, text="Write provenance JSON",
                        variable=self.merge_json_var).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=3, sticky="ew", pady=8)
        self.merge_button = ttk.Button(
            buttons, text="Merge + Verify", command=self.merge)
        self.merge_button.pack(side="left")

    # ---------------------------------------------------------------- actions
    def _browse_capture(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select a capture",
            filetypes=[("Audio", "*.wav *.flac *.ogg"), ("All files", "*.*")])
        if chosen:
            self._set_capture_target(chosen)

    def _set_capture_target(self, chosen: str) -> None:
        self.capture_var.set(chosen)
        self._sync_default_output()

    def _sync_default_output(self, _event: tk.Event | None = None) -> None:
        capture_target = self.capture_var.get().strip()
        if not capture_target:
            return
        current_output = self.output_var.get()
        if not current_output or current_output == self._suggested_output:
            suggested = str(self._default_output_path(capture_target))
            self.output_var.set(suggested)
            self._suggested_output = suggested

    def _default_output_path(self, capture_target: str | Path) -> Path:
        filename = Path(capture_target).with_suffix(".studybox").name
        return self._launch_directory / "output" / filename

    @staticmethod
    def _require_capture_file(target: str) -> None:
        if Path(target).is_dir():
            raise ValueError("select a capture file, not a folder")

    def _add_tooltip(self, widget: tk.Widget, text: str) -> None:
        self._tooltips.append(ToolTip(widget, text))

    def _browse_output(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Save .studybox", defaultextension=".studybox",
            filetypes=[("StudyBox", "*.studybox")])
        if chosen:
            self._suggested_output = None
            self.output_var.set(chosen)

    def _browse_merge_base(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select the base .studybox",
            filetypes=[("StudyBox", "*.studybox"), ("All files", "*.*")])
        if chosen:
            self.merge_base_var.set(chosen)
            self._sync_default_merge_output()

    def _sync_default_merge_output(self, _event: tk.Event | None = None) -> None:
        base_target = self.merge_base_var.get().strip()
        if not base_target:
            return
        current_output = self.merge_output_var.get()
        if not current_output or current_output == self._suggested_merge_output:
            suggested = str(self._default_merge_output_path(base_target))
            self.merge_output_var.set(suggested)
            self._suggested_merge_output = suggested

    def _default_merge_output_path(self, base_target: str | Path) -> Path:
        filename = f"{Path(base_target).stem}-merged.studybox"
        return self._launch_directory / "output" / filename

    def _add_merge_others(self) -> None:
        chosen = filedialog.askopenfilenames(
            title="Select other recordings of the same program",
            filetypes=[("StudyBox", "*.studybox"), ("All files", "*.*")])
        existing = set(self.merge_others.get(0, "end"))
        for path in chosen:
            if path not in existing:
                self.merge_others.insert("end", path)

    def _remove_merge_others(self) -> None:
        for index in reversed(self.merge_others.curselection()):
            self.merge_others.delete(index)

    def _browse_merge_output(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Save merged .studybox", defaultextension=".studybox",
            filetypes=[("StudyBox", "*.studybox")])
        if chosen:
            self._suggested_merge_output = None
            self.merge_output_var.set(chosen)

    @staticmethod
    def _parse_channel(value: str) -> int:
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"data channel must be an integer, got "
                             f"{value!r}") from exc

    def decode(self) -> None:
        # Snapshot the Tk variables on this (the Tk) thread: a job must not pick
        # up a filename the user types while it runs.
        capture_target = self.capture_var.get()
        channel_text = self.channel_var.get()
        seconds_text = self.seconds_var.get().strip()
        no_audio = bool(self.no_audio_var.get())
        output = self.output_var.get()
        if not output and capture_target:
            output = str(self._default_output_path(capture_target))
            self.output_var.set(output)
            self._suggested_output = output
        write_json = bool(self.write_json_var.get())

        def work() -> WorkerMessage:
            self._require_capture_file(capture_target)
            try:
                seconds = float(seconds_text) if seconds_text else None
            except ValueError as exc:
                raise ValueError("seconds must be a number") from exc
            outcome = api.decode_file(
                capture_target, channel=self._parse_channel(channel_text),
                seconds=seconds)
            json_path = Path(output).with_suffix(".json") if write_json else None
            # Preflight both destinations before either is written.
            api.require_distinct_outputs(
                [("container output", output), ("JSON sidecar", json_path)],
                outcome.capture)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            written, pages, audio_bytes = api.write_studybox(
                output, outcome.result, outcome.capture, no_audio=no_audio)
            payload = report.decode_report(outcome.result)
            if json_path is not None:
                report.write_json(json_path, payload)
            text = report.render_text(payload)
            verification = verify.verify_decode_result(outcome.result)
            if verification.passed:
                kind = "decode-pass"
                popup_title = "Dump looks good"
                summary = "Dump looks good: all strict verification checks passed."
                popup_text = (f"All strict verification checks passed.\n\n"
                              f"Saved to:\n{written}")
            else:
                kind = "decode-fail"
                popup_title = "Dump has problems"
                summary = ("Dump has problems: strict verification failed; "
                           "see the checks below.")
                popup_text = ("Strict verification found problems. The decoded file was "
                              f"still saved to:\n{written}\n\nSee the log for details.")
            suffix = f" and {json_path}" if json_path is not None else ""
            log_text = (f"decoded {pages} page(s), wrote {written}{suffix} "
                        f"({audio_bytes} audio bytes)\n{text}\n"
                        f"{verification.render()}\n{summary}")
            return WorkerMessage(kind, log_text, popup_text, popup_title)

        self._run_async(work, "decode")

    def merge(self) -> None:
        base = self.merge_base_var.get()
        others = list(self.merge_others.get(0, "end"))
        output = self.merge_output_var.get()
        if not output and base:
            output = str(self._default_merge_output_path(base))
            self.merge_output_var.set(output)
            self._suggested_merge_output = output
        write_json = bool(self.merge_json_var.get())

        def work() -> WorkerMessage:
            if not base or not output:
                raise ValueError("select a base container and an output path")
            if not others:
                raise ValueError("add at least one other recording to merge")
            json_path = Path(output).with_suffix(".json") if write_json else None
            paths.require_distinct(
                [("container output", output), ("JSON sidecar", json_path)],
                [("base container", base)]
                + [(f"other container {index}", path)
                   for index, path in enumerate(others, start=1)])
            outcome = merge.merge_files(base, others)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            if json_path is not None:
                report.write_json(json_path, outcome.provenance)
            outcome.box.write(output)
            open_conflicts = outcome.provenance["open_conflicts"]
            summary = (f"merged {len(outcome.box.pages)} page(s), "
                       f"{outcome.provenance['repaired']} repaired, "
                       f"{open_conflicts} open conflict(s)")
            suffix = f" and {json_path}" if json_path is not None else ""
            verification = verify.verify_studybox(outcome.box)
            if verification.passed and not open_conflicts:
                kind = "merge-pass"
                popup_title = "Merged dump looks good"
                result_summary = ("Merged dump looks good: all strict verification "
                                  "checks passed.")
                popup_text = (f"All strict verification checks passed.\n\n"
                              f"Saved to:\n{output}")
            else:
                kind = "merge-fail"
                popup_title = "Merged dump needs review"
                problems = []
                if not verification.passed:
                    problems.append("Strict verification found problems.")
                if open_conflicts:
                    problems.append(f"{open_conflicts} unresolved merge conflict(s) remain.")
                result_summary = "Merged dump needs review: " + " ".join(problems)
                popup_text = (f"{' '.join(problems)} The merged file was saved to:\n"
                              f"{output}\n\nSee the log and provenance report for details.")
            log_text = (f"{summary}, wrote {output}{suffix}\n"
                        f"{verification.render()}\n{result_summary}")
            return WorkerMessage(kind, log_text, popup_text, popup_title)

        self._run_async(work, "merge")

    # ---------------------------------------------------------------- workers
    def _run_async(self, work, label: str) -> None:
        if self._worker is not None and self._worker.is_alive():
            self.status_var.set("Busy with another task...")
            return
        self.status_var.set(f"{label.capitalize()}...")
        self._worker = threading.Thread(target=self._guarded, args=(work,),
                                        daemon=True)
        self._worker.start()

    def _guarded(self, work) -> None:
        try:
            result = work()
            if isinstance(result, WorkerMessage):
                message = result
            else:
                message = WorkerMessage("ok", result)
            self._messages.put(message)
        except Exception as exc:  # surface any failure in the log
            self._messages.put(
                WorkerMessage("error", f"{type(exc).__name__}: {exc}"))

    def _poll(self) -> None:
        try:
            while True:
                message = self._messages.get_nowait()
                self._log(message.text)
                if message.kind == "error":
                    self.status_var.set("Failed")
                    messagebox.showerror("StudyBox Dumper", message.text,
                                         parent=self.root)
                elif message.kind.endswith("-pass"):
                    self.status_var.set(message.popup_title or "Done")
                    messagebox.showinfo(message.popup_title or "Success", message.popup_text,
                                        parent=self.root)
                elif message.kind.endswith("-fail"):
                    self.status_var.set(message.popup_title or "Needs review")
                    messagebox.showwarning(message.popup_title or "Needs review",
                                           message.popup_text,
                                           parent=self.root)
                else:
                    self.status_var.set("Done")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI."""
    root = tk.Tk()
    StudyBoxApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
