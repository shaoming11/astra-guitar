"""Rudimentary start/stop recording window.

Two buttons: start recording, stop recording. On stop, the take runs through
the full pipeline (pitch -> fretting path -> tab) and three files land in the
output directory: the ASCII tab, the full robotab/1.0 JSON, and the compact
STRING-FRET timing file (see tabtext.render_string_tab) meant for a robot
controller to read directly.

    python -m tunefinder ui
    python -m tunefinder ui --tuning drop_d --outdir takes
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .config import PipelineConfig
from .pipeline import run
from .tabtext import render_string_tab
from .transcribe import StreamRecorder


class TunefinderUI:
    def __init__(self, cfg: PipelineConfig, outdir: Path, quantize: Optional[int] = 16):
        import tkinter as tk

        self.cfg = cfg
        self.outdir = Path(outdir)
        self.quantize = quantize
        self.recorder = StreamRecorder(sr=cfg.transcribe.sr)
        self._start_time = 0.0
        self._timer_job = None

        self.root = tk.Tk()
        self.root.title("tunefinder")
        self.root.geometry("520x420")
        self.root.minsize(420, 340)

        self.status = tk.StringVar(value="Ready. Press Start and hum.")
        tk.Label(self.root, textvariable=self.status, font=("Helvetica", 13)).pack(pady=(16, 4))

        self.timer_var = tk.StringVar(value="00:00")
        tk.Label(self.root, textvariable=self.timer_var, font=("Helvetica", 26)).pack(pady=4)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=12)
        self.start_btn = tk.Button(
            btn_frame, text="● Start Recording", width=18,
            command=self.start_recording,
        )
        self.start_btn.grid(row=0, column=0, padx=6)
        self.stop_btn = tk.Button(
            btn_frame, text="■ Stop Recording", width=18,
            command=self.stop_recording, state="disabled",
        )
        self.stop_btn.grid(row=0, column=1, padx=6)

        out_frame = tk.Frame(self.root)
        out_frame.pack(fill="both", expand=True, padx=12, pady=(8, 12))
        self.output = tk.Text(out_frame, height=14, wrap="none", state="disabled")
        self.output.pack(fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- helpers

    def _set_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", text)
        self.output.configure(state="disabled")

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._start_time
        m, s = divmod(int(elapsed), 60)
        self.timer_var.set(f"{m:02d}:{s:02d}")
        self._timer_job = self.root.after(200, self._tick)

    # ------------------------------------------------------------- actions

    def start_recording(self) -> None:
        try:
            self.recorder.start()
        except Exception as exc:
            self.status.set("Could not open the microphone.")
            self._set_output(f"{type(exc).__name__}: {exc}")
            return
        self._start_time = time.monotonic()
        self.status.set("● Recording -- hum your melody")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._tick()

    def stop_recording(self) -> None:
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
        self.stop_btn.configure(state="disabled")
        self.status.set("Processing...")
        self.root.update_idletasks()
        audio = self.recorder.stop()
        # Transcription runs off the UI thread so the window stays responsive;
        # results are marshalled back with root.after, the only thread-safe
        # way to touch Tkinter widgets from outside the main loop.
        threading.Thread(target=self._process, args=(audio,), daemon=True).start()

    def _process(self, audio: np.ndarray) -> None:
        sr = self.cfg.transcribe.sr
        if audio.size < int(sr * 0.3):
            self.root.after(0, self._process_failed,
                             "Recording too short -- hum for at least half a second.")
            return
        try:
            doc, arranged, tr = run((audio, sr), self.cfg, quantize=self.quantize)
        except Exception as exc:
            self.root.after(0, self._process_failed, f"{type(exc).__name__}: {exc}")
            return
        self.root.after(0, self._process_done, doc)

    def _process_failed(self, message: str) -> None:
        self.status.set("Error -- ready to try again.")
        self._set_output(message)
        self.start_btn.configure(state="normal")

    def _process_done(self, doc: dict) -> None:
        self.start_btn.configure(state="normal")
        n_notes = len(doc.get("notes", []))
        if n_notes == 0:
            self.status.set("No notes detected -- ready to try again.")
            self._set_output("No notes detected. Try humming louder, longer, or closer to the mic.")
            return

        self.outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        stab_path = self.outdir / f"hum-{stamp}.stab"
        json_path = self.outdir / f"hum-{stamp}.json"
        tab_path = self.outdir / f"hum-{stamp}.tab"

        stab_text = render_string_tab(doc)
        stab_path.write_text(stab_text + "\n")
        json_path.write_text(json.dumps(doc, indent=2))
        tab_path.write_text(doc["tab"] + "\n")

        self.status.set(f"Done -- {n_notes} notes. Ready to record again.")
        self._set_output(
            f"{doc['tab']}\n\n"
            f"wrote:\n  {stab_path}\n  {json_path}\n  {tab_path}\n\n"
            f"--- {stab_path.name} ---\n{stab_text}\n"
        )

    def _on_close(self) -> None:
        if self.recorder.is_recording:
            self.recorder.stop()
        self.root.destroy()

    def run_loop(self) -> None:
        self.root.mainloop()


def launch(cfg: Optional[PipelineConfig] = None, outdir: str = "recordings",
           quantize: Optional[int] = 16) -> None:
    cfg = cfg or PipelineConfig()
    ui = TunefinderUI(cfg, Path(outdir), quantize=quantize)
    ui.run_loop()


if __name__ == "__main__":
    launch()
