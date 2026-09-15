"""Full desktop front end for the tunefinder pipeline.

The window exposes the same operations as the command line interface:
microphone humming, local media/YouTube transcription, and single-note melody mode,
instrument and timing controls, preview rendering, demos, and JSON validation.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .cli import DEMO_TUNE, validate_document
from .config import CostWeights, GuitarSpec, PipelineConfig, TranscribeSpec, TUNINGS, Tuning
from .pipeline import run
from .synth import render_tab_audio, synth_notes, write_wav
from .tabtext import render_events, render_string_tab
from .transcribe import StreamRecorder


class TunefinderUI:
    def __init__(self, cfg: PipelineConfig, outdir: Path, quantize: Optional[int] = 16):
        import tkinter as tk

        self.cfg = cfg
        self.outdir = Path(outdir)
        self.quantize = quantize
        self.recorder = StreamRecorder(sr=cfg.transcribe.sr)
        self._record_cfg: Optional[PipelineConfig] = None
        self._start_time = 0.0
        self._timer_job = None

        self.root = tk.Tk()
        self.root.title("tunefinder")
        self.root.geometry("900x760")
        self.root.minsize(720, 620)

        self.status = tk.StringVar(value="Ready. Press Start and hum, or choose a video.")
        tk.Label(self.root, textvariable=self.status, font=("Helvetica", 13)).pack(pady=(12, 2))
        self.timer_var = tk.StringVar(value="00:00")
        tk.Label(self.root, textvariable=self.timer_var, font=("Helvetica", 24)).pack(pady=2)

        self._build_capture_controls(tk)
        self._build_settings(tk)
        self._build_output_controls(tk)

        out_frame = tk.Frame(self.root)
        out_frame.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        self.output = tk.Text(out_frame, height=12, wrap="none", state="disabled")
        self.output.pack(fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- construction

    def _build_capture_controls(self, tk) -> None:
        capture = tk.LabelFrame(self.root, text="Sources")
        capture.pack(fill="x", padx=12, pady=(4, 6))

        self.start_btn = tk.Button(
            capture, text="● Start Humming", width=17, command=self.start_recording,
        )
        self.start_btn.grid(row=0, column=0, padx=6, pady=7)
        self.stop_btn = tk.Button(
            capture, text="■ Stop", width=12, command=self.stop_recording, state="disabled",
        )
        self.stop_btn.grid(row=0, column=1, padx=6, pady=7)
        self.video_btn = tk.Button(
            capture, text="Choose Media...", width=17, command=self.choose_video,
        )
        self.video_btn.grid(row=0, column=2, padx=6, pady=7)
        self.url_entry = tk.Entry(capture)
        self.url_entry.insert(0, "YouTube URL")
        self.url_entry.grid(row=0, column=3, sticky="ew", padx=(0, 6), pady=7)
        self.url_entry.bind("<FocusIn>", self._clear_url_placeholder)
        self.url_btn = tk.Button(
            capture, text="Transcribe URL", width=16, command=self.transcribe_url,
        )
        self.url_btn.grid(row=0, column=4, padx=(0, 6), pady=7)
        capture.columnconfigure(3, weight=1)

        self.demo_melody_btn = tk.Button(
            capture, text="Demo Melody", width=14,
            command=self.start_demo,
        )
        self.demo_melody_btn.grid(row=1, column=0, padx=6, pady=(0, 7))
        self.validate_btn = tk.Button(
            capture, text="Validate Notes JSON...", width=17, command=self.validate_json,
        )
        self.validate_btn.grid(row=1, column=1, padx=6, pady=(0, 7))

    def _build_settings(self, tk) -> None:
        settings = tk.LabelFrame(self.root, text="Pipeline settings")
        settings.pack(fill="x", padx=12, pady=6)

        self.tuning_var = tk.StringVar(value=self.cfg.guitar.tuning.name)
        self.capo_var = tk.StringVar(value=str(self.cfg.guitar.capo))
        self.max_fret_var = tk.StringVar(value=str(self.cfg.guitar.max_fret))
        self.max_span_var = tk.StringVar(value=str(self.cfg.guitar.max_span))
        self.quantize_var = tk.StringVar(value=str(self.quantize or 0))
        self.min_note_var = tk.StringVar(value=str(self.cfg.transcribe.min_note_seconds))
        self.min_confidence_var = tk.StringVar(value=str(self.cfg.transcribe.min_confidence))
        self.voice_low_var = tk.StringVar(value=str(self.cfg.transcribe.voice_low_hz))
        self.voice_high_var = tk.StringVar(value=str(self.cfg.transcribe.voice_high_hz))
        self.noise_strength_var = tk.StringVar(
            value=str(self.cfg.transcribe.noise_reduction_strength)
        )
        self.gate_strength_var = tk.StringVar(value=str(self.cfg.transcribe.voice_gate_strength))

        self.allow_open_var = tk.BooleanVar(value=self.cfg.guitar.allow_open_strings)
        self.prefer_low_var = tk.BooleanVar(value=False)
        self.minimise_shifts_var = tk.BooleanVar(value=False)
        self.voice_isolation_var = tk.BooleanVar(value=True)
        self.preview_var = tk.BooleanVar(value=False)
        self.save_demo_var = tk.BooleanVar(value=False)
        self.events_var = tk.BooleanVar(value=False)

        instrument = tk.LabelFrame(settings, text="Instrument")
        instrument.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        tk.Label(instrument, text="Tuning").grid(row=0, column=0, sticky="w", padx=5, pady=3)
        tk.OptionMenu(instrument, self.tuning_var, *sorted(TUNINGS)).grid(
            row=0, column=1, sticky="ew", padx=5, pady=3
        )
        self._entry(instrument, tk, "Capo", self.capo_var, 1, 0)
        self._entry(instrument, tk, "Max fret", self.max_fret_var, 2, 0)
        self._entry(instrument, tk, "Max span", self.max_span_var, 3, 0)
        tk.Checkbutton(
            instrument, text="Allow open strings", variable=self.allow_open_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=2)
        tk.Checkbutton(
            instrument, text="Prefer low frets", variable=self.prefer_low_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=2)
        tk.Checkbutton(
            instrument, text="Minimise hand shifts", variable=self.minimise_shifts_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=2)

        transcription = tk.LabelFrame(settings, text="Transcription")
        transcription.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        tk.Label(
            transcription, text="Single-note melody mode", fg="#225522",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=3)
        self._entry(transcription, tk, "Quantize 1/N (0=off)", self.quantize_var, 1, 0)
        self._entry(transcription, tk, "Min note seconds", self.min_note_var, 2, 0)
        self._entry(transcription, tk, "Min confidence", self.min_confidence_var, 3, 0)
        tk.Checkbutton(
            transcription, text="Foreground-clean mic / vocals",
            variable=self.voice_isolation_var,
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=2)

        voice = tk.LabelFrame(settings, text="Voice cleanup")
        voice.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)
        self._entry(voice, tk, "Low Hz", self.voice_low_var, 0, 0)
        self._entry(voice, tk, "High Hz", self.voice_high_var, 1, 0)
        self._entry(voice, tk, "Noise reduction", self.noise_strength_var, 2, 0)
        self._entry(voice, tk, "Gate strength", self.gate_strength_var, 3, 0)
        tk.Label(
            voice, text="Used automatically for mic input.", fg="#555555",
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 2))

        for col in range(3):
            settings.columnconfigure(col, weight=1)

    @staticmethod
    def _entry(parent, tk, label: str, variable, row: int, column: int) -> None:
        tk.Label(parent, text=label).grid(row=row, column=column * 2, sticky="w", padx=(5, 2), pady=3)
        tk.Entry(parent, textvariable=variable, width=9).grid(
            row=row, column=column * 2 + 1, sticky="ew", padx=(0, 5), pady=3
        )

    def _build_output_controls(self, tk) -> None:
        output = tk.LabelFrame(self.root, text="Output")
        output.pack(fill="x", padx=12, pady=6)
        self.outdir_var = tk.StringVar(value=str(self.outdir))
        tk.Label(output, text="Folder").grid(row=0, column=0, padx=5, pady=6)
        tk.Entry(output, textvariable=self.outdir_var).grid(
            row=0, column=1, sticky="ew", padx=2, pady=6
        )
        self.outdir_btn = tk.Button(output, text="Browse...", command=self.choose_outdir)
        self.outdir_btn.grid(row=0, column=2, padx=5, pady=6)
        tk.Checkbutton(
            output, text="Render WAV preview", variable=self.preview_var,
        ).grid(row=0, column=3, padx=5, pady=6)
        tk.Checkbutton(
            output, text="Save demo source WAV", variable=self.save_demo_var,
        ).grid(row=0, column=4, padx=5, pady=6)
        tk.Checkbutton(
            output, text="Show event table", variable=self.events_var,
        ).grid(row=0, column=5, padx=5, pady=6)
        output.columnconfigure(1, weight=1)

        self._job_controls = [
            self.video_btn, self.url_btn, self.url_entry, self.demo_melody_btn,
            self.validate_btn, self.outdir_btn,
        ]

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

    def _set_job_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self._job_controls:
            widget.configure(state=state)

    def _read_config(self) -> PipelineConfig:
        def integer(variable, label):
            try:
                return int(variable.get())
            except ValueError as exc:
                raise ValueError(f"{label} must be an integer") from exc

        def decimal(variable, label):
            try:
                return float(variable.get())
            except ValueError as exc:
                raise ValueError(f"{label} must be a number") from exc

        tuning = Tuning.get(self.tuning_var.get())
        guitar = GuitarSpec(
            tuning=tuning,
            capo=integer(self.capo_var, "Capo"),
            max_fret=integer(self.max_fret_var, "Max fret"),
            max_span=integer(self.max_span_var, "Max span"),
            allow_open_strings=self.allow_open_var.get(),
        )
        weights = CostWeights()
        if self.prefer_low_var.get():
            weights.fret_height *= 2.0
        if self.minimise_shifts_var.get():
            weights.hand_shift *= 2.0
        transcribe_spec = TranscribeSpec(
            sr=self.cfg.transcribe.sr,
            hop_length=512,
            frame_length=2048,
            pyin_thresholds=48,
            min_note_seconds=decimal(self.min_note_var, "Min note seconds"),
            min_confidence=decimal(self.min_confidence_var, "Min confidence"),
            voice_low_hz=decimal(self.voice_low_var, "Voice low Hz"),
            voice_high_hz=decimal(self.voice_high_var, "Voice high Hz"),
            noise_reduction_strength=decimal(self.noise_strength_var, "Noise reduction"),
            voice_gate_strength=decimal(self.gate_strength_var, "Gate strength"),
        )
        return PipelineConfig(guitar=guitar, weights=weights, transcribe=transcribe_spec)

    def _read_output_options(self) -> dict:
        return {
            "outdir": Path(self.outdir_var.get()).expanduser(),
            "preview": self.preview_var.get(),
            "save_demo": self.save_demo_var.get(),
            "events": self.events_var.get(),
            "quantize": self._quantize(),
            "voice_isolation": self.voice_isolation_var.get(),
        }

    def _config_error(self, exc: Exception) -> None:
        self.status.set("Settings error -- ready to try again.")
        self._set_output(f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------- source actions

    def _clear_url_placeholder(self, _event=None) -> None:
        if self.url_entry.get() == "YouTube URL":
            self.url_entry.delete(0, "end")

    def choose_video(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Choose a video or audio file",
            filetypes=[
                ("Media", "*.mp4 *.mov *.m4v *.webm *.mkv *.avi *.wav *.mp3 *.m4a *.flac"),
                ("All files", "*"),
            ],
        )
        if path:
            self._start_source_processing(path)

    def choose_outdir(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.outdir_var.set(path)

    def transcribe_url(self) -> None:
        url = self.url_entry.get().strip()
        if not url or url == "YouTube URL":
            self.status.set("Paste a YouTube URL first.")
            return
        self._start_source_processing(url)

    def _start_source_processing(self, source: str) -> None:
        if self.recorder.is_recording:
            self.status.set("Stop the microphone recording before opening a source.")
            return
        try:
            cfg = self._read_config()
            options = self._read_output_options()
        except Exception as exc:  # noqa: BLE001 - show form errors in the UI
            self._config_error(exc)
            return
        self._set_job_controls(False)
        self.start_btn.configure(state="disabled")
        self.status.set("Loading media and processing...")
        self.root.update_idletasks()
        threading.Thread(
            target=self._process_source, args=(source, cfg, options), daemon=True
        ).start()

    def _process_source(self, source: str, cfg: PipelineConfig, options: dict) -> None:
        try:
            doc, arranged, _ = run(
                source, cfg, quantize=options["quantize"],
                voice_isolation=options["voice_isolation"],
                force_single_notes=True,
            )
        except Exception as exc:  # noqa: BLE001 - surface decoder errors in the UI
            self.root.after(0, self._process_failed, f"{type(exc).__name__}: {exc}")
            return
        self.root.after(0, self._process_done, doc, arranged, cfg, "media", options)

    # ------------------------------------------------------------- microphone/demo actions

    def start_recording(self) -> None:
        try:
            self._record_cfg = self._read_config()
            self._read_output_options()
            self.recorder.start()
        except Exception as exc:  # noqa: BLE001 - includes device and form errors
            self.status.set("Could not start recording.")
            self._set_output(f"{type(exc).__name__}: {exc}")
            return
        self._start_time = time.monotonic()
        self.status.set("● Recording -- hum your melody")
        self._set_job_controls(False)
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._tick()

    def stop_recording(self) -> None:
        if self._timer_job is not None:
            self.root.after_cancel(self._timer_job)
            self._timer_job = None
        self.stop_btn.configure(state="disabled")
        self.status.set("Cleaning audio and processing...")
        self.root.update_idletasks()
        audio = self.recorder.stop()
        cfg = self._record_cfg or self.cfg
        try:
            options = self._read_output_options()
        except Exception as exc:  # noqa: BLE001 - show edits made while recording
            self._process_failed(f"{type(exc).__name__}: {exc}")
            return
        threading.Thread(
            target=self._process_recording, args=(audio, cfg, options), daemon=True
        ).start()

    def _process_recording(self, audio: np.ndarray, cfg: PipelineConfig, options: dict) -> None:
        sr = cfg.transcribe.sr
        if audio.size < int(sr * 0.3):
            self.root.after(
                0, self._process_failed,
                "Recording too short -- hum for at least half a second.",
            )
            return
        try:
            doc, arranged, _ = run(
                (audio, sr), cfg, quantize=options["quantize"],
                voice_isolation=options["voice_isolation"],
                force_single_notes=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._process_failed, f"{type(exc).__name__}: {exc}")
            return
        doc["source"]["file"] = "microphone"
        self.root.after(0, self._process_done, doc, arranged, cfg, "hum", options, audio)

    def start_demo(self) -> None:
        try:
            cfg = self._read_config()
            options = self._read_output_options()
        except Exception as exc:  # noqa: BLE001
            self._config_error(exc)
            return
        self._set_job_controls(False)
        self.start_btn.configure(state="disabled")
        self.status.set("Rendering demo and processing...")
        threading.Thread(
            target=self._process_demo, args=(cfg, options), daemon=True
        ).start()

    def _process_demo(self, cfg: PipelineConfig, options: dict) -> None:
        try:
            sr = cfg.transcribe.sr
            y = synth_notes(DEMO_TUNE, sr=sr)
            doc, arranged, _ = run(
                (y, sr), cfg, quantize=options["quantize"],
                force_single_notes=True,
            )
            doc["source"]["file"] = "demo:melody"
        except Exception as exc:  # noqa: BLE001
            self.root.after(0, self._process_failed, f"{type(exc).__name__}: {exc}")
            return
        self.root.after(
            0, self._process_done, doc, arranged, cfg, "demo-melody", options, y,
        )

    def _quantize(self) -> Optional[int]:
        try:
            value = int(self.quantize_var.get())
        except ValueError as exc:
            raise ValueError("Quantize must be an integer") from exc
        if value < 0:
            raise ValueError("Quantize must be 0 or greater")
        return value or None

    # ------------------------------------------------------------- output / validation

    def _process_failed(self, message: str) -> None:
        self.status.set("Error -- ready to try again.")
        self._set_output(message)
        self.start_btn.configure(state="normal")
        self._set_job_controls(True)

    def _process_done(
        self, doc: dict, arranged: list, cfg: PipelineConfig,
        prefix: str, options: dict, source_audio: Optional[np.ndarray] = None,
    ) -> None:
        self.start_btn.configure(state="normal")
        self._set_job_controls(True)
        n_notes = len(doc.get("notes", []))
        outdir = options["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        stab_path = outdir / f"{prefix}-{stamp}.stab"
        json_path = outdir / f"{prefix}-{stamp}.json"
        tab_path = outdir / f"{prefix}-{stamp}.tab"
        stab_text = render_string_tab(doc)
        stab_path.write_text(stab_text + "\n")
        json_path.write_text(json.dumps(doc, indent=2))
        tab_path.write_text(doc["tab"] + "\n")

        preview_path = None
        if options["preview"] and n_notes:
            preview_path = outdir / f"{prefix}-{stamp}-preview.wav"
            write_wav(preview_path, render_tab_audio(doc["notes"], sr=22050), 22050)

        source_path = None
        if source_audio is not None and (prefix == "hum" or options["save_demo"]):
            source_path = outdir / f"{prefix}-{stamp}-source.wav"
            write_wav(source_path, source_audio, cfg.transcribe.sr)

        text = (
            f"{doc['tab']}\n\n"
            f"wrote:\n  {stab_path}\n  {json_path}\n  {tab_path}\n"
        )
        if preview_path:
            text += f"  {preview_path}\n"
        if source_path:
            text += f"  {source_path}\n"
        if options["events"]:
            text += f"\n--- events ---\n{render_events(arranged, cfg)}\n"
        text += f"\n--- {stab_path.name} ---\n{stab_text}\n"
        if n_notes:
            self.status.set(f"Done -- {n_notes} notes. Ready for another source.")
        else:
            self.status.set("No notes detected, but the humming files were saved.")
            text = "No notes detected. The recording and empty transcription files were saved.\n\n" + text
        self._set_output(text)

    def validate_json(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Choose a timed transcription JSON file",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
        if not path:
            return
        try:
            doc = json.loads(Path(path).read_text())
            problems = validate_document(doc)
        except Exception as exc:  # noqa: BLE001
            self.status.set("Validation failed.")
            self._set_output(f"{type(exc).__name__}: {exc}")
            return
        if problems:
            self.status.set(f"Validation found {len(problems)} problem(s).")
            body = "\n".join(f"problem: {problem}" for problem in problems)
        else:
            self.status.set("Timed transcription JSON is valid.")
            body = "ok: note timing and string/fret data are consistent"
        self._set_output(f"{path}\n\n{body}\n")

    def _on_close(self) -> None:
        if self.recorder.is_recording:
            self.recorder.stop()
        self.root.destroy()

    def run_loop(self) -> None:
        self.root.mainloop()


def launch(
    cfg: Optional[PipelineConfig] = None,
    outdir: str = "recordings",
    quantize: Optional[int] = 16,
) -> None:
    cfg = cfg or PipelineConfig()
    ui = TunefinderUI(cfg, Path(outdir), quantize=quantize)
    ui.run_loop()


if __name__ == "__main__":
    launch()
