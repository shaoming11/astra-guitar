"""Fast single-note melody transcription output."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import PipelineConfig
from .timing import build_beat_map, force_monophonic, quantize_notes
from .transcribe import transcribe
from .types import Transcription, midi_to_hz


def transcribe_melody(
    source,
    cfg: Optional[PipelineConfig] = None,
    quantize: Optional[int] = None,
    voice_isolation: bool = False,
) -> Transcription:
    """Transcribe any supported source directly into a single melody line.

    The function intentionally skips fretboard arrangement and tab rendering.
    With raw timing (``quantize=None``), beat tracking is
    skipped too, which makes the common UI path substantially faster.
    """
    cfg = cfg or PipelineConfig()
    tr = transcribe(
        source,
        cfg.transcribe,
        voice_isolation=voice_isolation,
        estimate_timing=quantize is not None,
    )
    tr.notes = force_monophonic(tr.notes, cfg.transcribe.min_note_seconds)
    if quantize:
        beat_map = build_beat_map(tr.beat_times, tr.tempo_bpm, tr.duration)
        tr.notes = quantize_notes(
            tr.notes,
            beat_map,
            subdivision=quantize,
            min_duration=cfg.transcribe.min_note_seconds,
        )
        tr.notes = force_monophonic(tr.notes, cfg.transcribe.min_note_seconds)
    return tr


def melody_document(tr: Transcription, quantized: Optional[int] = None) -> dict:
    """Return a small JSON-safe document containing timed note events."""
    notes = []
    for index, note in enumerate(tr.notes):
        notes.append({
            "id": f"n{index:05d}",
            "onset_s": round(float(note.onset), 5),
            "duration_s": round(float(note.duration), 5),
            "offset_s": round(float(note.offset), 5),
            "midi": int(note.midi),
            "note": note.name,
            "frequency_hz": round(float(midi_to_hz(note.midi)), 3),
            "velocity": round(float(note.velocity), 3),
            "confidence": round(float(note.confidence), 3),
        })
    return {
        "format": "melody/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "file": tr.source,
            "duration_s": round(float(tr.duration), 3),
            "sample_rate": int(tr.sr),
            "mode": "monophonic",
            "voice_isolated": bool(tr.meta.get("voice_isolated", False)),
        },
        "timing": {
            "tempo_bpm": round(float(tr.tempo_bpm), 2) if tr.tempo_bpm > 0 else None,
            "quantized_subdivision": quantized,
            "beat_times": [round(float(b), 4) for b in tr.beat_times],
            "clock": "seconds from start of source",
        },
        "stats": {"notes": len(notes)},
        "notes": notes,
    }


def write_melody_files(
    tr: Transcription,
    outdir: str | Path,
    prefix: str,
    quantized: Optional[int] = None,
) -> dict[str, Path]:
    """Write JSON and CSV timing files, including an empty result safely."""
    folder = Path(outdir).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    doc = melody_document(tr, quantized=quantized)
    json_path = folder / f"{prefix}.json"
    csv_path = folder / f"{prefix}.csv"
    json_path.write_text(json.dumps(doc, indent=2) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id", "onset_s", "duration_s", "offset_s", "note", "midi",
                "frequency_hz", "velocity", "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(doc["notes"])
    return {"json": json_path, "csv": csv_path}
