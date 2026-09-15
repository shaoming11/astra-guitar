"""Fast transcription document output."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

from .config import PipelineConfig
from .tabtext import render_string_tab, string_letters
from .types import ArrangedGroup, midi_to_hz, midi_to_name


def build_document(
    arranged: Sequence[ArrangedGroup],
    cfg: PipelineConfig,
    transcription,
    stats: dict,
    tab_text: str = "",
    quantized: Optional[int] = None,
) -> dict:
    """Create a timed melody file with fretboard placement and tab data."""
    letters = string_letters(cfg.guitar.tuning.open_midi)
    notes = []
    for group in arranged:
        by_midi = {n.midi: n for n in group.notes}
        for placement in group.shape.placements:
            src = by_midi.get(placement.midi)
            onset = src.onset if src else group.onset
            duration = src.duration if src else group.duration
            string_fret = (
                f"{letters[placement.string]}-{placement.fret}"
                if placement.string < len(letters) else f"?-{placement.fret}"
            )
            notes.append({
                "note_id": f"n{len(notes):04d}",
                "group": group.index,
                "onset_s": round(float(onset), 5),
                "duration_s": round(max(float(duration), 0.02), 5),
                "beat": round(group.beat, 4) if group.beat is not None else None,
                "measure": group.measure,
                "midi": int(placement.midi),
                "pitch": midi_to_name(placement.midi),
                "freq_hz": round(midi_to_hz(placement.midi), 3),
                "string": placement.string,
                "string_label": cfg.guitar.tuning.string_label(placement.string),
                "fret": placement.fret,
                "string_fret": string_fret,
                "finger": group.shape.fingers.get((placement.string, placement.fret), 0),
                "open": placement.is_open,
                "velocity": round(src.velocity if src else 0.7, 3),
                "confidence": round(src.confidence if src else 0.5, 3),
            })
    notes.sort(key=lambda n: (n["onset_s"], n["midi"]))

    doc = {
        "format": "tunefinder/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "file": transcription.source,
            "duration_s": round(transcription.duration, 3),
            "sample_rate": transcription.sr,
            "mode": "monophonic",
            "voice_isolated": bool(transcription.meta.get("voice_isolated", False)),
        },
        "instrument": {
            "type": "guitar",
            "tuning": cfg.guitar.tuning.name,
            "open_midi": list(cfg.guitar.tuning.open_midi),
            "string_index": "0 = lowest pitched string",
            "single_string": cfg.guitar.single_string,
            "single_string_label": (
                cfg.guitar.tuning.string_label(cfg.guitar.single_string)
                if cfg.guitar.single_string is not None else None
            ),
            "single_string_name": (
                letters[cfg.guitar.single_string]
                if cfg.guitar.single_string is not None
                and cfg.guitar.single_string < len(letters) else None
            ),
            "max_fret": cfg.guitar.max_fret,
            "capo": cfg.guitar.capo,
        },
        "timing": {
            "tempo_bpm": round(transcription.tempo_bpm, 2),
            "time_signature": "4/4",
            "quantized_subdivision": quantized,
            "beat_times": [round(b, 4) for b in transcription.beat_times],
            "clock": "seconds from start of take",
        },
        "stats": stats,
        "notes": notes,
        "warnings": [],
        "tab": tab_text,
    }
    # Keep a ready-to-save human/line-oriented representation beside the
    # structured per-note string/fret fields.
    doc["string_tab"] = render_string_tab(doc)
    return doc
