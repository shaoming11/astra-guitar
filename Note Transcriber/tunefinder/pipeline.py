"""One call from media to a timed transcription and guitar tab."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .arrange import arrange, arrangement_stats
from .config import PipelineConfig
from .fretboard import playable_range
from .document import build_document
from .tabtext import render_tab
from .timing import build_beat_map, force_monophonic, quantize_notes
from .transcribe import transcribe
from .types import Transcription, midi_to_hz


def run(
    source,
    cfg: Optional[PipelineConfig] = None,
    quantize: Optional[int] = 16,
    line_width: int = 72,
    voice_isolation: bool = False,
    force_single_notes: bool = True,
) -> Tuple[dict, list, Transcription]:
    """source is audio/video path, URL, or a (samples, sample_rate) tuple.

    Returns (document, arranged_groups, transcription).
    """
    cfg = cfg or PipelineConfig()

    # Match the pitch search to the instrument, so alternate tunings and a
    # raised fret ceiling are covered without the caller thinking about it.
    lo, hi = playable_range(cfg.guitar)
    if cfg.transcribe.fmin_hz is None:
        cfg.transcribe.fmin_hz = midi_to_hz(lo - 1)
    if cfg.transcribe.fmax_hz is None:
        cfg.transcribe.fmax_hz = midi_to_hz(hi + 1)

    tr = transcribe(
        source,
        cfg.transcribe,
        voice_isolation=voice_isolation,
        estimate_timing=quantize is not None,
    )

    if force_single_notes:
        tr.notes = force_monophonic(
            tr.notes, min_duration=cfg.transcribe.min_note_seconds,
        )

    if quantize:
        beat_map = build_beat_map(tr.beat_times, tr.tempo_bpm, tr.duration)
        tr.notes = quantize_notes(
            tr.notes, beat_map, subdivision=quantize,
            min_duration=cfg.transcribe.min_note_seconds,
        )
        if force_single_notes:
            tr.notes = force_monophonic(
                tr.notes, min_duration=cfg.transcribe.min_note_seconds,
            )

    arranged = arrange(tr, cfg)
    stats = arrangement_stats(arranged, cfg)
    tab = render_tab(arranged, cfg, line_width=line_width)
    doc = build_document(arranged, cfg, tr, stats, tab_text=tab, quantized=quantize)
    return doc, arranged, tr
