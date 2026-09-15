"""Turn detected pitches into a fretting path.

The fretboard is ambiguous: middle C sits in four places on a standard guitar.
Choosing note by note produces a tab a human, or a robot, cannot physically
play. This module picks the whole path at once with a Viterbi search whose
emission cost is shape difficulty and whose transition cost is hand travel.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import PipelineConfig
from .fretboard import enumerate_shapes, transition_cost
from .timing import build_beat_map, dedupe_unisons, fit_to_range, time_to_beat
from .types import ArrangedGroup, NoteEvent, Shape, Transcription


def _groups(notes: Sequence[NoteEvent]) -> List[List[NoteEvent]]:
    buckets: Dict[int, List[NoteEvent]] = defaultdict(list)
    for n in notes:
        buckets[n.group].append(n)
    ordered = sorted(buckets.values(), key=lambda g: min(x.onset for x in g))
    return [sorted(g, key=lambda n: n.midi) for g in ordered]


def _shapes_for_group(
    group: Sequence[NoteEvent], cfg: PipelineConfig
) -> Tuple[List[Shape], List[NoteEvent]]:
    """Enumerate shapes, dropping the quietest notes if the hand cannot cope."""
    notes = list(group)
    while notes:
        shapes = enumerate_shapes([n.midi for n in notes], cfg.guitar, cfg.weights)
        if shapes:
            sounded = {p.midi for p in shapes[0].placements}
            kept = [n for n in notes if n.midi in sounded]
            return shapes, kept or notes
        notes = sorted(notes, key=lambda n: -n.velocity)[:-1]
        notes.sort(key=lambda n: n.midi)
    return [], []


def arrange(transcription: Transcription, cfg: PipelineConfig) -> List[ArrangedGroup]:
    notes = dedupe_unisons(list(transcription.notes))
    fit_to_range(notes, cfg.guitar)
    groups = _groups(notes)
    if not groups:
        return []

    candidates: List[List[Shape]] = []
    kept_notes: List[List[NoteEvent]] = []
    for g in groups:
        shapes, kept = _shapes_for_group(g, cfg)
        if not shapes:
            continue
        candidates.append(shapes)
        kept_notes.append(kept)

    if not candidates:
        return []

    # ---- Viterbi ----
    n_steps = len(candidates)
    costs: List[np.ndarray] = []
    back: List[np.ndarray] = []

    first = np.array([s.static_cost for s in candidates[0]], dtype=float)
    costs.append(first)
    back.append(np.full(first.shape, -1, dtype=int))

    for i in range(1, n_steps):
        cur = candidates[i]
        prev = candidates[i - 1]
        prev_cost = costs[-1]
        step_cost = np.empty(len(cur), dtype=float)
        step_back = np.empty(len(cur), dtype=int)
        for j, shape in enumerate(cur):
            totals = [
                prev_cost[k] + transition_cost(prev[k], shape, cfg.weights)
                for k in range(len(prev))
            ]
            k_best = int(np.argmin(totals))
            step_cost[j] = totals[k_best] + shape.static_cost
            step_back[j] = k_best
        costs.append(step_cost)
        back.append(step_back)

    path = [int(np.argmin(costs[-1]))]
    for i in range(n_steps - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    path.reverse()

    beat_map = build_beat_map(
        transcription.beat_times, transcription.tempo_bpm, transcription.duration
    )

    arranged: List[ArrangedGroup] = []
    for i, (shape_idx, group) in enumerate(zip(path, kept_notes)):
        shape = candidates[i][shape_idx]
        onset = min(n.onset for n in group)
        duration = max(n.duration for n in group)
        beat = max(0.0, time_to_beat(onset, beat_map))
        arranged.append(
            ArrangedGroup(
                index=i,
                onset=onset,
                duration=duration,
                notes=group,
                shape=shape,
                beat=beat,
                measure=int(beat // 4),
            )
        )
    return arranged


def arrangement_stats(arranged: Sequence[ArrangedGroup], cfg: PipelineConfig) -> dict:
    if not arranged:
        return {"groups": 0, "notes": 0}
    frets = [p.fret for g in arranged for p in g.shape.placements]
    shifts = [
        abs(b.shape.hand_position - a.shape.hand_position)
        for a, b in zip(arranged, arranged[1:])
        if a.shape.hand_position and b.shape.hand_position
    ]
    return {
        "groups": len(arranged),
        "notes": sum(len(g.notes) for g in arranged),
        "chords": sum(1 for g in arranged if len(g.shape.placements) > 1),
        "open_strings": sum(1 for f in frets if f == 0),
        "max_fret": max(frets) if frets else 0,
        "mean_fret": round(float(np.mean(frets)), 2) if frets else 0.0,
        "hand_shifts": len([s for s in shifts if s > 0]),
        "total_shift_frets": int(sum(shifts)),
        "barres": sum(1 for g in arranged if g.shape.barre),
    }
