"""Musical time: beat mapping, quantisation, octave folding."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .config import GuitarSpec
from .fretboard import playable_range
from .types import NoteEvent


def build_beat_map(beat_times: Sequence[float], bpm: float, duration: float) -> np.ndarray:
    """Return a monotonically increasing array of beat boundary times.

    bpm <= 0 (beat tracking found nothing, e.g. onset-poor hummed audio)
    would otherwise divide down to a near-zero rate and a beat step of
    minutes, collapsing every note's quantized onset onto beat zero.
    """
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0
    beats = [b for b in beat_times if b >= 0]
    if len(beats) >= 2:
        arr = np.asarray(beats, dtype=float)
        step = float(np.median(np.diff(arr)))
        if step <= 0:
            step = 60.0 / max(bpm, 1e-6)
        # extend backwards to 0 and forwards to the end of the take
        pre = np.arange(arr[0] - step, -step / 2, -step)[::-1]
        post = np.arange(arr[-1] + step, duration + step, step)
        return np.concatenate([pre, arr, post])
    step = 60.0 / max(bpm, 1e-6)
    return np.arange(0.0, duration + step, step)


def time_to_beat(t: float, beat_map: np.ndarray) -> float:
    """Fractional beat number for an absolute time."""
    if beat_map.size < 2:
        return 0.0
    i = int(np.clip(np.searchsorted(beat_map, t) - 1, 0, beat_map.size - 2))
    span = beat_map[i + 1] - beat_map[i]
    frac = (t - beat_map[i]) / span if span > 0 else 0.0
    return float(i + frac)


def beat_to_time(beat: float, beat_map: np.ndarray) -> float:
    if beat_map.size < 2:
        return float(beat)
    i = int(np.clip(int(np.floor(beat)), 0, beat_map.size - 2))
    frac = beat - i
    return float(beat_map[i] + frac * (beat_map[i + 1] - beat_map[i]))


def quantize_notes(
    notes: List[NoteEvent],
    beat_map: np.ndarray,
    subdivision: int = 4,
    max_shift_beats: float = 0.35,
    min_duration: float = 0.045,
) -> List[NoteEvent]:
    """Snap onsets and durations to a 1/`subdivision` of a beat grid.

    A note is only moved when the snap is within `max_shift_beats`, so a
    deliberately loose performance is not forced onto the grid.
    """
    step = 1.0 / subdivision
    for n in notes:
        b = time_to_beat(n.onset, beat_map)
        snapped = round(b / step) * step
        if abs(snapped - b) <= max_shift_beats:
            end_b = time_to_beat(n.onset + n.duration, beat_map)
            snapped_end = round(end_b / step) * step
            if snapped_end <= snapped:
                snapped_end = snapped + step
            n.onset = beat_to_time(snapped, beat_map)
            n.duration = max(beat_to_time(snapped_end, beat_map) - n.onset, 0.02)
        n.onset = max(n.onset, 0.0)
    notes = [n for n in notes if n.duration >= min_duration]
    notes.sort(key=lambda n: (n.onset, n.midi))
    return notes


def force_monophonic(notes: List[NoteEvent], min_duration: float = 0.045) -> List[NoteEvent]:
    """Collapse simultaneous/overlapping detections into one melody line.

    This is useful when simplifying a full song or a noisy recording for a
    one-note instrument.  When multiple notes start together, the strongest
    and most confident detection wins.  Later notes trim an earlier overlap,
    so the returned events never sound at the same time.
    """
    ordered = sorted(notes, key=lambda n: (n.onset, -n.confidence, -n.velocity, n.midi))
    out: List[NoteEvent] = []
    for note in ordered:
        if out and abs(note.onset - out[-1].onset) <= 1e-6:
            previous = out[-1]
            previous_score = previous.confidence * 0.7 + previous.velocity * 0.3
            note_score = note.confidence * 0.7 + note.velocity * 0.3
            if note_score > previous_score:
                out[-1] = note
            continue

        if out and note.onset < out[-1].offset:
            out[-1].duration = note.onset - out[-1].onset
            if out[-1].duration < min_duration:
                out.pop()
        if note.duration >= min_duration:
            out.append(note)

    for index, note in enumerate(out):
        note.group = index
    return out


def fit_to_range(notes: List[NoteEvent], guitar: GuitarSpec) -> int:
    """Fold notes outside the instrument range into it by whole octaves.

    Returns how many notes were moved.
    """
    lo, hi = playable_range(guitar)
    moved = 0
    for n in notes:
        original = n.midi
        while n.midi < lo:
            n.midi += 12
        while n.midi > hi:
            n.midi -= 12
        n.midi = int(np.clip(n.midi, lo, hi))
        if n.midi != original:
            moved += 1
    return moved


def dedupe_unisons(notes: List[NoteEvent]) -> List[NoteEvent]:
    """Within a group, one pitch only once. Two strings cannot share a pitch slot."""
    seen = {}
    out = []
    for n in sorted(notes, key=lambda n: (n.group, n.midi, -n.velocity)):
        key = (n.group, n.midi)
        if key in seen:
            continue
        seen[key] = True
        out.append(n)
    out.sort(key=lambda n: (n.onset, n.midi))
    return out


def infer_time_signature(beats_per_bar: Optional[int]) -> str:
    return f"{beats_per_bar or 4}/4"
