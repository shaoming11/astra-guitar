"""Fretboard geometry: where a pitch can live, and which shapes a hand can hold."""

from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from .config import CostWeights, GuitarSpec
from .types import Placement, Shape


def positions_for(midi: int, guitar: GuitarSpec) -> List[Placement]:
    """Every allowed (string, fret) pair that produces this pitch."""
    out: List[Placement] = []
    if guitar.single_string is None:
        strings = enumerate(guitar.tuning.open_midi)
    else:
        index = int(guitar.single_string)
        if index < 0 or index >= guitar.tuning.n_strings:
            return out
        strings = ((index, guitar.tuning.open_midi[index]),)
    for s, open_midi in strings:
        fret = midi - open_midi - guitar.capo
        if fret < 0 or fret > guitar.max_fret:
            continue
        if fret == 0 and not guitar.allow_open_strings:
            continue
        out.append(Placement(string=s, fret=fret, midi=midi))
    return out


def playable_range(guitar: GuitarSpec) -> Tuple[int, int]:
    """Pitch range used for placement, narrowed to the selected string."""
    if guitar.single_string is not None:
        index = int(guitar.single_string)
        if 0 <= index < guitar.tuning.n_strings:
            lo = guitar.tuning.open_midi[index] + guitar.capo
            return lo, lo + guitar.max_fret
    lo = min(guitar.tuning.open_midi) + guitar.capo
    hi = max(guitar.tuning.open_midi) + guitar.capo + guitar.max_fret
    return lo, hi


def assign_fingers(
    placements: Sequence[Placement], guitar: GuitarSpec
) -> Tuple[Optional[Dict[Tuple[int, int], int]], Optional[Tuple[int, int, int]]]:
    """Map fretted placements to fingers 1..4.

    Returns (fingers, barre). `fingers` is None when the shape needs more
    fingers than the hand has. `barre` is (fret, low_string, high_string).
    """
    fretted = [p for p in placements if p.fret > 0]
    if not fretted:
        return {}, None

    min_fret = min(p.fret for p in fretted)
    at_min = [p for p in fretted if p.fret == min_fret]

    fingers: Dict[Tuple[int, int], int] = {}
    barre: Optional[Tuple[int, int, int]] = None
    next_finger = 1

    # A barre is slower to form than independent fingers, so only reach for one
    # when three or more strings share the low fret, or when the hand would
    # otherwise run out of fingers.
    need_barre = len(at_min) >= 3 or len(fretted) > guitar.max_fingers

    if need_barre and len(at_min) > 1:
        barre = (min_fret, min(p.string for p in at_min), max(p.string for p in at_min))
        for p in at_min:
            fingers[(p.string, p.fret)] = 1
        next_finger = 2
        rest = [p for p in fretted if p.fret != min_fret]
    else:
        rest = list(fretted)

    for p in sorted(rest, key=lambda q: (q.fret, q.string)):
        if next_finger > guitar.max_fingers:
            return None, None
        fingers[(p.string, p.fret)] = next_finger
        next_finger += 1

    return fingers, barre


def _static_cost(
    placements: Sequence[Placement],
    fingers: Dict[Tuple[int, int], int],
    barre: Optional[Tuple[int, int, int]],
    span: int,
    w: CostWeights,
) -> float:
    cost = 0.0
    for p in placements:
        if p.is_open:
            cost += w.open_string
        else:
            cost += w.fret_height * p.fret
            cost += w.high_position_bias * max(0, p.fret - 5)
    cost += w.span * max(0, span - 1)
    cost += w.finger_count * len(set(fingers.values()))
    if barre is not None:
        cost += w.barre
    return cost


def make_shape(
    placements: Sequence[Placement], guitar: GuitarSpec, w: CostWeights
) -> Optional[Shape]:
    """Validate a candidate combination and wrap it as a Shape."""
    strings = [p.string for p in placements]
    if len(set(strings)) != len(strings):
        return None                                  # two notes on one string

    fretted = [p for p in placements if p.fret > 0]
    if fretted:
        lo = min(p.fret for p in fretted)
        hi = max(p.fret for p in fretted)
        span = hi - lo + 1
        if span > guitar.max_span:
            return None
        hand_position = lo
    else:
        span = 0
        hand_position = 0

    fingers, barre = assign_fingers(placements, guitar)
    if fingers is None:
        return None

    return Shape(
        placements=tuple(sorted(placements, key=lambda p: p.string)),
        hand_position=hand_position,
        span=span,
        fingers=fingers,
        barre=barre,
        static_cost=_static_cost(placements, fingers, barre, span, w),
    )


def enumerate_shapes(
    midis: Sequence[int],
    guitar: GuitarSpec,
    w: CostWeights,
    limit: int = 96,
) -> List[Shape]:
    """All hand shapes that sound the requested pitch set, cheapest first."""
    pitches = sorted(set(int(m) for m in midis))
    per_note = [positions_for(m, guitar) for m in pitches]
    if any(len(c) == 0 for c in per_note):
        # Drop pitches outside the instrument range rather than failing outright.
        keep = [(m, c) for m, c in zip(pitches, per_note) if c]
        if not keep:
            return []
        pitches = [m for m, _ in keep]
        per_note = [c for _, c in keep]

    shapes: List[Shape] = []
    for combo in product(*per_note):
        shape = make_shape(combo, guitar, w)
        if shape is not None:
            shapes.append(shape)

    shapes.sort(key=lambda s: s.static_cost)
    return shapes[:limit]


def transition_cost(prev: Optional[Shape], nxt: Shape, w: CostWeights) -> float:
    if prev is None:
        return 0.0
    cost = 0.0
    if nxt.hand_position and prev.hand_position:
        cost += w.hand_shift * abs(nxt.hand_position - prev.hand_position)
    prev_mean = sum(prev.strings) / len(prev.strings)
    next_mean = sum(nxt.strings) / len(nxt.strings)
    travel = abs(next_mean - prev_mean)
    cost += w.string_travel * travel
    if travel > 0:
        cost += w.string_change
    held = set(prev.fingers.keys()) & set(nxt.fingers.keys())
    cost -= 0.25 * len(held)          # reward keeping fingers planted
    return cost
