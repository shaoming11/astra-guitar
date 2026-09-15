"""Data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


@dataclass
class NoteEvent:
    """One detected note before it is placed on the fretboard."""

    onset: float
    duration: float
    midi: int
    velocity: float = 0.8
    confidence: float = 1.0
    group: int = 0          # notes sharing a group are struck together

    @property
    def offset(self) -> float:
        return self.onset + self.duration

    @property
    def name(self) -> str:
        return midi_to_name(self.midi)


@dataclass(frozen=True)
class Placement:
    """A single note realised on the fretboard."""

    string: int
    fret: int
    midi: int

    @property
    def is_open(self) -> bool:
        return self.fret == 0


@dataclass
class Shape:
    """A set of simultaneous placements plus the fretting-hand geometry."""

    placements: Tuple[Placement, ...]
    hand_position: int          # lowest fretted fret, i.e. index-finger fret
    span: int
    fingers: dict               # (string, fret) -> finger 1..4
    barre: Optional[Tuple[int, int, int]] = None   # (fret, low_string, high_string)
    static_cost: float = 0.0

    @property
    def strings(self) -> Tuple[int, ...]:
        return tuple(p.string for p in self.placements)


@dataclass
class ArrangedGroup:
    """A chord group after fretboard assignment."""

    index: int
    onset: float
    duration: float
    notes: List[NoteEvent]
    shape: Shape
    beat: Optional[float] = None
    measure: Optional[int] = None


@dataclass
class Transcription:
    notes: List[NoteEvent]
    tempo_bpm: float
    beat_times: List[float]
    duration: float
    sr: int
    source: str = ""
    polyphonic: bool = False
    meta: dict = field(default_factory=dict)
