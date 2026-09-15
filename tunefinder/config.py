"""Configuration objects for the transcription -> tab -> robot pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

# String index 0 is the LOWEST pitched string (6th string, low E in standard).
# String index 5 is the HIGHEST pitched string (1st string, high E).
# This ordering is used everywhere internally. Human-facing string numbers
# (6..1) are derived only at render time.

TUNINGS = {
    "standard": (40, 45, 50, 55, 59, 64),      # E2 A2 D3 G3 B3 E4
    "drop_d": (38, 45, 50, 55, 59, 64),        # D2 A2 D3 G3 B3 E4
    "open_g": (38, 43, 50, 55, 59, 62),        # D2 G2 D3 G3 B3 D4
    "dadgad": (38, 45, 50, 55, 57, 62),        # D2 A2 D3 G3 A3 D4
    "half_step_down": (39, 44, 49, 54, 58, 63),
    "bass4": (28, 33, 38, 43),                 # E1 A1 D2 G2
}


@dataclass(frozen=True)
class Tuning:
    name: str
    open_midi: Tuple[int, ...]

    @property
    def n_strings(self) -> int:
        return len(self.open_midi)

    def string_label(self, idx: int) -> str:
        """Human string number: index 0 -> '6' for a 6 string guitar."""
        return str(self.n_strings - idx)

    @classmethod
    def get(cls, name: str) -> "Tuning":
        key = name.lower()
        if key not in TUNINGS:
            raise ValueError(f"unknown tuning {name!r}; have {sorted(TUNINGS)}")
        return cls(key, TUNINGS[key])


@dataclass
class GuitarSpec:
    """Physical limits of the instrument and of the fretting robot."""

    tuning: Tuning = field(default_factory=lambda: Tuning.get("standard"))
    max_fret: int = 15              # highest fret the robot can reach
    max_span: int = 4               # fret window the fretting hand covers
    max_fingers: int = 4
    allow_open_strings: bool = True
    capo: int = 0


@dataclass
class CostWeights:
    """Weights for the fretting-path optimiser. Lower total cost wins."""

    fret_height: float = 0.25       # prefer low frets (better tone, less reach)
    hand_shift: float = 1.0         # cost per fret of hand translation
    span: float = 0.6               # cost per fret of stretch inside a shape
    finger_count: float = 0.3       # prefer fewer fingers down
    open_string: float = -0.8       # bonus per open string used
    string_travel: float = 0.35     # picking arm cost per string crossed
    string_change: float = 0.15     # flat cost for changing string at all
    barre: float = 0.5              # barres are slower to form than free fingers
    high_position_bias: float = 0.1  # extra pull toward first position


@dataclass
class TranscribeSpec:
    sr: int = 22050
    hop_length: int = 256
    frame_length: int = 2048
    # Pitch search range. Left as None, the pipeline derives it from the
    # instrument so alternate tunings are covered automatically.
    fmin_hz: Optional[float] = None
    fmax_hz: Optional[float] = None
    min_note_seconds: float = 0.045
    min_confidence: float = 0.5    # fraction of frames agreeing on the pitch
    silence_floor: float = 0.04    # fraction of peak RMS below which a segment is silence
    artifact_max_seconds: float = 0.12  # shorter excursions with no attack are dropped
    strip_continuations: bool = True    # drop unattacked stepwise pass-through pitches
    continuation_rise: float = 1.10     # envelope rise that marks a real pluck
    reattack_ratio: float = 1.25   # envelope rise needed to call a re-pluck
    polyphonic: bool = False
    poly_max_notes: int = 6
    poly_rel_threshold: float = 0.16
    poly_onset_floor: float = 0.12   # ignore onsets that are only decay tail
    chord_window: float = 0.045     # onsets closer than this = one chord


@dataclass
class RobotSpec:
    """Timing and kinematics envelope for the two arms."""

    press_lead: float = 0.060       # seconds the finger lands before the pluck
    release_lag: float = 0.015      # seconds after note end before lifting
    min_press_gap: float = 0.010    # minimum dwell between release and re-press
    strum_stagger: float = 0.018    # per-string delay inside a strum
    fret_travel_speed: float = 45.0  # frets per second the fret arm can slide
    string_travel_speed: float = 22.0  # strings per second the pick arm can cross
    alternate_picking: bool = True


@dataclass
class PipelineConfig:
    guitar: GuitarSpec = field(default_factory=GuitarSpec)
    weights: CostWeights = field(default_factory=CostWeights)
    transcribe: TranscribeSpec = field(default_factory=TranscribeSpec)
    robot: RobotSpec = field(default_factory=RobotSpec)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["guitar"]["tuning"] = {
            "name": self.guitar.tuning.name,
            "open_midi": list(self.guitar.tuning.open_midi),
        }
        return d
