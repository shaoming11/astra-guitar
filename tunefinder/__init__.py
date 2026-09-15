"""tunefinder: audio in, guitar tab and a robot command stream out."""

from .config import (
    CostWeights,
    GuitarSpec,
    PipelineConfig,
    RobotSpec,
    TranscribeSpec,
    Tuning,
    TUNINGS,
)
from .pipeline import run
from .types import ArrangedGroup, NoteEvent, Placement, Shape, Transcription

__all__ = [
    "run",
    "PipelineConfig",
    "GuitarSpec",
    "Tuning",
    "TUNINGS",
    "CostWeights",
    "RobotSpec",
    "TranscribeSpec",
    "NoteEvent",
    "Placement",
    "Shape",
    "ArrangedGroup",
    "Transcription",
]
