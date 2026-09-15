"""Test suite. Run with: python -m pytest tests/ -q   (or: python tests/test_pipeline.py)"""

from __future__ import annotations

import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunefinder import PipelineConfig, Tuning, run                      # noqa: E402
from tunefinder.config import GuitarSpec, CostWeights, TranscribeSpec   # noqa: E402
from tunefinder.fretboard import (assign_fingers, enumerate_shapes,     # noqa: E402
                                  positions_for)
from tunefinder.synth import karplus_strong, synth_notes                # noqa: E402
from tunefinder.types import NoteEvent, Placement, midi_to_hz             # noqa: E402
from tunefinder.media import is_youtube_url, load_media                # noqa: E402
from tunefinder.transcribe import isolate_humming                       # noqa: E402
from tunefinder.timing import force_monophonic                           # noqa: E402


# --------------------------------------------------------------- fretboard

def test_positions_for():
    g = GuitarSpec(single_string=None)
    # middle C sits in several places on a standard guitar
    spots = positions_for(60, g)
    assert (3, 5) in [(p.string, p.fret) for p in spots]
    assert (5, 0) in [(p.string, p.fret) for p in positions_for(64, g)]
    assert positions_for(20, g) == []          # below the instrument


def test_span_limit_rejects_impossible_shapes():
    g = GuitarSpec(max_span=4)
    w = CostWeights()
    # C2 and a note nine frets away on the next string cannot be held together
    shapes = enumerate_shapes([40, 45 + 9], g, w)
    for s in shapes:
        assert s.span <= g.max_span


def test_open_chord_needs_no_barre():
    g, w = GuitarSpec(single_string=None), CostWeights()
    e_minor = [40, 47, 52, 55, 59, 64]
    best = enumerate_shapes(e_minor, g, w)[0]
    assert best.barre is None
    assert {(p.string, p.fret) for p in best.placements} == {
        (0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)
    }


def test_barre_used_when_fingers_run_out():
    g, w = GuitarSpec(single_string=None), CostWeights()
    f_major = [41, 48, 53, 57, 60, 65]         # F barre chord at the first fret
    best = enumerate_shapes(f_major, g, w)[0]
    assert best.barre is not None
    assert len(set(best.fingers.values())) <= g.max_fingers


def test_finger_assignment_is_ordered():
    g = GuitarSpec()
    placements = [Placement(4, 3, 62), Placement(3, 2, 57), Placement(2, 4, 59)]
    fingers, barre = assign_fingers(placements, g)
    assert barre is None
    assert fingers[(3, 2)] < fingers[(4, 3)] < fingers[(2, 4)]


def test_too_many_fingers_is_rejected():
    g = GuitarSpec(max_fingers=4)
    placements = [Placement(i, i + 1, 40 + i) for i in range(5)]
    fingers, _ = assign_fingers(placements, g)
    assert fingers is None


# ----------------------------------------------------------------- arranger

def test_arranger_keeps_a_run_on_one_string():
    """An ascending run should not hop across strings note by note.

    Every pitch here is reachable on three different strings, so a note by
    note choice would scatter. The travel term in the transition cost is what
    holds the run together.
    """
    cfg = PipelineConfig()
    events = [(m, i * 0.35, 0.3) for i, m in enumerate([52, 54, 56, 57, 59])]
    y = synth_notes(events, sr=cfg.transcribe.sr)
    doc, arranged, _ = run((y, cfg.transcribe.sr), cfg)
    assert len({n["string"] for n in doc["notes"]}) == 1
    positions = [g.shape.hand_position for g in arranged]
    assert positions == sorted(positions)          # hand only moves one way


def test_alternate_tuning_changes_the_tab():
    """D2 is below a standard guitar. Drop D reaches it on the open low string."""
    cfg_std = PipelineConfig(guitar=GuitarSpec(single_string=None))
    cfg_drop = PipelineConfig(guitar=GuitarSpec(tuning=Tuning.get("drop_d"), single_string=None))
    y = synth_notes([(38, 0.0, 0.5), (50, 0.6, 0.5)], sr=cfg_std.transcribe.sr)

    drop, _, _ = run((y, cfg_drop.transcribe.sr), cfg_drop)
    low = [n for n in drop["notes"] if n["midi"] == 38]
    assert low and low[0]["string"] == 0 and low[0]["fret"] == 0

    std, _, _ = run((y, cfg_std.transcribe.sr), cfg_std)
    assert all(n["midi"] != 38 for n in std["notes"])      # folded into range
    assert min(n["midi"] for n in std["notes"]) >= 40


def test_out_of_range_notes_are_folded_not_dropped():
    cfg = PipelineConfig()
    y = synth_notes([(28, 0.0, 0.6), (36, 0.7, 0.6)], sr=cfg.transcribe.sr)
    doc, _, _ = run((y, cfg.transcribe.sr), cfg)
    lo = cfg.guitar.tuning.open_midi[cfg.guitar.single_string] + cfg.guitar.capo
    for n in doc["notes"]:
        assert n["midi"] >= lo


# ------------------------------------------------------------ transcription output

def _demo_doc():
    cfg = PipelineConfig()
    events = [(m, i * 0.5, 0.45) for i, m in enumerate([64, 67, 71, 67, 64])]
    y = synth_notes(events, sr=cfg.transcribe.sr)
    return run((y, cfg.transcribe.sr), cfg)


def test_document_shape():
    doc, _, _ = _demo_doc()
    assert doc["format"] == "tunefinder/1.0"
    assert "robot" not in doc
    assert "commands" not in doc
    for key in ("instrument", "timing", "stats", "notes", "string_tab", "tab"):
        assert key in doc
    for n in doc["notes"]:
        assert 0 <= n["string"] < cfgless_n_strings()
        assert n["string"] == 2
        assert n["string_fret"].startswith("D-")
        assert 0 <= n["fret"] <= 15
        assert 0 <= n["finger"] <= 4
        assert "string_fret" in n
        assert abs(n["freq_hz"] - midi_to_hz(n["midi"])) < 0.01


def cfgless_n_strings() -> int:
    return PipelineConfig().guitar.tuning.n_strings


# ------------------------------------------------------------- end to end

RIFF = [(57, 0.0, 0.4), (60, 0.4, 0.4), (62, 0.8, 0.4), (64, 1.2, 0.4),
        (67, 1.6, 0.8), (64, 2.4, 0.2), (62, 2.6, 0.2), (60, 2.8, 0.2),
        (57, 3.0, 0.6), (69, 3.6, 0.4), (67, 4.0, 0.4), (64, 4.4, 0.8)]


def _plucked_riff(sr=22050, noise=0.004):
    total = max(s + d for _, s, d in RIFF) + 0.8
    y = np.zeros(int(sr * total) + 1, dtype=np.float32)
    for m, s, d in RIFF:
        w = karplus_strong(midi_to_hz(m), d + 0.3, sr) * 0.9
        i = int(s * sr)
        y[i:i + w.size] += w[:y.size - i]
    y += np.random.default_rng(0).normal(0, noise, y.size).astype(np.float32)
    return y, sr


def test_pitch_accuracy_on_plucked_riff():
    y, sr = _plucked_riff()
    cfg = PipelineConfig(guitar=GuitarSpec(max_fret=24))
    doc, _, _ = run((y, sr), cfg)
    assert [n["midi"] for n in doc["notes"]] == [m for m, _, _ in RIFF]


def test_onset_accuracy_on_plucked_riff():
    y, sr = _plucked_riff()
    doc, _, _ = run((y, sr), PipelineConfig(), quantize=None)
    errors = [abs(n["onset_s"] - s) for n, (_, s, _) in zip(doc["notes"], RIFF)]
    errors.sort()
    median = errors[len(errors) // 2]
    assert median < 0.020, f"median onset error {median*1000:.0f} ms"


def test_chord_recognition():
    """Open Em, C, G and D should come back as their standard voicings."""
    cfg = PipelineConfig(
        guitar=GuitarSpec(single_string=None),
        transcribe=TranscribeSpec(polyphonic=True),
    )
    stacks = [((40, 47, 52, 55, 59, 64), 0.0), ((48, 52, 55, 60, 64), 1.2),
              ((43, 47, 50, 55, 59, 67), 2.4), ((50, 57, 62, 66), 3.6)]
    events = [(m, s, 1.2) for stack, s in stacks for m in stack]
    y = synth_notes(events, sr=cfg.transcribe.sr)
    doc, arranged, _ = run((y, cfg.transcribe.sr), cfg, force_single_notes=False)
    assert len(arranged) == 4
    shapes = [{p.string: p.fret for p in g.shape.placements} for g in arranged]
    assert shapes[0] == {0: 0, 1: 2, 2: 2, 3: 0, 4: 0, 5: 0}       # Em
    assert shapes[1] == {1: 3, 2: 2, 3: 0, 4: 1, 5: 0}             # C
    assert shapes[2][0] == 3 and shapes[2][5] == 3                 # G
    assert shapes[3] == {2: 0, 3: 2, 4: 3, 5: 2}                   # D


def test_silence_produces_an_empty_but_valid_document():
    sr = 22050
    y = np.zeros(sr * 2, dtype=np.float32)
    doc, arranged, _ = run((y, sr), PipelineConfig())
    assert arranged == []
    assert doc["notes"] == []
    assert doc["string_tab"].startswith("# tunefinder stringtab v1")
    assert doc["format"] == "tunefinder/1.0"


# -------------------------------------------------------------- media / hum

def test_youtube_url_detection():
    assert is_youtube_url("https://www.youtube.com/watch?v=abc")
    assert is_youtube_url("https://youtu.be/abc")
    assert not is_youtube_url("https://example.com/video.mp4")


def test_local_video_is_decoded_to_mono_audio(tmp_path):
    """A video upload is treated as an audio source without a video library."""
    if shutil.which("ffmpeg") is None:
        return  # FFmpeg is an external runtime requirement for media input.
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4",
            "-f", "lavfi", "-i", "color=c=black:s=160x120:d=0.4",
            "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )
    y, sr = load_media(video, 22050)
    assert sr == 22050
    assert y.ndim == 1
    assert y.size > 5000


def test_humming_cleanup_suppresses_quiet_room_noise():
    sr = 22050
    t = np.arange(sr * 2) / sr
    active = (t >= 0.3) & (t < 1.3)
    voice = np.zeros_like(t)
    voice[active] = 0.25 * np.sin(2 * np.pi * 220 * t[active])
    noise = np.random.default_rng(1).normal(0, 0.05, t.size)
    cleaned = isolate_humming(voice + noise, sr, TranscribeSpec())
    active_rms = float(np.sqrt(np.mean(cleaned[int(0.4 * sr):int(1.2 * sr)] ** 2)))
    quiet_rms = float(np.sqrt(np.mean(cleaned[int(1.5 * sr):int(1.9 * sr)] ** 2)))
    assert np.isfinite(cleaned).all()
    assert active_rms > quiet_rms * 20


def test_force_monophonic_removes_simultaneous_and_overlapping_notes():
    notes = [
        NoteEvent(0.0, 0.5, 60, velocity=0.7, confidence=0.7),
        NoteEvent(0.0, 0.5, 64, velocity=0.8, confidence=0.9),
        NoteEvent(0.3, 0.4, 67, velocity=0.8, confidence=0.9),
    ]
    simplified = force_monophonic(notes)
    assert [n.midi for n in simplified] == [64, 67]
    assert simplified[0].offset <= simplified[1].onset


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
