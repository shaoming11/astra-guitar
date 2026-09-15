"""Test suite. Run with: python -m pytest tests/ -q   (or: python tests/test_pipeline.py)"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunefinder import PipelineConfig, Tuning, run                      # noqa: E402
from tunefinder.config import GuitarSpec, CostWeights, TranscribeSpec   # noqa: E402
from tunefinder.fretboard import (assign_fingers, enumerate_shapes,     # noqa: E402
                                  positions_for)
from tunefinder.synth import karplus_strong, synth_notes                # noqa: E402
from tunefinder.types import Placement, midi_to_hz                      # noqa: E402


# --------------------------------------------------------------- fretboard

def test_positions_for():
    g = GuitarSpec()
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
    g, w = GuitarSpec(), CostWeights()
    e_minor = [40, 47, 52, 55, 59, 64]
    best = enumerate_shapes(e_minor, g, w)[0]
    assert best.barre is None
    assert {(p.string, p.fret) for p in best.placements} == {
        (0, 0), (1, 2), (2, 2), (3, 0), (4, 0), (5, 0)
    }


def test_barre_used_when_fingers_run_out():
    g, w = GuitarSpec(), CostWeights()
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
    events = [(m, i * 0.35, 0.3) for i, m in enumerate([72, 74, 76, 77, 79])]
    y = synth_notes(events, sr=cfg.transcribe.sr)
    doc, arranged, _ = run((y, cfg.transcribe.sr), cfg)
    assert len({n["string"] for n in doc["notes"]}) == 1
    positions = [g.shape.hand_position for g in arranged]
    assert positions == sorted(positions)          # hand only moves one way


def test_alternate_tuning_changes_the_tab():
    """D2 is below a standard guitar. Drop D reaches it on the open low string."""
    cfg_std = PipelineConfig()
    cfg_drop = PipelineConfig(guitar=GuitarSpec(tuning=Tuning.get("drop_d")))
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
    lo = min(cfg.guitar.tuning.open_midi)
    for n in doc["notes"]:
        assert n["midi"] >= lo


# ------------------------------------------------------------ robot output

def _demo_doc():
    cfg = PipelineConfig()
    events = [(m, i * 0.5, 0.45) for i, m in enumerate([64, 67, 71, 67, 64])]
    y = synth_notes(events, sr=cfg.transcribe.sr)
    return run((y, cfg.transcribe.sr), cfg)


def test_commands_are_time_ordered():
    doc, _, _ = _demo_doc()
    times = [c["t"] for c in doc["commands"]]
    assert times == sorted(times)
    assert [c["seq"] for c in doc["commands"]] == list(range(len(doc["commands"])))


def test_every_press_lands_before_its_pluck():
    doc, _, _ = _demo_doc()
    plucks = {c["note_id"]: c["t"] for c in doc["commands"]
              if c["arm"] == "pick" and "note_id" in c}
    for c in doc["commands"]:
        if c["arm"] == "fret" and c["action"] == "press":
            assert c["t"] <= plucks[c["note_id"]] + 1e-6, (
                f"finger lands after the pluck for {c['note_id']}"
            )


def test_no_string_is_pressed_twice_without_release():
    doc, _, _ = _demo_doc()
    held = {}
    for c in doc["commands"]:
        if c["arm"] != "fret":
            continue
        if c["action"] == "press":
            assert c["string"] not in held
            held[c["string"]] = c["fret"]
        elif c["action"] == "release":
            held.pop(c["string"], None)
        elif c["action"] == "release_all":
            held.clear()


def test_open_strings_produce_no_fret_commands():
    cfg = PipelineConfig()
    y = synth_notes([(64, 0.0, 0.6)], sr=cfg.transcribe.sr)   # open high E
    doc, _, _ = run((y, cfg.transcribe.sr), cfg)
    assert doc["notes"][0]["fret"] == 0
    assert doc["notes"][0]["finger"] == 0
    presses = [c for c in doc["commands"] if c.get("action") == "press"]
    assert presses == []


def test_hand_moves_are_feasible():
    doc, _, _ = _demo_doc()
    for c in doc["commands"]:
        if c.get("action") == "move":
            assert c["feasible"] is True


def test_document_shape():
    doc, _, _ = _demo_doc()
    assert doc["format"] == "robotab/1.0"
    for key in ("instrument", "timing", "robot", "stats", "notes", "commands", "tab"):
        assert key in doc
    for n in doc["notes"]:
        assert 0 <= n["string"] < cfgless_n_strings()
        assert 0 <= n["fret"] <= 15
        assert 0 <= n["finger"] <= 4
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
    doc, _, _ = run((y, sr), PipelineConfig())
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
    cfg = PipelineConfig(transcribe=TranscribeSpec(polyphonic=True))
    stacks = [((40, 47, 52, 55, 59, 64), 0.0), ((48, 52, 55, 60, 64), 1.2),
              ((43, 47, 50, 55, 59, 67), 2.4), ((50, 57, 62, 66), 3.6)]
    events = [(m, s, 1.2) for stack, s in stacks for m in stack]
    y = synth_notes(events, sr=cfg.transcribe.sr)
    doc, arranged, _ = run((y, cfg.transcribe.sr), cfg)
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
    assert doc["commands"] == []
    assert doc["format"] == "robotab/1.0"


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
