"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (CostWeights, GuitarSpec, PipelineConfig,
                     TranscribeSpec, Tuning, TUNINGS)
from .tabtext import render_events


DEMO_TUNE = [
    # (midi, start, duration) - "Twinkle" fragment, 100 bpm, quarter = 0.6 s
    (64, 0.00, 0.55), (64, 0.60, 0.55), (71, 1.20, 0.55), (71, 1.80, 0.55),
    (73, 2.40, 0.55), (73, 3.00, 0.55), (71, 3.60, 1.15),
    (69, 4.80, 0.55), (69, 5.40, 0.55), (68, 6.00, 0.55), (68, 6.60, 0.55),
    (66, 7.20, 0.55), (66, 7.80, 0.55), (64, 8.40, 1.15),
]

def build_config(args) -> PipelineConfig:
    guitar = GuitarSpec(
        tuning=Tuning.get(args.tuning),
        max_fret=args.max_fret,
        max_span=args.max_span,
        allow_open_strings=not args.no_open_strings,
        capo=args.capo,
    )
    weights = CostWeights()
    if args.prefer_low_frets:
        weights.fret_height *= 2.0
    if args.minimise_shifts:
        weights.hand_shift *= 2.0
    transcribe_spec = TranscribeSpec(
        min_note_seconds=args.min_note,
        min_confidence=args.min_confidence,
        voice_low_hz=args.voice_low_hz,
        voice_high_hz=args.voice_high_hz,
        noise_reduction_strength=args.noise_reduction_strength,
        voice_gate_strength=args.voice_gate_strength,
    )
    return PipelineConfig(guitar=guitar, weights=weights,
                          transcribe=transcribe_spec)


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--out", default=None, help="write the timed transcription JSON here")
    p.add_argument("--tab", default=None, help="write the ASCII tab here")
    p.add_argument("--strtab", default=None,
                   help="write the compact STRING-FRET timing file here")
    p.add_argument("--preview", default=None,
                   help="render the result to a WAV with a plucked string model")
    p.add_argument("--tuning", default="standard", choices=sorted(TUNINGS))
    p.add_argument("--capo", type=int, default=0)
    p.add_argument("--max-fret", type=int, default=15)
    p.add_argument("--max-span", type=int, default=4)
    p.add_argument("--no-open-strings", action="store_true")
    p.add_argument("--quantize", type=int, default=16,
                   help="snap to 1/N of a beat, 0 disables")
    p.add_argument("--min-note", type=float, default=0.045)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--voice-isolation", action="store_true",
                   help="foreground-clean a vocal/hummed source before tracking")
    p.add_argument("--raw-mic", action="store_true",
                   help="disable microphone voice isolation (debugging)")
    p.add_argument("--voice-low-hz", type=float, default=65.0)
    p.add_argument("--voice-high-hz", type=float, default=2600.0)
    p.add_argument("--noise-reduction-strength", type=float, default=1.6)
    p.add_argument("--voice-gate-strength", type=float, default=1.8)
    p.add_argument("--prefer-low-frets", action="store_true")
    p.add_argument("--minimise-shifts", action="store_true")
    p.add_argument("--events", action="store_true", help="print the event table")
    p.add_argument("--quiet", action="store_true")


def emit(doc, arranged, cfg, args) -> None:
    from .synth import render_tab_audio, write_wav
    from .tabtext import render_string_tab

    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2))
    if args.tab:
        Path(args.tab).write_text(doc["tab"] + "\n")
    if args.strtab:
        Path(args.strtab).write_text(render_string_tab(doc) + "\n")
    if args.preview:
        y = render_tab_audio(doc["notes"], sr=22050)
        write_wav(args.preview, y, 22050)

    if not args.quiet:
        print(doc["tab"])
        print()
        if args.events:
            print(render_events(arranged, cfg))
            print()
        s = doc["stats"]
        print(f"tempo {doc['timing']['tempo_bpm']} bpm | "
              f"{s.get('notes', 0)} notes in {s.get('groups', 0)} events | "
              f"max fret {s.get('max_fret', 0)} | "
              f"{s.get('hand_shifts', 0)} hand shifts")
        for wmsg in doc["warnings"][:8]:
            print(f"  warning: {wmsg}")
        if len(doc["warnings"]) > 8:
            print(f"  ... {len(doc['warnings']) - 8} more warnings")
    if args.out and not args.quiet:
        print(f"wrote {args.out}")


def cmd_transcribe(args) -> int:
    from .pipeline import run
    from .media import MediaInputError

    cfg = build_config(args)
    try:
        doc, arranged, _ = run(
            args.input, cfg, quantize=args.quantize or None,
            voice_isolation=args.voice_isolation,
            force_single_notes=True,
        )
    except MediaInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    emit(doc, arranged, cfg, args)
    return 0


def cmd_demo(args) -> int:
    from .pipeline import run
    from .synth import synth_notes, write_wav

    cfg = build_config(args)
    sr = cfg.transcribe.sr
    events = DEMO_TUNE
    y = synth_notes(events, sr=sr)
    if args.write_audio:
        write_wav(args.write_audio, y, sr)
    doc, arranged, _ = run(
        (y, sr), cfg, quantize=args.quantize or None,
        force_single_notes=True,
    )
    doc["source"]["file"] = "demo:melody"
    emit(doc, arranged, cfg, args)
    return 0


def cmd_listen(args) -> int:
    from .pipeline import run
    from .transcribe import record_audio

    cfg = build_config(args)
    if not args.quiet:
        print(f"recording {args.seconds:.1f}s ...", file=sys.stderr)
    y = record_audio(args.seconds, cfg.transcribe.sr)
    doc, arranged, _ = run(
        (y, cfg.transcribe.sr), cfg, quantize=args.quantize or None,
        voice_isolation=not args.raw_mic,
        force_single_notes=True,
    )
    doc["source"]["file"] = "microphone"
    emit(doc, arranged, cfg, args)
    return 0


def cmd_ui(args) -> int:
    from .ui import launch

    guitar = GuitarSpec(
        tuning=Tuning.get(args.tuning),
        max_fret=args.max_fret,
        capo=args.capo,
    )
    cfg = PipelineConfig(guitar=guitar)
    launch(cfg, outdir=args.outdir, quantize=args.quantize or None)
    return 0


def cmd_validate(args) -> int:
    doc = json.loads(Path(args.file).read_text())
    problems = validate_document(doc)

    print(f"{doc.get('format')} | {len(doc.get('notes', []))} notes")
    if problems:
        for p in problems[:20]:
            print(f"  problem: {p}")
        return 1
    print("  ok: notes are ordered, non-overlapping and physically placed")
    return 0


def validate_document(doc: dict) -> list[str]:
    """Return structural and timing problems in a transcription document."""
    problems = []
    if doc.get("format") != "tunefinder/1.0":
        problems.append("not a tunefinder transcription document")
        return problems
    notes = doc.get("notes", [])
    if not isinstance(notes, list):
        return problems + ["notes must be a list"]
    previous_end = -1e-6
    previous_onset = -1e-6
    for index, note in enumerate(notes):
        for field in ("onset_s", "duration_s", "midi", "string", "fret"):
            if field not in note:
                problems.append(f"note {index} is missing {field}")
        if not all(field in note for field in ("onset_s", "duration_s", "string", "fret")):
            continue
        try:
            onset = float(note["onset_s"])
            duration = float(note["duration_s"])
            string = int(note["string"])
            fret = int(note["fret"])
        except (TypeError, ValueError):
            problems.append(f"note {index} has a non-numeric timing or placement")
            continue
        end = onset + duration
        if onset < previous_onset - 1e-6:
            problems.append(f"notes out of order at note {index}")
        if onset < previous_end - 1e-6:
            problems.append(f"overlapping notes at note {index}")
        if duration <= 0:
            problems.append(f"note {index} has non-positive duration")
        if string < 0 or fret < 0:
            problems.append(f"note {index} has a negative string or fret")
        previous_onset = onset
        previous_end = max(previous_end, end)
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="tunefinder",
        description="Listen to a tune and return a machine readable guitar tab.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_t = sub.add_parser("transcribe", help="transcribe local media or a YouTube URL")
    p_t.add_argument("input")
    add_common(p_t)
    p_t.set_defaults(func=cmd_transcribe)

    p_d = sub.add_parser("demo", help="run the pipeline on a synthetic tune")
    p_d.add_argument("--write-audio", default=None)
    add_common(p_d)
    p_d.set_defaults(func=cmd_demo)

    p_l = sub.add_parser("listen", help="record from the microphone")
    p_l.add_argument("--seconds", type=float, default=8.0)
    add_common(p_l)
    p_l.set_defaults(func=cmd_listen)

    p_v = sub.add_parser("validate", help="check a timed transcription JSON file")
    p_v.add_argument("file")
    p_v.set_defaults(func=cmd_validate)

    p_u = sub.add_parser("ui", help="rudimentary start/stop recording window")
    p_u.add_argument("--tuning", default="standard", choices=sorted(TUNINGS))
    p_u.add_argument("--capo", type=int, default=0)
    p_u.add_argument("--max-fret", type=int, default=15)
    p_u.add_argument("--outdir", default="recordings",
                     help="directory each take's files are written to")
    p_u.add_argument("--quantize", type=int, default=16,
                     help="snap to 1/N of a beat, 0 disables")
    p_u.set_defaults(func=cmd_ui)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
