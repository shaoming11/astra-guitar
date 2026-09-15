"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (CostWeights, GuitarSpec, PipelineConfig, RobotSpec,
                     TranscribeSpec, Tuning, TUNINGS)
from .tabtext import render_events


DEMO_TUNE = [
    # (midi, start, duration) - "Twinkle" fragment, 100 bpm, quarter = 0.6 s
    (64, 0.00, 0.55), (64, 0.60, 0.55), (71, 1.20, 0.55), (71, 1.80, 0.55),
    (73, 2.40, 0.55), (73, 3.00, 0.55), (71, 3.60, 1.15),
    (69, 4.80, 0.55), (69, 5.40, 0.55), (68, 6.00, 0.55), (68, 6.60, 0.55),
    (66, 7.20, 0.55), (66, 7.80, 0.55), (64, 8.40, 1.15),
]

DEMO_CHORDS = [
    # Em, C, G, D voicings as pitch stacks
    ((40, 47, 52, 55, 59, 64), 0.0, 1.2),
    ((48, 52, 55, 60, 64), 1.2, 1.2),
    ((43, 47, 50, 55, 59, 67), 2.4, 1.2),
    ((50, 57, 62, 66), 3.6, 1.2),
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
        polyphonic=args.poly,
        min_note_seconds=args.min_note,
        min_confidence=args.min_confidence,
        chord_window=args.chord_window,
    )
    robot = RobotSpec(
        press_lead=args.press_lead,
        strum_stagger=args.strum_stagger,
        alternate_picking=not args.no_alternate_picking,
    )
    return PipelineConfig(guitar=guitar, weights=weights,
                          transcribe=transcribe_spec, robot=robot)


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-o", "--out", default=None, help="write the robot JSON here")
    p.add_argument("--tab", default=None, help="write the ASCII tab here")
    p.add_argument("--strtab", default=None,
                   help="write the compact STRING-FRET timing file here, "
                        "for a robotics protocol (see tabtext.render_string_tab)")
    p.add_argument("--preview", default=None,
                   help="render the result to a WAV with a plucked string model")
    p.add_argument("--tuning", default="standard", choices=sorted(TUNINGS))
    p.add_argument("--capo", type=int, default=0)
    p.add_argument("--max-fret", type=int, default=15)
    p.add_argument("--max-span", type=int, default=4)
    p.add_argument("--no-open-strings", action="store_true")
    p.add_argument("--poly", action="store_true", help="chord mode (experimental)")
    p.add_argument("--quantize", type=int, default=16,
                   help="snap to 1/N of a beat, 0 disables")
    p.add_argument("--min-note", type=float, default=0.045)
    p.add_argument("--min-confidence", type=float, default=0.5)
    p.add_argument("--chord-window", type=float, default=0.045)
    p.add_argument("--press-lead", type=float, default=0.060)
    p.add_argument("--strum-stagger", type=float, default=0.018)
    p.add_argument("--no-alternate-picking", action="store_true")
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
              f"{s.get('hand_shifts', 0)} hand shifts | "
              f"{len(doc['commands'])} robot commands")
        for wmsg in doc["warnings"][:8]:
            print(f"  warning: {wmsg}")
        if len(doc["warnings"]) > 8:
            print(f"  ... {len(doc['warnings']) - 8} more warnings")
    if args.out and not args.quiet:
        print(f"wrote {args.out}")


def cmd_transcribe(args) -> int:
    from .pipeline import run
    cfg = build_config(args)
    doc, arranged, _ = run(args.input, cfg, quantize=args.quantize or None)
    emit(doc, arranged, cfg, args)
    return 0


def cmd_demo(args) -> int:
    from .pipeline import run
    from .synth import synth_notes, write_wav

    cfg = build_config(args)
    sr = cfg.transcribe.sr
    if args.chords:
        cfg.transcribe.polyphonic = True
        events = [(m, s, d) for stack, s, d in DEMO_CHORDS for m in stack]
    else:
        events = DEMO_TUNE
    y = synth_notes(events, sr=sr)
    if args.write_audio:
        write_wav(args.write_audio, y, sr)
    doc, arranged, _ = run((y, sr), cfg, quantize=args.quantize or None)
    doc["source"]["file"] = "demo:chords" if args.chords else "demo:melody"
    emit(doc, arranged, cfg, args)
    return 0


def cmd_listen(args) -> int:
    from .pipeline import run
    from .transcribe import record_audio

    cfg = build_config(args)
    if not args.quiet:
        print(f"recording {args.seconds:.1f}s ...", file=sys.stderr)
    y = record_audio(args.seconds, cfg.transcribe.sr)
    doc, arranged, _ = run((y, cfg.transcribe.sr), cfg, quantize=args.quantize or None)
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
    problems = []
    if doc.get("format", "").split("/")[0] != "robotab":
        problems.append("not a robotab document")
    last_t = -1.0
    for c in doc.get("commands", []):
        if c["t"] < last_t - 1e-6:
            problems.append(f"commands out of order at seq {c.get('seq')}")
        last_t = c["t"]
    held = {}
    for c in doc.get("commands", []):
        if c["arm"] != "fret":
            continue
        if c["action"] == "press":
            if c["string"] in held:
                problems.append(f"double press on string {c['string']} at t={c['t']}")
            held[c["string"]] = c["fret"]
        elif c["action"] == "release":
            held.pop(c["string"], None)
        elif c["action"] == "release_all":
            held.clear()
    infeasible = [c for c in doc.get("commands", [])
                  if c.get("action") == "move" and c.get("feasible") is False]
    for c in infeasible:
        problems.append(f"infeasible hand move at t={c['t']} to fret {c['position']}")

    print(f"{doc.get('format')} | {len(doc.get('notes', []))} notes | "
          f"{len(doc.get('commands', []))} commands")
    if problems:
        for p in problems[:20]:
            print(f"  problem: {p}")
        return 1
    print("  ok: command stream is ordered, consistent and physically feasible")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="tunefinder",
        description="Listen to a tune and return a machine readable guitar tab.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_t = sub.add_parser("transcribe", help="transcribe an audio file")
    p_t.add_argument("input")
    add_common(p_t)
    p_t.set_defaults(func=cmd_transcribe)

    p_d = sub.add_parser("demo", help="run the pipeline on a synthetic tune")
    p_d.add_argument("--chords", action="store_true")
    p_d.add_argument("--write-audio", default=None)
    add_common(p_d)
    p_d.set_defaults(func=cmd_demo)

    p_l = sub.add_parser("listen", help="record from the microphone")
    p_l.add_argument("--seconds", type=float, default=8.0)
    add_common(p_l)
    p_l.set_defaults(func=cmd_listen)

    p_v = sub.add_parser("validate", help="check a robotab JSON file")
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
