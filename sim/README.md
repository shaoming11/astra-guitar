# sim/ — MuJoCo rig

Two SO-101 arms + a guitar. Arm A frets the A string, arm B plucks. Works with no
hardware, and the agent loop is identical to the real rig's.

```bash
.venv/bin/python sim/play.py press_mm=2.6 behind_mm=7.0 pluck=0.6 wav=out.wav
.venv/bin/python sim/record.py runs/my-run      # mp4 + audio
```

## How the guitar works

MuJoCo cannot simulate a vibrating string, and does not need to. The guitar geometry is
**visual only** (`contype=0 conaffinity=0`) — collisions would just block the arms from
reaching the strings. What the sim provides is *geometry and timing*; `string_model.py`
maps tip position to sound, reproducing the failure modes a real guitar has:

| condition | result |
|---|---|
| press depth < 1.2 mm | string not stopped → open string |
| press depth < 2.0 mm | buzz |
| press > 20 mm behind the wire | buzz, and at high frets lands on the *next* fret |
| press > 6 mm | string bent sharp |
| pluck speed < 0.02 | silent |

Pitch is `110 Hz × 2^(fret/12)` (A string). Audio is a decaying harmonic stack, mixed down
so the same scorer works on sim and real audio.

## Files
- `build_scene.py` — composes the scene with `MjSpec.attach` (two arms, prefixes `A_`/`B_`).
  Equal-tempered fret positions: `fret_x(n) = L(1 − 2^(−n/12))`, L = 0.648 m.
- `ik.py` — damped-least-squares IK on each arm's tool-tip site. The real rig has no IK and
  uses hand-recorded poses; in sim we can solve, so `fret(n)` works for any n.
- `string_model.py` — geometry → note, quality, audio.
- `rig.py` — `fret(n, press_mm, behind_mm)` · `release()` · `pluck(speed)` · `observe()`.
- `song.py` — Seven Nation Army: frets 7 7 10 7 5 3 2 → E3 E3 G3 E3 D3 C3 B2.
- `scorer.py` — compact JSON the agent sees: per note want/heard/quality/why.
- `agent_loop.py` — logs each attempt to `runs/<name>/` (wav + decisions.jsonl).
- `record.py` — renders mp4 with muxed audio.

## Run log: Fable 5.1 learning the riff

Starting from deliberately bad parameters on three axes. One variable changed per turn.

| turn | change | result |
|---|---|---|
| 1 | press 1.0 mm, 22 mm behind fret, pluck 0.015 | 0/7 — half silent, half open-string |
| 2 | pluck 0.015 → 0.6 (can't diagnose through silence) | 0/7 — all audible now, all bad |
| 3 | press 1.0 → 2.6 mm (notes read as fret 0 = not stopping the string) | 0/7 — but fret 10 came out as **fret 11** |
| 4 | behind 22 → 7 mm (frets narrow toward the bridge; fixed offset overshoots the wire) | **7/7 clean** |

Audio verified by FFT on the rendered waveform: 164, 164, 196, 164, 148, 132, 124 Hz.

## Known gap
Note *timing* is not yet controlled — `song.py` emits `at_ms` but `rig.py` plays each note as
fast as IK settling allows, so the riff comes out evenly spaced. Adding a scheduler is the
next step, and gives the agent a second thing to learn (servo lag pre-compensation).
