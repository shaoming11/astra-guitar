# sense/ — what Astra hears and sees

Decisions so far (2026-09-15):
- Guitar: **electric, unplugged**. Quiet, thin signal → mic must be close (MacBook at bridge
  height, ~20–30 cm from the strings). Expect SNR to be the main problem, not pitch.
- Audio input: MacBook built-in mic via `sounddevice`, default device, 44.1 kHz mono.
- Camera: MacBook camera at bridge height, one frame per turn, sent to Astra with the scorer.
- Dev mode: live capture only; validate by strumming by hand.

## Scorer outputs (chosen)
1. **Onset count / snag detection** — number of distinct string onsets inside the strum window.
   Clean sweep ≈ 5–6 onsets within ~150 ms; a snag shows as 1–2 onsets then a gap; too shallow
   shows as 0. Needs a noise-floor estimate first (see Calibration).
2. **Timing / rhythm accuracy** — for multi-strum patterns: onset time of the first string vs the
   expected beat time. Requires a shared clock between the interpolator (when the strum was
   commanded) and the audio buffer (when it was heard). Report `offset_ms` per strum.

Implicitly required by both (they fall out for free):
- RMS over the window → "did it ring at all" and the noise floor.
- (Recommended, not yet chosen) chroma vs target chord → tells Astra *which end* of the strum
  fell short, which is the direction signal for `depth_mm`. Onset count alone says "3 strings"
  but not which 3.

## Modules (to be written)
- `mic.py`      — open stream once, ring buffer; `capture(window_s)` returns the last N seconds
                  aligned to a timestamp so timing can be measured against the command clock.
- `onsets.py`   — noise floor, onset detection in the window, snag heuristic.
- `timing.py`   — expected beat times from the pattern + tempo; `offset_ms` per detected onset.
- `camera.py`   — `grab()` → one downscaled JPEG (≈768 px wide, q≈70) as bytes + base64.
- `packet.py`   — assemble the scorer JSON string Astra gets as `function_call_output`.
- `calibrate.py`— 2 s of silence → noise floor; 3 hand strums → onset threshold sanity print.

## Calibration (run at the table before every session)
1. Servo whine: record 2 s with the arm holding `above_strings` under torque — that's the floor,
   not silence. Threshold = floor × k.
2. Hand strum 3×; print onsets and RMS. If onsets < 4 on a clean hand strum, move the laptop closer
   or mute the room.
3. Check macOS mic/camera permission for the terminal (nonzero RMS, non-black frame).

## Timing decisions
- **Clock:** command timestamp. `t_cmd` = `time.monotonic()` captured by the interpolator the
  instant the first `send_action` of a strum goes out. Every onset is reported relative to it.
  No metronome, no backing track — nothing else makes sound in the room except servos.
- **Alignment:** always-on ring buffer. `mic.py` opens one `sounddevice.InputStream` at session
  start with a callback that appends to a ring buffer (~10 s) and stores the monotonic time of
  the first sample. `capture(t_cmd, pre=0.1, post=1.2)` slices by time, not by "start recording
  now" — stream-startup latency never touches the measurement.
  Measure and subtract fixed input latency once (`stream.latency`); it's usually 10–40 ms on
  the built-in mic.
- **Pattern:** Astra picks it. `strum` becomes
  `play(pattern: [{"dir":"down"|"up","at_ms":int}], depth_mm, speed)` — a list of strums with
  intended onset times relative to the first. The interpolator executes them on that schedule
  and records the actual `t_cmd` of each. The scorer then reports, per intended strum, the
  nearest detected onset and `offset_ms` = heard − intended. Astra is graded against the plan
  *it* wrote, so timing error is purely mechanical (servo lag, snag), which is what it can fix.

## Scorer JSON — compact (to Astra, as `function_call_output`)
```
{
  "rang": true,               // rms over floor
  "strums": [
    {"dir":"down","onsets":5,"offset_ms":+38,"snag":false},
    {"dir":"up",  "onsets":2,"offset_ms":+210,"snag":true}
  ],
  "note": "2nd strum snagged; late by 210ms"   // one-line rule-based summary
}
```
Keep it to these fields. Recommended extra if chord-match is enabled later:
`"missing":["B","E"]` per strum.

## Scorer JSON — full (to `runs/<run>/decisions.jsonl` only)
Everything above plus: `rms_db`, `floor_db`, `onset_times_ms` (absolute, relative to t_cmd),
`onset_strengths`, detector params (hop, threshold, k), `t_cmd` per strum, `mic_latency_ms`,
window bounds, and the frame filename. This is what `replay` and the dashboard read.

## Snag heuristic (first cut)
- Window per strum: `[intended_at − 60 ms, next_intended_at − 60 ms)` or +400 ms for the last.
- `onsets` = peaks of a spectral-flux onset envelope above `floor × k` inside the window.
- `snag` = onsets ≥ 1 and (max inter-onset gap > 120 ms or onsets < 3).
- `offset_ms` = first onset in window − intended time; `null` if no onset.
Tune k and the 120 ms gap against hand strums in `calibrate.py`, not against the arm.
