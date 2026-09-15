# astra-guitar — Design

Goal: GPT-6 Astra controls an SO-101 arm to strum open-tuned chords on a real guitar,
and self-corrects from microphone + camera feedback — the guitar version of
thijs (@cdngdev)'s Golden Gate painting timelapse.

## 1. What the painting demo actually was

- Hardware: SO-101 (LeRobot 5-DoF arm) holding a brush, one fixed camera aimed at the canvas.
- Brain: `gpt-6-astra` via the OpenAI **Responses API** (function calling only works there).
  Limited rollout as of 2026-09-04; $10 / $50 per MTok in/out.
- The loop:
  1. Camera frame + goal + small tool vocabulary → Astra.
  2. Astra emits strokes as tool calls; code executes them through LeRobot.
  3. New frame → back to Astra with "here's what you did, here's the goal, what next?"
  4. Repeat. "Self-correct" = model comparing its own frame to its intent and revising.
- innate-os PR #817 pattern: minimal Responses body, only tools it can really call in the
  prompt, workspace/joint guards that reject bad targets before motion, telemetry re-read
  after the model decides.
- Robocurve eval settings worth copying: medium thinking, ~20-call budget, 25% speed cap.

## 2. Mapping to guitar

Painting is judged by a camera; guitar is judged by a **microphone**. Audio is easier to
score than an image — give Astra a number it can push up.

- Scope: strum open-tuned chords with one arm. Open-G tuning (D-G-D-G-B-D) → open strings
  are G major; a capo turns that into any major chord. Human moves the capo between segments.
- Tool vocabulary (3 tools, `strict: true`, numeric limits in the schema):
  - `play(pattern: [{dir:"down"|"up", at_ms:int}], depth_mm: 0–15, speed: 0.2–1.0)`
    — Astra writes the rhythm; the interpolator executes it on schedule and the scorer grades
    timing against the plan Astra itself wrote (see sense/README.md).
  - `done(reason)`
- Astra tunes `pattern`, `depth_mm` and `speed` (the "brush pressure" analog). It never sees joints.

## 3. Architecture

```
mic ──► scorer ─┐
                ├─► Astra (Responses API, tools) ─► guard ─► LeRobot SO101Follower
cam ──► frame ──┘         ▲                                          │
                          └────── observation after every action ◄──┘
```

### Robot layer (LeRobot)
- `SO101FollowerConfig(port, id, cameras)` → `SO101Follower(cfg)` → `connect()`.
- `get_observation()` → `{"<motor>.pos": deg, ..., "<cam>": ndarray}`.
- `send_action({"<motor>.pos": deg, ...})`.
- No built-in IK → do **not** expose Cartesian `move_to`. Record named poses instead:
  `above_strings`, `strum_start`, `strum_end`, `rest`.
- `strum()` = linear interpolation strum_start → strum_end; `depth_mm` offsets wrist_flex
  (degrees-per-mm measured empirically); `speed` sets step delay.

### Scorer (replaces "look at the canvas")
After each strum, record ~1.2 s and compute:
- RMS loudness (did strings get hit at all — #1 failure mode is depth too shallow).
- Chroma / pitch-class energy (`librosa.chroma_cqt`) vs target chord → `chord_match` 0–1
  plus `missing` notes (missing high strings = strum stopped short; missing low = started deep).
- Onset count (all 6 rang vs pick snagged).
Return as the `function_call_output` string, e.g.
`{"rms":0.04,"chord_match":0.55,"missing":["B","E"],"onsets":3,"note":"strum ended early"}`.

### Astra loop (Responses API)
- `client.responses.create(model="gpt-6-astra", instructions, tools, input)`.
- Each turn: execute every `function_call` item; append `function_call_output` with matching
  `call_id` containing scorer JSON **and** an `input_image` (base64 data URL) of the frame.
- Use `previous_response_id` for continuity; only send new items per turn.
- `reasoning.effort = "medium"`; cap turns (~20), spend, and wall time.
- Retry only on 429/5xx; never retry a turn whose tool already executed.
- System prompt: physical setup (tuning, capo, pick, camera view), goal, metric semantics
  ("chord_match ≥ 0.85 and onsets ≥ 5 is clean"), and "change one parameter at a time and
  say why" — that text is the timelapse narration.

### Guards (separate module, pure functions, unit-tested with no hardware)
- Clamp every interpolated target to calibrated range AND a tighter workspace box recorded
  around the strings; reject the whole strum if any step leaves it (no silent clamping).
- Re-read `get_observation()` before executing; abort if arm isn't within a few degrees of
  where the last action left it.
- Speed cap independent of what the model asks for.
- Keyboard kill switch → `rest` + disconnect.

## 4. What Astra sees per turn (observation packet)

Raw joint angles alone don't work — no reference frame. Send, in fixed order:
1. Named-pose state: `arm: at strum_end (within 2°)`, `last_strum: depth_mm=6, speed=0.5, direction=down`.
2. Raw joints labeled with the recorded reference next to them (drift / stall detection only).
3. Camera frame (bridge-height view) + scorer JSON.

| You implement (deterministic)                    | Astra decides         |
|--------------------------------------------------|-----------------------|
| depth_mm → wrist_flex offset                      | what depth to try     |
| speed → interpolator step delay                   | what speed to try     |
| pose interpolation, clamps, workspace box         | nothing               |
| "arm is at pose X" summarizer                     | reading that summary  |

## 5. Camera + mic placement

- Guitar flat on the table, face up, body clamped so the arm can't walk it.
- **One camera at string height, ~20–30 cm off the end of the bridge, looking up the neck.**
  Six strings stack vertically; pick depth reads as distance below the top string line.
  Overhead is useless for depth — only use it as a recording camera for the timelapse.
- Optional wrist camera (LeRobot mount) — nice-to-have.
- White paper/tape behind the strings on the far side for contrast.
- Guitar is an unplugged electric → quiet; MacBook must sit within ~20–30 cm of the strings.
- MacBook camera = OpenCV index 0; grab one frame per turn with `cv2.VideoCapture(0)`.
  MacBook mic via `sounddevice`. Grant macOS camera/mic permission to the terminal first;
  check RMS is nonzero before blaming the scorer.
- Downscale frames to ~768 px wide, JPEG q≈70 before base64.

## 6. Repo infrastructure

```
astra-guitar/
  robot/       poses.json, interpolator, guards, so101 wrapper, fake_robot
  sense/       camera grab, mic record, chord scorer
  agent/       tools schema, prompt builder, responses loop, budget
  runs/        one folder per run: frames, wavs, decisions.jsonl
  dash/        local web page that tails a run (the timelapse view)
  scripts/     calibrate, record_poses, replay
```

1. **Observation packet builder** — one function, fixed order, stored per turn for replay.
2. **Tool registry** — `(json_schema, callable, summarizer)` per tool; loop dispatches by name.
3. **FakeRobot + canned scorer** — same dict shapes; run the whole Astra loop with no hardware.
4. **Run recorder** — per turn: `frame_NNN.jpg`, `strum_NNN.wav`, `decisions.jsonl`
   `{turn, tool_calls, args, scorer, arm_state, model_text, latency_ms, tokens}`.
5. **Responses loop** — previous_response_id, budgets (turns/spend/time), safe retries, effort flag.
6. **Dashboard** — streams `output_text.delta` + `output_item.done`, latest frame, chord_match
   sparkline over SSE/WebSocket. Reads from the run folder → works on replays too.
7. **Guards module** — the only path to the servo bus.
8. **Scripts** — `record_poses`, `calibrate_depth`, `test_scorer`, `replay <run> <turn>`.

## 7. Build order

1. Recorder + FakeRobot + Responses loop (software only).
2. Scorer against hand strumming; validate mic + permissions.
3. Real poses + interpolator + guards (needs the arm).
4. Connect real strum; start deliberately too shallow so there's something to correct.
5. Dashboard last. Screen-record it next to the arm — that's the demo.

## Sources
- https://x.com/cdngdev/status/2097339677128982873
- https://github.com/zjwzcx/Awesome-Astra-Embodied-AI
- https://openai.robocurve.org/gpt-6-astra/
- https://github.com/innate-inc/innate-os/pull/817
- https://github.com/Anil-matcha/awesome-gpt-6-astra
- https://github.com/huggingface/lerobot/blob/main/docs/source/il_robots.mdx
- https://developers.openai.com/api/docs/guides/function-calling
