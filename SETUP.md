# astra-guitar — connecting everything

End-to-end wiring for the demo: two SO-101 arms, MacBook camera + mic, GPT-6 Astra, and
π0.5 as the System-1 proposer. Read `CONNECT.md` for the arm-level details (ports,
calibration, motor IDs); this file is the whole-system map.

```
                         ┌──────────────────────────────┐
                         │  GPU box (cloud/laptop)       │
                         │  lerobot policy_server        │
                         │  π0.5  hqfang/pi05-so100_101  │
                         └──────────────▲───────────────┘
                                        │ obs → 50-step chunk (gRPC, LAN/tunnel)
┌───────────────────────────────────────┴──────────────────────────────────────┐
│  MacBook (this repo, .venv)                                                  │
│                                                                              │
│   sense/   camera (bridge view) ─┐                                           │
│            mic ring buffer ──────┤                                           │
│                                  ▼                                           │
│   agent/   proposer ──► observation packet ──► Astra (Responses API) ──┐     │
│            (pose table | π0.5)         accept / correct tool calls     │     │
│                                                                        ▼     │
│   robot/   guards ──► interpolator ──► FeetechMotorsBus ──► USB serial       │
└──────────────────────────────────────────────────────────────┬───────────────┘
                                                               │ /dev/cu.usbmodem…
                                              ┌────────────────▼───────────────┐
                                              │  Waveshare bus board + PSU      │
                                              │  arm A (fret)  IDs 1–6          │
                                              │  arm B (pluck) IDs 7–12 chained │
                                              └─────────────────────────────────┘
```

## 0. Physical layout

- Guitar flat, face up, on a box so the strings sit at ~half arm height; body clamped.
- Foam under 5 strings at the bridge; **A string live** (Seven Nation Army: frets 7 7 10 7 5 3 2).
- **Arm A (fret)** base beside the neck, centered on fret ~6, reaching sideways; rubber eraser
  glued to the gripper tip.
- **Arm B (pluck)** base beside the body near the pickups; pick clamped (scripts/grip_pick.py).
- **MacBook** at the bridge end, lid angled so the camera is at string height looking up the
  neck. Mic is then ~20 cm from the pick. White tape behind the strings for contrast.
- Workspaces never overlap: A owns the neck, B owns the body. Same rule in `robot/guards.py`.

## 1. Arms — one bus board, both arms

1. Arm A: already verified on `/dev/cu.usbmodem5A7C1220421` (IDs 1–6). Calibrate it
   (`CONNECT.md` §4) — the July `my_arm.json` is broken except for the gripper.
2. Arm B has no board. Chain it onto A's board:
   - One arm-B servo at a time on the board, run the ID-assign script (to be written:
     `scripts/set_motor_id.py --from 1 --to 7`, then 8 … 12).
   - Reconnect the chain, plug B's first servo into A's last with a long 3-pin cable.
   - `broadcast_ping` should list IDs 1–12.
3. Arm B talks through a raw `FeetechMotorsBus` motor map with IDs 7–12 (LeRobot's
   `SO101Follower` hardcodes 1–6). Calibrate B by recording min/max per joint with the raw bus.
4. Power: one PSU for 12 servos is fine for plucking; if arm A browns out during a press,
   split onto a second Waveshare board (order one as backup).

## 2. Sensing (MacBook)

- Camera: OpenCV index 0; one frame per turn, downscaled to 768 px, JPEG q70.
- Mic: `sounddevice` InputStream opened once at session start, ring buffer, monotonic
  timestamps. Scorer slices `[t_cmd − 0.1 s, t_cmd + 1.2 s]`.
- Grant macOS camera + mic permission to **the terminal app you run from** (first run prompts).
- Validate with `sense/calibrate.py`: 2 s of servo-under-torque noise floor → threshold;
  3 hand plucks → onset + pitch sanity print.
- Full spec: `sense/README.md`.

## 3. π0.5 (System 1 proposer)

### 3.0 Decision (2026-09-15): parked, not deployed
Zero-shot π0.5 doesn't learn from the mic and its prior is pick-and-place; the paper's
RoboLab result says an off-task prior makes Astra *worse*. The pose table is our real
System 1. Everything below is the recipe for the GPU box if we want the comparison chart;
only the 4 MB of metadata (README, tokenizer, processor configs) is kept under
`pi05/checkpoints/pi05-so100_101/` for reference — the 16.6 GB weights were not pulled.

### 3.1 Checkpoint
`hqfang/pi05-so100_101` — π0.5 trained on 1,209 community SO-100/101 datasets; the only
checkpoint that knows this arm zero-shot. Facts from its README:
- **16.6 GB**, 4.1B params FP32. Download on the GPU box, not the laptop:
  `hf download hqfang/pi05-so100_101 --local-dir ./pi05-so100_101`
- **Needs a patched LeRobot**: clone lerobot, `git checkout b6ec0060779550c0a157ae34feb89e0cf86012a8`,
  `git apply pi05-so100_101/code/lerobot.patch`, `pip install -e '.[pi]'` — in its **own env**
  (Python 3.12, transformers 5.5.4). Incompatible with this repo's `.venv` (3.10 / 4.57).
- Load with the saved processors (`make_pre_post_processors(cfg, pretrained_path=root)`),
  override `cfg.text_tokenizer_name` to `root/tokenizer/tokenizer.model`, `compile_model=False`.
- I/O contract: `task` text; `observation.state` = 6 joints in our order
  (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper), raw units;
  1–4 RGB views as `observation.images.camera_0..3` in [0,1] (letterboxed to 224), pad slots
  flagged with `camera_N_is_pad`; output = 50-step chunk of absolute joint targets. Reset
  between episodes.

The MacBook `.venv` has `lerobot[transformers-dep,async]` (grpc client side) — enough to talk
to a policy server, not to run the model.

### 3.2 Where it runs — NOT CONNECTED YET
Inference needs an NVIDIA GPU (RTX 5070 Ti-class or a cloud L4/A10). The MacBook can run it
on MPS at ~5–10 s per chunk, acceptable only for a one-note-per-turn demo.

**GPU box** (patched env from §3.1):
```bash
hf download hqfang/pi05-so100_101 --local-dir ./pi05-so100_101
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080 --fps=30
```

**MacBook:** do not use `lerobot.async_inference.robot_client` directly — it drives the arm
itself, bypassing Astra and the guards. Instead `agent/proposer.py::Pi05Proposer` opens the
same gRPC channel to `<gpu-host>:8080`, sends `{observation.state (6-D, arm A),
observation.images.front (bridge cam), task: "press the A string at fret 7"}` and receives a
50-step joint chunk. Cloud box → expose 8080 over Tailscale or an SSH tunnel
(`ssh -L 8080:localhost:8080 user@gpu`), never the open internet.

### 3.3 What to expect
Single-arm only (6-D action; the paper's bimanual 14-D setup doesn't apply). Proposals will
reach toward the neck and be wrong about fret and pressure most of the time — it has never
seen a guitar. That's fine: the demo is Astra's accept/correct rate on π0.5 vs. the pose
table, the paper's experiment in miniature.

## 4. Proposer → Astra → guards (the loop)

Per note:
1. `proposer.propose(obs)` → joint chunk + a text summary ("wrist ends 4 mm above fret 6,
   press 1.5 mm") + `source` (`pose_table` | `pi05`).
2. Observation packet = arm summary + proposal summary + scorer JSON from the last note +
   camera frame. Fixed order, stored to `runs/<run>/`.
3. Astra (Responses API, `gpt-6-astra`, `reasoning.effort=medium`) gets tools:
   `accept(steps: 1–15)` · `correct(fret, press_mm, velocity, at_ms)` · `load_song` · `done`.
4. Chosen chunk → `robot/guards.py` (workspace box, joint clamp, speed cap, "arm is where I
   left it" check) → interpolator → bus. Rejected proposals are reported back to Astra as text.
5. Pluck (arm B), scorer runs, loop.

`FakeRobot` + canned scorer let the whole loop run with no hardware; do that first.

## 5. Astra

- `OPENAI_API_KEY` in the environment (never in the repo). Model id `gpt-6-astra`.
- Responses API only; `previous_response_id` for continuity; budgets: 20 turns / $X / 10 min.
- Retry only on 429/5xx; never re-run a turn whose tool already executed.

## 6. Bring-up order

1. `FakeRobot` + Astra loop + run recorder (no hardware, no GPU).
2. `sense/calibrate.py` with hand plucks.
3. Arm A calibrated; `scripts/record_poses.py` → `frets.json`; one clean note at fret 7 by hand-tuned `press_mm`.
4. Arm B chained (IDs 7–12), pick clamped, pluck pose recorded.
5. Astra on the pose-table proposer → it learns the riff.
6. GPU box up, `Pi05Proposer` swapped in, correction-rate comparison on the dashboard.
