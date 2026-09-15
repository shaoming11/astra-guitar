# Connecting the SO-101 arms

Two arms: **A = fretting** (holds a string down at a fret), **B = plucking** (holds the pick).
Both are SO-101 followers — there is no leader/teleop arm in this project. Each has its own
USB cable and its own power brick.

## 0. Hardware checklist (do this before anything else)

- [ ] Power brick plugged into the arm's servo bus board **and** the wall. The USB serial
      chip is on that board; with no power the Mac often sees nothing at all.
- [ ] USB-C cable is a **data** cable (many are charge-only). Swap it if the port never appears.
- [ ] Plugged directly into the Mac, not through a hub, for the first test.
- [ ] Servo daisy-chain cables clicked in; nothing hot; arm free to move.

Verify the Mac sees it:

```bash
ls /dev/cu.usb*
```

You want something like `/dev/cu.usbmodem5AB90687491` (one per arm). If it prints
`no matches found`, the arm hasn't enumerated — go back to the checklist. For a deeper look:

```bash
system_profiler SPUSBDataType | grep -iE "usbmodem|serial|ch34|cp210|product"
```

## 1. Software (one time)

Python 3.10 via pyenv is already on this machine; lerobot needs 3.10–3.12.

```bash
cd ~/Downloads/hackathons/astra-guitar && uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python "lerobot[feetech]"
```

Then for every new terminal:

```bash
source ~/Downloads/hackathons/astra-guitar/.venv/bin/activate
```

## 2. Find each arm's port

Plug in **one arm at a time** and run:

```bash
lerobot-find-port
```

It asks you to unplug the USB, press Enter, and reports the port that disappeared. Write it down:

| arm | role  | port                          | id           |
|-----|-------|-------------------------------|--------------|
| A   | fret  | /dev/cu.usbmodem5A7C1220421   | fret_arm     |  (verified 2026-09-15: 6× STS3215, IDs 1–6)
| B   | pluck | /dev/cu.usbmodem___________   | pluck_arm    |

macOS port names are stable for a given board+USB port, so keep each arm on the same
physical Mac port for the hackathon.

## 3. Set motor IDs (only if the arm was never configured)

A factory SO-101 kit sometimes ships with every servo at ID 1. If calibration complains about
duplicate IDs, run this with **only one servo** connected at a time and follow the prompts:

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/cu.usbmodemXXXX
```

Skip this if the arm came pre-assembled and tested.

## 4. Calibrate (once per arm; saved under ~/.cache/huggingface/lerobot/calibration)

Put the arm roughly mid-range on every joint first, then:

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/cu.usbmodemXXXX --robot.id=fret_arm
```

It will ask you to move each joint through its full range, then press Enter. Repeat for
`pluck_arm` with its own port. The `--robot.id` is what ties a calibration file to an arm —
keep the names exactly `fret_arm` and `pluck_arm`; the code will look them up by these.

## 5. Smoke test

With the venv active, this reads joint positions once and disconnects (no motion):

```bash
python -c "from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig as C; r=SO101Follower(C(port='/dev/cu.usbmodemXXXX', id='fret_arm')); r.connect(); print(r.get_observation()); r.disconnect()"
```

Expected: a dict of six `<motor>.pos` values in degrees. Wiggle a joint by hand and run it
again — the number should change.

If `connect()` raises about calibration, step 4 didn't save under that id. If it raises a
serial/permission error, another process (a previous script, the calibration tool) still
has the port open — close it.

## 6. Where this plugs into the code

`robot/so101.py` opens both arms from a small config:

```
arms:
  fret:  {port: /dev/cu.usbmodem..., id: fret_arm}
  pluck: {port: /dev/cu.usbmodem..., id: pluck_arm}
```

Everything else (poses, guards, interpolator) talks to that wrapper, never to the port directly.

## Gotchas

- Two arms = two USB ports on the Mac. If you must use a hub, use a powered one; the serial
  boards are picky about bus power.
- Torque stays on after a script crashes. Power-cycle the arm before moving it by hand to
  record poses, or use `robot.bus.disable_torque()` in the pose-recording script.
- The MacBook camera/mic permission prompt appears the first time the terminal opens them;
  if you're running from an IDE terminal, the permission is granted to the IDE, not to Terminal.
