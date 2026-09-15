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

---

# Two arms on one bus — renumbering arm B to IDs 7–12

**Symptom:** both arms powered (all servo LEDs red), chained together, but `broadcast_ping`
reports only 6 motors. Plugging in either arm alone works.

**Cause:** every SO-101 ships with its six servos at IDs **1–6**. On a shared half-duplex bus
two servos answer to the same address and garble each other's replies. It is an addressing
collision, not a cable, power, or driver problem. There is no config that makes 12 appear —
arm B's servos must be given new IDs first.

**Plan:** arm A keeps 1–6. Arm B becomes 7–12, in the same joint order:

| joint          | arm A | arm B |
|----------------|-------|-------|
| shoulder_pan   | 1     | 7     |
| shoulder_lift  | 2     | 8     |
| elbow_flex     | 3     | 9     |
| wrist_flex     | 4     | 10    |
| wrist_roll     | 5     | 11    |
| gripper        | 6     | 12    |

New ID = old ID + 6, so you don't have to track which servo you're holding — the script
prints the ID it found.

## Procedure

1. **Unplug arm A entirely** from the bus board. Only arm B's servos get touched here.
2. **Break arm B's daisy chain.** Connect exactly **one** arm-B servo to the board with a
   single 3-pin cable. Everything else off the bus.
3. Check what's there:
   ```bash
   .venv/bin/python scripts/set_motor_id.py --scan
   ```
   Expect exactly one ID. If it prints more than one, another servo is still chained —
   the script refuses to run in that case, by design.
4. Assign `found + 6`:
   ```bash
   .venv/bin/python scripts/set_motor_id.py 7
   ```
   (`8` for the servo that reported 2, `9` for 3, and so on.)
5. Unplug that servo, connect the next arm-B servo alone, repeat until all six are 7–12.
6. Rebuild arm B's daisy chain, then chain arm B onto arm A's last servo with a long
   3-pin cable. Reconnect arm A to the board.
7. Verify all twelve:
   ```bash
   .venv/bin/python scripts/set_motor_id.py --scan
   ```
   Expect `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]`.

## Notes

- The ID lives in the servo's EEPROM, so it survives power cycles. Do this once.
- `setup_motor` also writes the bus default baud rate (1 Mbps) — if a servo was at a different
  rate, this fixes it at the same time.
- `SO101Follower` hardcodes IDs 1–6, so **arm B is driven through a raw `FeetechMotorsBus`**
  with a 7–12 motor map (same pattern as `scripts/grip_pick.py`). Calibrate arm B by recording
  min/max per joint on the raw bus, not with `lerobot-calibrate`.
- Label the arms physically (tape) once renumbered. A 7–12 arm plugged in alone will look
  "broken" to any tool expecting 1–6.
- **Alternative:** a second Waveshare Bus Servo Adapter (~$8) gives arm B its own port and
  skips all of this — both arms keep IDs 1–6 and both work with stock `SO101Follower`. Worth
  ordering as a backup regardless, since one PSU driving 12 servos can brown out when the
  fretting arm presses.

---

# Diagnosing a silent bus: is it the board or the wiring?

`scripts/diagnose_bus.py` sweeps raw serial across every plausible baud rate and looks at
whether the returned bytes **change**. That is the discriminator, not whether bytes come back:

| observation | meaning |
|---|---|
| a reply starting `FF FF` | the bus works — it's a baud / ID / model mismatch |
| bytes **differ** per baud | real data on the line, wrong rate |
| bytes **identical** across a wide baud range | no data at all; the UART is framing a stuck DC level → the adapter's transceiver is dead |
| nothing at all | line idle → check servo power and the board→first-servo cable |

```bash
.venv/bin/python scripts/diagnose_bus.py /dev/cu.usbmodemXXXX
```

**2026-09-15:** arm A's board returned `000080bf03` identically for every baud from 9600 to
256000, and `8080c000` identically from 460800 to 1.5M. Real serial data cannot be invariant
across a 26× sampling-rate change, so there was no data on the line — dead transceiver,
despite the board enumerating on USB and all servo LEDs being lit. Replacement ordered.
Lesson: LEDs prove VCC only; the data wire is an independent conductor and fails on its own.
