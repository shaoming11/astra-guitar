"""Assign a new ID to a single Feetech servo.

ONE servo on the bus at a time. Two SO-101s both ship with IDs 1-6, so arm B must be
renumbered to 7-12 before the arms can share a bus - duplicate IDs collide and you only
ever see 6 motors.

Usage:
    python scripts/set_motor_id.py 7      # renumber the lone connected servo to ID 7
    python scripts/set_motor_id.py --scan # just report what's on the bus right now
"""
import argparse, sys
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

DEFAULT_PORT = "/dev/cu.usbmodem5A7C1220421"

ap = argparse.ArgumentParser()
ap.add_argument("target_id", nargs="?", type=int, help="new ID to assign (1-253)")
ap.add_argument("--port", default=DEFAULT_PORT)
ap.add_argument("--scan", action="store_true", help="list motors on the bus and exit")
args = ap.parse_args()

bus = FeetechMotorsBus(port=args.port, motors={"m": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)})
bus.connect(handshake=False)
found = bus.broadcast_ping() or {}
print(f"on the bus: {sorted(found)}")

if args.scan:
    bus.disconnect(); sys.exit(0)
if args.target_id is None:
    print("give a target id, or use --scan"); bus.disconnect(); sys.exit(1)
if len(found) != 1:
    print(f"ERROR: need exactly ONE servo connected, found {len(found)}.")
    print("Unplug the daisy chain and connect a single servo to the board.")
    bus.disconnect(); sys.exit(1)

current = next(iter(found))
if current == args.target_id:
    print(f"already ID {current}, nothing to do"); bus.disconnect(); sys.exit(0)

bus.disconnect()
bus = FeetechMotorsBus(port=args.port,
                       motors={"m": Motor(args.target_id, "sts3215", MotorNormMode.RANGE_M100_100)})
bus.connect(handshake=False)
print(f"{current} -> {args.target_id} ...")
bus.setup_motor("m", initial_id=current)
after = bus.broadcast_ping() or {}
print(f"done. bus now: {sorted(after)}")
bus.disconnect()
