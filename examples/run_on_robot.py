"""Reference controller: how to execute a robotab document.

This is the consumer side of the format. It runs the command stream against a
clock and dispatches to whatever your hardware layer looks like. Swap
`PrintArms` for your servo, solenoid or motor driver and the rest stands.

    python examples/run_on_robot.py take.json --dry-run
    python examples/run_on_robot.py take.json --speed 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


class PrintArms:
    """Stand in for the hardware layer. Every method is one physical action."""

    def move(self, position: int, travel_s: float) -> None:
        print(f"    fret arm -> position {position} (allow {travel_s*1000:.0f} ms)")

    def press(self, string: int, fret: int, finger: int, barre=None) -> None:
        kind = f"barre {barre}" if barre else f"finger {finger}"
        print(f"    press  string {string} fret {fret:>2} with {kind}")

    def release(self, string: int, fret: int, finger: int) -> None:
        print(f"    lift   string {string} fret {fret:>2}")

    def release_all(self) -> None:
        print("    lift all fingers")

    def pluck(self, string: int, direction: str, velocity: float) -> None:
        print(f"    pluck  string {string} {direction} at {velocity:.2f}")

    def rest(self) -> None:
        print("    pick arm to rest")


def execute(doc: dict, arms=None, speed: float = 1.0, dry_run: bool = False) -> None:
    arms = arms or PrintArms()
    commands = doc["commands"]

    # Pre-take setup runs before the clock starts. The arms are already in
    # position when the first note is due.
    setup = [c for c in commands if c.get("pre_take")]
    for c in setup:
        arms.move(c["position"], c["travel_s"])

    infeasible = [c for c in commands if c.get("feasible") is False]
    if infeasible:
        print(f"warning: {len(infeasible)} hand shift(s) do not fit the time available")
        if not dry_run:
            raise SystemExit("refusing to run an infeasible sequence")

    start = time.perf_counter()
    for c in commands:
        if c.get("pre_take"):
            continue
        due = c["t"] / speed
        if not dry_run:
            wait = due - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
        print(f"{c['t']:7.3f}s  [{c['arm']}]")

        action = c["action"]
        if action == "move":
            arms.move(c["position"], c["travel_s"])
        elif action == "press":
            arms.press(c["string"], c["fret"], c["finger"], c.get("barre"))
        elif action == "release":
            arms.release(c["string"], c["fret"], c["finger"])
        elif action == "release_all":
            arms.release_all()
        elif action in ("pluck", "strum"):
            arms.pluck(c["string"], c["direction"], c["velocity"])
        elif action == "rest":
            arms.rest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("document", help="a robotab/1.0 JSON file")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate, 0.5 runs at half speed for bring up")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the sequence without waiting on the clock")
    args = ap.parse_args()

    doc = json.loads(Path(args.document).read_text())
    if doc.get("format") != "robotab/1.0":
        raise SystemExit(f"unsupported format: {doc.get('format')}")

    inst = doc["instrument"]
    print(f"{inst['type']} in {inst['tuning']}, {len(doc['notes'])} notes, "
          f"{doc['timing']['tempo_bpm']} bpm\n")
    execute(doc, speed=args.speed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
