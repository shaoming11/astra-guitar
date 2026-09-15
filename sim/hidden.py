"""Per-run hidden calibration.

The physics thresholds are randomized per run and written to a sealed file that the
agent does not read. This is what makes a run an actual test: the agent can only learn
the rig's limits from scorer feedback, not from source. Reveal after the run to grade.
"""
import json, random
from pathlib import Path

FIELDS = ("MIN_PRESS_MM", "GOOD_PRESS_MM", "MAX_PRESS_MM", "FRET_WINDOW_MM", "MIN_PLUCK")

def draw(seed: int) -> dict:
    r = random.Random(seed)
    # The viable press band is deliberately NARROW (0.6-1.8 mm wide) and placed anywhere
    # in 1-6 mm, so no fixed guess clears a useful share of seeds - the agent has to search.
    good = round(r.uniform(1.0, 6.0), 2)
    return {"MIN_PRESS_MM": round(good * r.uniform(0.45, 0.85), 2),
            "GOOD_PRESS_MM": good,
            "MAX_PRESS_MM": round(good + r.uniform(0.6, 1.8), 2),
            "FRET_WINDOW_MM": round(r.uniform(3.0, 26.0), 1),
            "MIN_PLUCK": round(r.uniform(0.02, 0.80), 3)}

def seal(seed: int, path: Path) -> dict:
    s = draw(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(s, indent=1))
    return s
