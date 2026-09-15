"""The Astra-style loop: propose params, play, observe, correct.

Each turn the agent sees ONLY the compact scorer JSON (never joint angles), and
returns the next parameter set plus a one-line reason. Mirrors the paper's
accept/correct framing: the rig proposes the motion, the agent edits parameters.
"""
import json, time
from pathlib import Path
import numpy as np
from play import run
from scorer import summary

RUNS = Path(__file__).resolve().parents[1] / "runs"

class Session:
    """A blind session: physics thresholds are drawn from `seed` and sealed to disk.
    The agent sees only scorer feedback."""
    def __init__(self, name=None, seed=None, budget=12):
        self.dir = RUNS / (name or time.strftime("sim-%Y%m%d-%H%M%S"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.turn = 0; self.budget = budget; self.secret = None
        if seed is not None:
            import hidden as sealed
            self.secret = sealed.seal(seed, self.dir / "SEALED_do_not_read.json")

    def attempt(self, reason, **params):
        self.turn += 1
        if self.turn > self.budget:
            raise RuntimeError(f"turn budget of {self.budget} exhausted")
        g, rig = run(**params, save_wav=str(self.dir / f"attempt_{self.turn}.wav"),
                     verbose=False, secret=self.secret)
        rec = {"turn": self.turn, "reason": reason, "params": params,
               "score": g["score"], "clean": g["clean"], "total": g["total"],
               "summary": summary(g), "notes": g["notes"]}
        with (self.dir / "decisions.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"[turn {self.turn}] {reason}")
        print(f"          params {params}")
        print(f"       -> {summary(g)}\n")
        return g
