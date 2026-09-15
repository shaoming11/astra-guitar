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
    def __init__(self, name=None):
        self.dir = RUNS / (name or time.strftime("sim-%Y%m%d-%H%M%S"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.turn = 0

    def attempt(self, reason, **params):
        self.turn += 1
        g, rig = run(**params, save_wav=str(self.dir / f"attempt_{self.turn}.wav"),
                     verbose=False)
        rec = {"turn": self.turn, "reason": reason, "params": params,
               "score": g["score"], "clean": g["clean"], "total": g["total"],
               "summary": summary(g), "notes": g["notes"]}
        with (self.dir / "decisions.jsonl").open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        print(f"[turn {self.turn}] {reason}")
        print(f"          params {params}")
        print(f"       -> {summary(g)}\n")
        return g
