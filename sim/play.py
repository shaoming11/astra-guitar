"""Play the riff once with a given parameter set; return the graded result."""
import json, sys, numpy as np
from rig import Rig
from song import schedule
from scorer import grade, summary
import string_model as sm

def run(press_mm=2.0, behind_mm=8.0, pluck=0.6, save_wav=None, verbose=True, secret=None):
    if secret: sm.apply_secret(secret)
    rig, plan = Rig(), schedule()
    for step in plan:
        rig.fret(step["fret"], press_mm=press_mm, behind_mm=behind_mm)
        rig.pluck(pluck)
    g = grade(rig.events, plan)
    g["params"] = {"press_mm": press_mm, "behind_mm": behind_mm, "pluck": pluck}
    if save_wav:
        from scipy.io import wavfile
        wavfile.write(save_wav, sm.SR, (rig.mixdown() * 32767).astype(np.int16))
    if verbose:
        print(summary(g))
    return g, rig

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
    g, _ = run(press_mm=float(kw.get("press_mm", 2.0)),
               behind_mm=float(kw.get("behind_mm", 8.0)),
               pluck=float(kw.get("pluck", 0.6)),
               save_wav=kw.get("wav"))
    print(json.dumps(g["notes"], indent=1))
