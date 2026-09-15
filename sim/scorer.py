"""Grade a played phrase against the score. Compact JSON out for the agent."""
import numpy as np, string_model as sm

def grade(events, plan):
    notes, clean = [], 0
    for i, want in enumerate(plan):
        got = events[i] if i < len(events) else None
        want_hz = sm.OPEN_HZ * 2 ** (want["fret"] / 12.0)
        if got is None:
            notes.append({"i": i, "want": sm.hz_to_name(want_hz), "heard": None,
                          "quality": "missing"}); continue
        cents = 1200 * np.log2(got["freq"] / want_hz) if got["freq"] > 0 else -9999
        ok = got["quality"] == "clean" and abs(cents) < 25
        clean += ok
        n = {"i": i, "want": sm.hz_to_name(want_hz), "heard": got["note"],
             "quality": got["quality"], "fret_got": got["fret"],
             "fret_want": want["fret"], "depth_mm": got["press"]["depth_mm"]}
        if not ok:
            n["why"] = (f"wrong fret ({got['fret']} vs {want['fret']})" if got["fret"] != want["fret"]
                        else {"buzz": "press too shallow or too far behind the fret",
                              "muted": "finger touching but not stopping the string",
                              "sharp": "pressing too hard, string bent sharp",
                              "silent": "pluck too weak"}.get(got["quality"], "off pitch"))
        notes.append(n)
    return {"clean": clean, "total": len(plan),
            "score": round(clean / max(1, len(plan)), 2), "notes": notes}

def summary(g):
    bad = [n for n in g["notes"] if n.get("why")]
    s = f"{g['clean']}/{g['total']} clean"
    if bad:
        s += " | " + "; ".join(f"note{n['i']}({n['want']}): {n['why']}" for n in bad[:4])
    return s
