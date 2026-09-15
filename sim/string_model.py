"""Turn MuJoCo tip positions into notes, buzz, and audio.

MuJoCo cannot simulate a vibrating string, and it does not need to: what the
agent must learn is *where* and *how hard* to press and *when* to pluck. So the
sim reports geometry, and this module maps geometry -> sound with the same
failure modes as a real guitar:

  press too shallow            -> buzz / dead note
  press too far from the fret  -> buzz (the classic "press behind the fret" rule)
  no press                     -> open string
  pluck without contact        -> silence
"""
import numpy as np
from build_scene import fret_x, string_y, STRING_Z, SCALE_LEN, LIVE_STRING

SR = 44100
OPEN_HZ = 110.0          # A string, standard tuning
# Defaults; overwritten per run by secrets.draw() so the agent cannot read them off.
MIN_PRESS_MM = 1.2       # below this the string is not properly stopped
GOOD_PRESS_MM = 2.0
MAX_PRESS_MM = 6.0       # beyond this the string bends sharp
FRET_WINDOW = 0.020      # must land within this distance behind the fret wire
MIN_PLUCK = 0.02

def apply_secret(s):
    global MIN_PRESS_MM, GOOD_PRESS_MM, MAX_PRESS_MM, FRET_WINDOW, MIN_PLUCK
    MIN_PRESS_MM = s["MIN_PRESS_MM"]; GOOD_PRESS_MM = s["GOOD_PRESS_MM"]
    MAX_PRESS_MM = s["MAX_PRESS_MM"]; FRET_WINDOW = s["FRET_WINDOW_MM"] / 1000.0
    MIN_PLUCK = s["MIN_PLUCK"]
NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def hz_to_name(hz):
    if hz <= 0: return "-"
    n = round(12 * np.log2(hz / 440.0)) + 69
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"

def nearest_fret(x):
    """Which fret is a tip at x stopping? Returns (fret, mm_behind_wire)."""
    best, bestd = 0, 1e9
    for n in range(1, 13):
        d = x - fret_x(n)            # positive = toward the bridge, the correct side
        if 0 <= d < bestd:
            best, bestd = n, d
    return best, bestd * 1000.0

def press_state(tip_xyz):
    """Analyse arm A's fingertip against the live string."""
    x, y, z = tip_xyz
    off_string_mm = abs(y - string_y(LIVE_STRING)) * 1000.0
    depth_mm = (STRING_Z - z) * 1000.0
    if off_string_mm > 6.0 or depth_mm < 0 or not (0 < x < SCALE_LEN):
        return {"pressed": False, "fret": 0, "depth_mm": round(depth_mm, 2),
                "off_string_mm": round(off_string_mm, 1), "behind_fret_mm": None}
    fret, behind = nearest_fret(x)
    return {"pressed": depth_mm >= MIN_PRESS_MM, "fret": fret,
            "depth_mm": round(depth_mm, 2), "off_string_mm": round(off_string_mm, 1),
            "behind_fret_mm": round(behind, 1)}

def sound_note(press, pluck_speed):
    """(freq, quality) for a pluck given the current press state."""
    fret = press["fret"] if press["pressed"] else 0
    freq = OPEN_HZ * 2 ** (fret / 12.0)
    d, behind = press["depth_mm"], press["behind_fret_mm"]
    q = "clean"
    if pluck_speed < MIN_PLUCK:
        return freq, "silent"
    if press["pressed"]:
        if d < GOOD_PRESS_MM:                 q = "buzz"       # not stopping cleanly
        elif behind is not None and behind > FRET_WINDOW * 1000: q = "buzz"
        elif d > MAX_PRESS_MM:                q = "sharp"
    elif press["depth_mm"] > 0.2:             q = "muted"      # touching but not stopping
    return freq, q

def synth(freq, quality, speed, dur=1.2):
    """Plucked-string audio: decaying harmonic stack, with buzz/mute colouring."""
    t = np.arange(int(SR * dur)) / SR
    if quality == "silent":
        return np.zeros_like(t)
    amp = float(np.clip(speed * 2.0, 0.05, 1.0))
    if quality == "muted":
        amp *= 0.12
    if quality == "sharp":
        freq *= 1.03
    decay = np.exp(-t * (14.0 if quality in ("buzz", "muted") else 3.0))
    y = sum((1.0 / h) * np.sin(2 * np.pi * freq * h * t) for h in (1, 2, 3, 4, 5))
    y *= decay
    if quality == "buzz":                      # rattle against the fret wire
        y += 0.5 * decay * np.sin(2 * np.pi * freq * 13 * t) * np.random.default_rng(0).uniform(.5, 1, t.size)
    y = amp * y / (np.abs(y).max() + 1e-9)
    return y.astype(np.float32)
