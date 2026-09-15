"""The sim rig: same shape of API as the real SO-101 rig.

    rig.fret(n, press_mm)   move arm A's fingertip to fret n at a press depth
    rig.release()           lift arm A off the string
    rig.pluck(speed)        sweep arm B's pick across the live string
    rig.observe()           joint state + tip geometry (what the agent sees)

Audio from each pluck is accumulated so the scorer can grade a whole phrase.
"""
import mujoco, numpy as np
from build_scene import build, fret_x, string_y, STRING_Z, LIVE_STRING
from ik import ArmIK
import string_model as sm

PLUCK_Y_SPAN = 0.022
PLUCK_X = 0.52                      # over the pickups

class Rig:
    def __init__(self, seed=0):
        self.spec = build(); self.m = self.spec.compile(); self.d = mujoco.MjData(self.m)
        self.ik_a = ArmIK(self.m, "A_", "A_tip")
        self.ik_b = ArmIK(self.m, "B_", "B_tip")
        self.audio = []                     # (t_start, samples)
        self.events = []                    # one dict per pluck
        self.t = 0.0
        mujoco.mj_forward(self.m, self.d)
        self._home()

    # ---- low level -------------------------------------------------------
    def _settle(self, steps=250):
        for _ in range(steps):
            mujoco.mj_step(self.m, self.d)
        self.t = self.d.time

    def _drive(self, ik, q):
        self.d.ctrl[ik.act] = q

    def _home(self):
        """Park both tips just above the string, out of each other's way."""
        qa, _ = self.ik_a.solve(self.d, [fret_x(5), string_y(LIVE_STRING), STRING_Z + 0.05])
        qb, _ = self.ik_b.solve(self.d, [PLUCK_X, string_y(LIVE_STRING) + PLUCK_Y_SPAN, STRING_Z])
        self.d.qpos[self.ik_a.qadr] = qa; self.d.qpos[self.ik_b.qadr] = qb
        self._drive(self.ik_a, qa); self._drive(self.ik_b, qb)
        mujoco.mj_forward(self.m, self.d); self._settle(100)

    def tip(self, which):
        return self.d.site(f"{which}_tip").xpos.copy()

    # ---- actions ---------------------------------------------------------
    def fret(self, n: int, press_mm: float = 2.0, behind_mm: float = 8.0):
        """Press the live string just behind fret n at the given depth."""
        x = fret_x(n) + behind_mm / 1000.0
        z = STRING_Z - press_mm / 1000.0
        # lift, travel, lower - never drag the tip along the string
        for target in ([x, string_y(LIVE_STRING), STRING_Z + 0.03], [x, string_y(LIVE_STRING), z]):
            q, err = self.ik_a.solve(self.d, target, q0=self.d.qpos[self.ik_a.qadr])
            self._drive(self.ik_a, q); self._settle(120)
        return {"requested_fret": n, **sm.press_state(self.tip("A"))}

    def release(self):
        p = self.tip("A")
        q, _ = self.ik_a.solve(self.d, [p[0], p[1], STRING_Z + 0.04],
                               q0=self.d.qpos[self.ik_a.qadr])
        self._drive(self.ik_a, q); self._settle(100)

    def pluck(self, speed: float = 0.5):
        """Sweep the pick across the live string; record what it sounded like."""
        y0 = string_y(LIVE_STRING) + PLUCK_Y_SPAN
        y1 = string_y(LIVE_STRING) - PLUCK_Y_SPAN
        press = sm.press_state(self.tip("A"))
        start = self.t
        for frac in np.linspace(0, 1, 7):
            q, _ = self.ik_b.solve(self.d, [PLUCK_X, y0 + (y1 - y0) * frac, STRING_Z],
                                   q0=self.d.qpos[self.ik_b.qadr])
            self._drive(self.ik_b, q)
            self._settle(max(6, int(40 * (1.0 - speed) + 8)))
        freq, quality = sm.sound_note(press, speed)
        ev = {"t": round(start, 3), "fret": press["fret"] if press["pressed"] else 0,
              "freq": round(freq, 1), "note": sm.hz_to_name(freq), "quality": quality,
              "press": press}
        self.events.append(ev)
        self.audio.append((start, sm.synth(freq, quality, speed)))
        # reset the pick for the next note
        q, _ = self.ik_b.solve(self.d, [PLUCK_X, y0, STRING_Z], q0=self.d.qpos[self.ik_b.qadr])
        self._drive(self.ik_b, q); self._settle(40)
        return ev

    def observe(self):
        a, b = self.tip("A"), self.tip("B")
        return {"t": round(self.t, 3),
                "A_tip": [round(v, 4) for v in a], "B_tip": [round(v, 4) for v in b],
                "press": sm.press_state(a),
                "A_joints": [round(v, 3) for v in self.d.qpos[self.ik_a.qadr]],
                "B_joints": [round(v, 3) for v in self.d.qpos[self.ik_b.qadr]]}

    def mixdown(self):
        if not self.audio: return np.zeros(1, dtype=np.float32)
        end = max(t for t, _ in self.audio) + 1.3
        out = np.zeros(int(sm.SR * end) + 1, dtype=np.float32)
        for t, y in self.audio:
            i = int(t * sm.SR); out[i:i + y.size] += y[:out.size - i]
        return out / (np.abs(out).max() + 1e-9)
