"""Damped least-squares IK for one arm's tool tip.

The real rig hand-records poses because it has no IK. In sim we can solve, which
is what lets `fret(n)` work for any fret without a lookup table.
"""
import mujoco, numpy as np

class ArmIK:
    def __init__(self, model, prefix: str, tip_site: str):
        self.m, self.prefix = model, prefix
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_site)
        self.jnt, self.qadr, self.act = [], [], []
        for j in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if nm and nm.startswith(prefix):
                self.jnt.append(j); self.qadr.append(model.jnt_qposadr[j])
        for a in range(model.nu):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if nm and nm.startswith(prefix):
                self.act.append(a)
        self.qadr = np.array(self.qadr)
        self.lo = model.jnt_range[self.jnt, 0].copy()
        self.hi = model.jnt_range[self.jnt, 1].copy()

    def solve(self, data, target_xyz, q0=None, iters=300, lam=0.15, step=0.6):
        """Return joint angles putting the tip at target_xyz (best effort)."""
        d = mujoco.MjData(self.m)
        d.qpos[:] = data.qpos
        if q0 is not None:
            d.qpos[self.qadr] = q0
        jacp = np.zeros((3, self.m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)
            err = np.asarray(target_xyz) - d.site_xpos[self.site]
            if np.linalg.norm(err) < 2e-4:
                break
            mujoco.mj_jacSite(self.m, d, jacp, None, self.site)
            J = jacp[:, self.qadr]
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), err)
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + step * dq, self.lo, self.hi)
        mujoco.mj_kinematics(self.m, d)
        return d.qpos[self.qadr].copy(), float(np.linalg.norm(
            np.asarray(target_xyz) - d.site_xpos[self.site]))
