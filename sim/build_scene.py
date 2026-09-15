"""Compose the MuJoCo scene: two SO-101 arms + a guitar lying face up.

Arm A (fret) sits beside the neck. Arm B (pluck) sits beside the body.
The guitar is static geometry; string *sound* is modelled analytically in
string_model.py - MuJoCo only has to tell us where the fingertip and pick
are touching. Run directly to dump the composed XML for inspection.
"""
from pathlib import Path
import mujoco, numpy as np

HERE = Path(__file__).parent
SO101 = HERE / "so101" / "so101.xml"

# --- guitar geometry (metres) -------------------------------------------------
SCALE_LEN = 0.648          # nut -> bridge, standard 25.5"
STRING_Z = 0.085           # string height above the table
NUT_X = 0.0                # nut at origin, neck runs +X
BRIDGE_X = SCALE_LEN
N_STRINGS = 6
STRING_DY = 0.0105         # spacing between strings
STRING_Y0 = -0.5 * (N_STRINGS - 1) * STRING_DY
LIVE_STRING = 4            # index of the A string (0 = high E ... 5 = low E)
TIP_OFFSET = [0.0, -0.10, 0.015]   # tool tip relative to the gripper body

def fret_x(n: int) -> float:
    """Distance from the nut to fret n (equal temperament)."""
    return SCALE_LEN * (1.0 - 2.0 ** (-n / 12.0))

def string_y(i: int) -> float:
    return STRING_Y0 + i * STRING_DY

def build() -> mujoco.MjSpec:
    w = mujoco.MjSpec()
    w.compiler.autolimits = True
    w.option.timestep = 0.002
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    w.option.impratio = 10
    w.option.ls_iterations = 20
    getattr(w, "visual").global_.offwidth = 1280
    getattr(w, "visual").global_.offheight = 720

    w.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                  builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                  width=300, height=300, rgb1=[.2, .3, .4], rgb2=[.1, .15, .2])
    mat = w.add_material(name="grid", texrepeat=[6, 6], reflectance=.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    w.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                         size=[2, 2, .05], material="grid")
    w.worldbody.add_light(pos=[0.3, 0, 1.2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_SPOT)

    # --- guitar ---------------------------------------------------------------
    g = w.worldbody.add_body(name="guitar", pos=[0, 0, 0])
    NOCOL = dict(contype=0, conaffinity=0)   # visual only; sound is analytic
    # neck slab, from nut to where the body starts
    g.add_geom(name="neck", type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[SCALE_LEN * 0.35, 0, STRING_Z - 0.013],
               size=[SCALE_LEN * 0.35, 0.032, 0.012], rgba=[.35, .22, .12, 1], **NOCOL)
    # body slab under the pickups / bridge
    g.add_geom(name="body", type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[SCALE_LEN * 0.88, 0, STRING_Z - 0.028],
               size=[0.17, 0.16, 0.027], rgba=[.5, .12, .12, 1], **NOCOL)
    # nut + bridge
    for nm, x in (("nut", NUT_X), ("bridge", BRIDGE_X)):
        g.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[x, 0, STRING_Z - 0.003], size=[0.004, 0.038, 0.004],
                   rgba=[.9, .9, .85, 1], **NOCOL)
    # frets 1..12 (thin bars across the neck)
    for n in range(1, 13):
        g.add_geom(name=f"fret_{n}", type=mujoco.mjtGeom.mjGEOM_BOX,
                   pos=[fret_x(n), 0, STRING_Z - 0.0012],
                   size=[0.0011, 0.034, 0.0016], rgba=[.75, .75, .8, 1], **NOCOL)
    # strings: thin capsules along X. Only LIVE_STRING is scored; the rest are
    # "muted with foam" exactly as on the real rig.
    for i in range(N_STRINGS):
        live = (i == LIVE_STRING)
        g.add_geom(name=f"string_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                   fromto=[NUT_X, string_y(i), STRING_Z, BRIDGE_X, string_y(i), STRING_Z],
                   size=[0.0011 if live else 0.0008],
                   rgba=[.95, .85, .4, 1] if live else [.6, .6, .62, 1], **NOCOL)

    # --- arms -----------------------------------------------------------------
    # A frets: beside the neck, centred on fret ~6, reaching sideways.
    a_site = w.worldbody.add_site(name="A_mount", pos=[fret_x(6), -0.26, 0.0],
                                  quat=[0.7071, 0, 0, 0.7071])
    # B plucks: beside the body, over the pickup area.
    b_site = w.worldbody.add_site(name="B_mount", pos=[SCALE_LEN * 0.82, 0.30, 0.0],
                                  quat=[0.7071, 0, 0, -0.7071])
    arm = mujoco.MjSpec.from_file(str(SO101))
    a_site.attach_body(arm.body("base"), "A_", "")
    arm_b = mujoco.MjSpec.from_file(str(SO101))
    b_site.attach_body(arm_b.body("base"), "B_", "")

    # Tool tips: the fingertip that presses the string (arm A) and the pick (arm B).
    # Both hang off the gripper body so IK can drive them straight to a target.
    w.body("A_gripper").add_site(name="A_tip", pos=TIP_OFFSET, size=[0.004],
                                 rgba=[0, 1, 0, 1])
    w.body("B_gripper").add_site(name="B_tip", pos=TIP_OFFSET, size=[0.004],
                                 rgba=[1, 0, 1, 1])
    return w

if __name__ == "__main__":
    spec = build()
    model = spec.compile()
    print(f"compiled: {model.nq} dof, {model.nu} actuators, {model.ngeom} geoms")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print("actuators:", names)
    (HERE / "scene_generated.xml").write_text(spec.to_xml())
    print("wrote sim/scene_generated.xml")
