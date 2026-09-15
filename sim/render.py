"""Render the scene to a PNG so we can eyeball the layout."""
import sys, mujoco, numpy as np
from pathlib import Path
from build_scene import build
out = Path(sys.argv[1] if len(sys.argv) > 1 else "sim/scene.png")
spec = build(); model = spec.compile(); data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.34, 0.0, 0.10]; cam.distance = 1.35
cam.azimuth = float(sys.argv[2]) if len(sys.argv) > 2 else 125.0
cam.elevation = float(sys.argv[3]) if len(sys.argv) > 3 else -32.0
with mujoco.Renderer(model, 720, 1280) as r:
    r.update_scene(data, cam)
    import PIL.Image; PIL.Image.fromarray(r.render()).save(out)
print("wrote", out)
