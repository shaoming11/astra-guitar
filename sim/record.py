"""Play the riff and record frames -> mp4/gif, with the audio muxed if ffmpeg exists."""
import subprocess, shutil, sys
from pathlib import Path
import numpy as np, mujoco, PIL.Image
from rig import Rig
from song import schedule
from scorer import grade, summary
import string_model as sm

def record(out_dir, press_mm=2.6, behind_mm=7.0, pluck=0.6, fps=30):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rig, plan = Rig(), schedule()
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.30, 0.0, 0.09]; cam.distance = 0.95
    cam.azimuth, cam.elevation = 118.0, -26.0
    frames = []
    with mujoco.Renderer(rig.m, 720, 1280) as r:
        def grab():
            r.update_scene(rig.d, cam); frames.append(r.render().copy())
        orig = rig._settle
        def settle(steps=250):                 # grab a frame every ~1/fps of sim time
            every = max(1, int((1/fps) / rig.m.opt.timestep))
            for i in range(steps):
                mujoco.mj_step(rig.m, rig.d)
                if i % every == 0: grab()
            rig.t = rig.d.time
        rig._settle = settle
        for step in plan:
            rig.fret(step["fret"], press_mm=press_mm, behind_mm=behind_mm)
            rig.pluck(pluck)
        rig._settle = orig
    g = grade(rig.events, plan); print(summary(g), f"| {len(frames)} frames")
    from scipy.io import wavfile
    wav = out / "audio.wav"
    wavfile.write(wav, sm.SR, (rig.mixdown() * 32767).astype(np.int16))
    vid = out / "play.mp4"
    if shutil.which("ffmpeg"):
        pr = subprocess.Popen(
            ["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
             "-s","1280x720","-r",str(fps),"-i","-","-i",str(wav),
             "-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-shortest",str(vid)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for f in frames: pr.stdin.write(np.ascontiguousarray(f).tobytes())
            pr.stdin.close()
        except BrokenPipeError:
            pass
        pr.wait()
        err = pr.stderr.read().decode()[:600]
        if pr.returncode: print("ffmpeg failed:", err)
        print("wrote", vid)
    else:
        gif = out / "play.gif"
        imgs = [PIL.Image.fromarray(f).resize((640,360)) for f in frames[::3]]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(3000/fps), loop=0)
        print("wrote", gif, "(no ffmpeg, made a gif)")
    return g

if __name__ == "__main__":
    record(sys.argv[1] if len(sys.argv)>1 else "../runs/sim-fable-seven-nation-army/video")
