"""Seven Nation Army, main riff, on the A string."""
RIFF = [  # (fret, beats)
    (7, 1.0), (7, 0.5), (10, 0.5), (7, 0.5), (5, 0.5), (3, 1.0), (2, 1.0),
]
BPM = 96
def schedule(bpm=BPM):
    t, out = 0.0, []
    for fret, beats in RIFF:
        out.append({"fret": fret, "at_ms": round(t * 1000)})
        t += beats * 60.0 / bpm
    return out
