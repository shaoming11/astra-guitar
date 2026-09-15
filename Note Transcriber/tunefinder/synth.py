"""Synthesis helpers.

`synth_notes` builds test material. `render_tab_audio` plays the generated tab
back with a plucked string model so you can hear whether the transcription and
the fretboard assignment actually match the input.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from .types import midi_to_hz


def write_wav(path: str, y: np.ndarray, sr: int) -> None:
    import soundfile as sf
    peak = float(np.max(np.abs(y))) or 1.0
    sf.write(path, (y / peak * 0.9).astype(np.float32), sr)


def synth_notes(
    events: Iterable[Tuple[int, float, float]],
    sr: int = 22050,
    tail: float = 0.4,
    harmonics: Sequence[float] = (1.0, 0.45, 0.22, 0.11),
) -> np.ndarray:
    """Additive synthesis of (midi, start_s, duration_s) triples."""
    events = list(events)
    if not events:
        return np.zeros(int(sr * tail), dtype=np.float32)
    total = max(s + d for _, s, d in events) + tail
    y = np.zeros(int(sr * total) + 1, dtype=np.float32)

    for midi, start, dur in events:
        f = midi_to_hz(midi)
        n = int(sr * dur)
        if n <= 0:
            continue
        t = np.arange(n) / sr
        wave = np.zeros(n, dtype=np.float32)
        for k, amp in enumerate(harmonics, start=1):
            wave += amp * np.sin(2 * np.pi * f * k * t)
        attack = max(1, int(0.008 * sr))
        release = max(1, int(min(0.12, dur * 0.4) * sr))
        env = np.ones(n, dtype=np.float32)
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] *= np.linspace(1, 0, release)
        i = int(start * sr)
        y[i:i + n] += wave * env / len(harmonics)
    return y


def karplus_strong(freq: float, dur: float, sr: int, damping: float = 0.996) -> np.ndarray:
    n_delay = max(2, int(sr / max(freq, 1e-6)))
    n = int(sr * dur)
    rng = np.random.default_rng(int(freq * 1000) % 2**31)
    buf = rng.uniform(-1, 1, n_delay).astype(np.float32)
    out = np.empty(n, dtype=np.float32)
    idx = 0
    for i in range(n):
        v = buf[idx]
        out[i] = v
        buf[idx] = damping * 0.5 * (v + buf[(idx + 1) % n_delay])
        idx = (idx + 1) % n_delay
    return out


def render_tab_audio(note_records: Sequence[dict], sr: int = 22050, tail: float = 0.6) -> np.ndarray:
    """Render the arranged note list back to audio for A/B checking."""
    if not note_records:
        return np.zeros(int(sr * tail), dtype=np.float32)
    total = max(r["onset_s"] + r["duration_s"] for r in note_records) + tail
    y = np.zeros(int(sr * total) + 1, dtype=np.float32)
    for r in note_records:
        dur = min(r["duration_s"] + 0.35, 3.0)
        wave = karplus_strong(r["freq_hz"], dur, sr) * float(r.get("velocity", 0.8))
        i = int(r["onset_s"] * sr)
        y[i:i + wave.size] += wave[: max(0, y.size - i)]
    return y
