"""Audio in, note events out.

Two modes:

* monophonic  - pYIN fundamental tracking segmented by onsets and pitch changes.
                Accurate on single-note melodies.
* polyphonic  - constant-Q peak picking at each onset with harmonic
                suppression. Handles chords and double stops. Coarser.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np

from .config import TranscribeSpec
from .media import load_media
from .types import NoteEvent, Transcription

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


def _librosa():
    import librosa  # imported lazily so `--help` works without the dep
    return librosa


def load_audio(path: str, sr: int) -> Tuple[np.ndarray, int]:
    """Backward-compatible alias for loading local audio/video media."""
    y, sr_out = load_media(path, sr)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32, copy=False), sr_out


def isolate_humming(y: np.ndarray, sr: int, spec: TranscribeSpec) -> np.ndarray:
    """Keep foreground humming and suppress steady room/background noise.

    This is intentionally a conservative single-microphone cleanup rather
    than a claim of speaker identification.  A mono recording cannot prove
    which person produced a sound.  In practice, a voice band-pass, adaptive
    spectral subtraction, and an RMS voice-activity gate remove fan noise,
    keyboard noise, and quieter competing voices before pYIN sees the signal.
    """
    if y.size < 32:
        return np.asarray(y, dtype=np.float32)

    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(y, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x, copy=False)
    x = x - float(np.mean(x))
    nyquist = sr / 2.0
    low = max(20.0, min(float(spec.voice_low_hz), nyquist * 0.45))
    high = max(low + 20.0, min(float(spec.voice_high_hz), nyquist * 0.95))
    try:
        sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
        x = sosfiltfilt(sos, x).astype(np.float32, copy=False)
    except ValueError:
        # Very short or unusual sample rates should still reach the tracker.
        pass

    librosa = _librosa()
    n_fft = 1024 if x.size >= 1024 else 256
    hop = min(256, max(64, n_fft // 4))
    stft = librosa.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft)
    magnitude = np.abs(stft)
    if magnitude.shape[1] > 2:
        frame_energy = np.sqrt(np.mean(magnitude * magnitude, axis=0))
        noise_count = max(1, int(round(frame_energy.size * 0.20)))
        quiet_frames = np.argsort(frame_energy)[:noise_count]
        noise_profile = np.median(magnitude[:, quiet_frames], axis=1, keepdims=True)
        strength = max(0.0, float(spec.noise_reduction_strength))
        # Wiener-style gain avoids the metallic artifacts caused by hard
        # spectral subtraction while still pulling down stationary noise.
        gain = (magnitude * magnitude) / (
            magnitude * magnitude + strength * noise_profile * noise_profile + 1e-8
        )
        cleaned = librosa.istft(stft * gain, hop_length=hop, win_length=n_fft, length=x.size)
        x = np.asarray(cleaned, dtype=np.float32)

    # Soft-gate frames below the adaptive foreground threshold.  The gate is
    # smooth at frame boundaries so pYIN does not interpret clicks as notes.
    frame_rms = librosa.feature.rms(y=x, frame_length=n_fft, hop_length=hop)[0]
    if frame_rms.size:
        noise_floor = float(np.percentile(frame_rms, 20))
        signal_floor = float(np.percentile(frame_rms, 55))
        threshold = max(
            noise_floor * max(1.15, float(spec.voice_gate_strength)),
            signal_floor * 0.32,
            float(np.max(frame_rms)) * 0.012,
        )
        softness = max(threshold - noise_floor, 1e-7)
        frame_gain = np.clip((frame_rms - noise_floor) / softness, 0.0, 1.0)
        frame_gain = np.convolve(frame_gain, np.ones(3) / 3.0, mode="same")
        sample_gain = np.interp(
            np.arange(x.size), np.arange(frame_gain.size) * hop, frame_gain,
            left=float(frame_gain[0]), right=float(frame_gain[-1]),
        )
        x = x * sample_gain.astype(np.float32)

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-6:
        x = x / peak
    return x.astype(np.float32, copy=False)


def record_audio(seconds: float, sr: int) -> np.ndarray:
    """Capture from the default input device. Requires sounddevice."""
    import sounddevice as sd

    frames = int(seconds * sr)
    buf = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return buf.reshape(-1)


class StreamRecorder:
    """Open-ended microphone capture for a start/stop UI, not a fixed --seconds.

    sounddevice's blocking sd.rec() needs the length up front. This runs an
    InputStream with a callback instead, so recording lasts exactly as long
    as the caller wants: call start(), then stop() when the user is done.
    """

    def __init__(self, sr: int = 22050, channels: int = 1):
        self.sr = sr
        self.channels = channels
        self._stream = None
        self._chunks: List[np.ndarray] = []

    @property
    def is_recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            return
        self._chunks = []

        def _callback(indata, frames, time_info, status):
            self._chunks.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sr, channels=self.channels, dtype="float32",
            callback=_callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop capture and return the recorded audio as a 1-D float32 array."""
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._chunks, axis=0).reshape(-1)
        self._chunks = []
        return audio


# ---------------------------------------------------------------- monophonic

def pitch_range(spec: TranscribeSpec) -> Tuple[float, float]:
    """Pitch search bounds in Hz, defaulting to a standard guitar."""
    from .types import midi_to_hz
    fmin = spec.fmin_hz if spec.fmin_hz else midi_to_hz(40)   # E2
    fmax = spec.fmax_hz if spec.fmax_hz else midi_to_hz(88)   # E6
    return float(fmin), float(fmax)


def _median_filter(values: np.ndarray, width: int = 5) -> np.ndarray:
    if width < 3:
        return values
    pad = width // 2
    padded = np.pad(values, pad, mode="edge")
    out = np.empty_like(values)
    for i in range(values.size):
        out[i] = np.median(padded[i:i + width])
    return out


def _merge_segments(
    segments,
    fps: float,
    hard_onsets,
    max_gap_s: float = 0.12,
):
    """Join same-pitch segments split by a tracking dropout.

    A gap only survives as a note boundary when the onset detector saw a real
    re-attack there. Everything else is one sustained note.
    """
    out = []
    for a, b, pitch in segments:
        if out and out[-1][2] == pitch:
            prev_a, prev_b, _ = out[-1]
            gap_ok = (a - prev_b) / fps <= max_gap_s
            clean = not any(f in hard_onsets for f in range(prev_b, a + 1))
            if gap_ok and clean:
                out[-1] = (prev_a, b, pitch)
                continue
        out.append((a, b, pitch))
    return out


def _strip_artifacts(
    segments,
    fps: float,
    hard_onsets,
    rms,
    max_s: float = 0.12,
    strip_continuations: bool = True,
    continuation_rise: float = 1.10,
):
    """Remove pitch excursions that no attack accounts for.

    Two things get removed. A brief dip to a neighbouring semitone inside a
    held note, and the semitone the tracker passes through while one note
    decays into the next. Both share a signature: no attack at the start, a
    pitch within a tone of the note before, and an envelope that is still
    falling. A real note on a plucked string starts with a rise.

    Set `strip_continuations` false when transcribing legato playing, where
    hammer-ons and pull-offs legitimately arrive without a fresh attack.
    """
    changed = True
    segs = list(segments)
    eps = 1e-9

    def envelope_rise(a: int) -> float:
        """Ratio of the envelope just after a boundary to just before it.

        This is a second, more sensitive attack test than the spectral onset
        detector, which backtracks and can miss a soft pluck.
        """
        before = rms[max(0, a - 6):max(1, a - 1)]
        after = rms[max(0, a - 1):a + 4]
        if before.size == 0 or after.size == 0:
            return float("inf")
        return float(after.max()) / (float(np.median(before)) + eps)

    while changed and len(segs) > 1:
        changed = False
        for i, (a, b, pitch) in enumerate(segs):
            if any(f in hard_onsets for f in range(a, min(b, a + 4))):
                continue
            short = (b - a) / fps < max_s
            continuation = False
            if strip_continuations and i > 0:
                prev_pitch = segs[i - 1][2]
                continuation = (
                    abs(pitch - prev_pitch) <= 2
                    and envelope_rise(a) < continuation_rise
                )
            if not (short or continuation):
                continue
            if i > 0:
                pa, pb, pp = segs[i - 1]
                segs[i - 1] = (pa, b, pp)
                segs.pop(i)
            elif len(segs) > 1:
                na, nb, np_ = segs[i + 1]
                segs[i + 1] = (a, nb, np_)
                segs.pop(i)
            changed = True
            break
    return segs


def transcribe_monophonic(y: np.ndarray, sr: int, spec: TranscribeSpec) -> List[NoteEvent]:
    librosa = _librosa()
    fmin, fmax = pitch_range(spec)

    f0, voiced, voiced_prob = librosa.pyin(
        y=y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=spec.frame_length,
        hop_length=spec.hop_length,
        n_thresholds=spec.pyin_thresholds,
    )
    times = librosa.frames_to_time(np.arange(f0.size), sr=sr, hop_length=spec.hop_length)

    ok = np.isfinite(f0) & voiced
    raw = np.full(f0.shape, -1.0)
    raw[ok] = np.round(librosa.hz_to_midi(f0[ok]))
    smooth = _median_filter(raw.copy(), 5)
    smooth[~ok] = -1.0

    rms = librosa.feature.rms(
        y=y, frame_length=spec.frame_length, hop_length=spec.hop_length
    )[0]
    rms_max = float(rms.max()) or 1.0
    # A short window resolves the attack far better than the pitch-analysis
    # window, which smears it by tens of milliseconds.
    rms_fine = librosa.feature.rms(
        y=y, frame_length=512, hop_length=spec.hop_length
    )[0]

    onset_frames = set(
        int(f) for f in librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=spec.hop_length, backtrack=True
        )
    )

    fps = sr / spec.hop_length
    min_split_frames = max(3, int(max(spec.min_note_seconds, 0.09) * fps))

    def rearticulated(i: int) -> bool:
        back = rms[max(0, i - 4):i]
        if back.size == 0:
            return True
        return float(rms[i]) > spec.reattack_ratio * float(np.median(back))

    hard_onsets = {i for i in onset_frames if rearticulated(i)}

    # A pitch change always starts a new note. A repeated pitch only starts a
    # new one when the envelope rises, which is what separates a re-pluck from
    # vibrato or a tracking wobble.
    segments = []
    cur_pitch = -1
    start = 0
    for i in range(smooth.size):
        pitch = int(smooth[i])
        boundary = (pitch != cur_pitch) or (
            i - start >= min_split_frames and i in hard_onsets
        )
        if boundary:
            if cur_pitch > 0:
                segments.append((start, i, cur_pitch))
            cur_pitch = pitch
            start = i
    if cur_pitch > 0:
        segments.append((start, smooth.size, cur_pitch))

    segments = _strip_artifacts(
        segments, fps, hard_onsets, rms,
        spec.artifact_max_seconds, spec.strip_continuations,
        spec.continuation_rise,
    )
    segments = _merge_segments(segments, fps, hard_onsets)

    def refine_onset(a: int) -> int:
        """Snap a segment start to the steepest nearby envelope rise.

        pYIN locks onto a new pitch while the analysis window still straddles
        the attack, which puts segment starts a frame or two early. The energy
        derivative is the sharper landmark.
        """
        lo, hi = max(1, a - 4), min(rms_fine.size, a + 5)
        if hi - lo < 2:
            return a
        deltas = np.diff(rms_fine[lo - 1:hi])
        return int(lo + int(np.argmax(deltas)))

    notes: List[NoteEvent] = []
    for a, b, pitch in segments:
        a = min(refine_onset(a), max(a, b - 2))
        onset = float(times[a])
        end = float(times[min(b, times.size - 1)])
        dur = max(end - onset, 0.0)
        if dur < spec.min_note_seconds:
            continue
        # Confidence is pitch stability: how much of the unsmoothed track
        # agrees with the note. pYIN's own voicing probability is unreliable
        # on plucked strings, so it is only used to reject silence.
        window = raw[a:b]
        voiced_window = window[window > 0]
        if voiced_window.size == 0:
            continue
        # Gate on a one semitone tolerance so absorbed glide frames do not
        # disqualify a note, but report exact agreement as the confidence.
        near = float(np.mean(np.abs(voiced_window - pitch) <= 1))
        stability = float(np.mean(voiced_window == pitch))
        loudness = float(rms[a:b].max()) / rms_max if b > a else 0.0
        if near < spec.min_confidence or loudness < spec.silence_floor:
            continue
        vel = float(np.clip(rms[a:b].max() / rms_max, 0.15, 1.0)) if b > a else 0.6
        notes.append(
            NoteEvent(onset=max(onset, 0.0), duration=dur, midi=int(pitch),
                      velocity=vel, confidence=round(stability, 3))
        )

    for i, n in enumerate(notes):
        n.group = i
    return notes


# ---------------------------------------------------------------- polyphonic

_HARMONIC_INTERVALS = (12, 19, 24, 28, 31, 36)


def transcribe_polyphonic(y: np.ndarray, sr: int, spec: TranscribeSpec) -> List[NoteEvent]:
    librosa = _librosa()
    fmin_hz, fmax_hz = pitch_range(spec)
    n_bins = int(round(12 * np.log2(fmax_hz / fmin_hz))) + 1
    C = np.abs(
        librosa.cqt(y=y, sr=sr, fmin=fmin_hz, n_bins=n_bins,
                    bins_per_octave=12, hop_length=spec.hop_length)
    )
    base_midi = int(round(librosa.hz_to_midi(fmin_hz)))
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=spec.hop_length)

    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=spec.hop_length, backtrack=True, units="time"
    )
    if len(onsets) == 0:
        onsets = np.array([0.0])
    total = float(len(y) / sr)
    bounds = list(onsets) + [total]

    global_peak = float(C.max()) or 1.0

    def frame_of(t: float) -> int:
        return int(np.clip(np.searchsorted(times, t), 0, C.shape[1] - 1))

    notes: List[NoteEvent] = []
    group = 0
    for i, t0 in enumerate(onsets):
        t1 = bounds[i + 1]
        a = frame_of(t0 + 0.02)
        b = max(frame_of(min(t0 + 0.14, t1)), a + 1)
        frame = C[:, a:b].mean(axis=1)
        peak = float(frame.max())
        if peak <= 1e-6 or peak < spec.poly_onset_floor * global_peak:
            continue                      # decay tail, not a real attack

        padded = np.concatenate([[0.0], frame, [0.0]])
        cands = []
        for k in range(n_bins):
            if padded[k + 1] >= padded[k] and padded[k + 1] >= padded[k + 2]:
                if frame[k] >= spec.poly_rel_threshold * peak:
                    cands.append((base_midi + k, float(frame[k])))
        cands.sort(key=lambda c: -c[1])
        cands = cands[: spec.poly_max_notes * 2]

        kept: List[Tuple[int, float]] = []
        for pitch, mag in sorted(cands, key=lambda c: c[0]):
            harmonic_of = any(
                (pitch - lower) in _HARMONIC_INTERVALS and mag < 0.65 * lmag
                for lower, lmag in kept
            )
            if not harmonic_of:
                kept.append((pitch, mag))
        kept.sort(key=lambda c: -c[1])
        kept = sorted(kept[: spec.poly_max_notes], key=lambda c: c[0])

        dur = max(t1 - t0, spec.min_note_seconds)
        for pitch, mag in kept:
            notes.append(
                NoteEvent(onset=float(t0), duration=float(dur), midi=int(pitch),
                          velocity=float(np.clip(mag / peak, 0.2, 1.0)),
                          confidence=float(np.clip(mag / peak, 0.0, 1.0)),
                          group=group)
            )
        group += 1
    return notes


# ---------------------------------------------------------------- grouping

def group_chords(notes: List[NoteEvent], window: float) -> List[NoteEvent]:
    """Re-label groups so notes struck within `window` seconds share a group."""
    if not notes:
        return notes
    ordered = sorted(notes, key=lambda n: (n.onset, n.midi))
    group = 0
    anchor = ordered[0].onset
    for n in ordered:
        if n.onset - anchor > window:
            group += 1
            anchor = n.onset
        n.group = group
    return ordered


def estimate_tempo(y: np.ndarray, sr: int, hop_length: int) -> Tuple[float, List[float]]:
    """Beat-track for a tempo grid.

    Humming and other onset-poor material regularly gives beat_track nothing
    to lock onto, and it returns 0 bpm with no beats. That 0 must not reach
    the quantiser: a fallback tempo with no beats tells it to fall back to a
    fixed-interval grid instead of collapsing every note onto beat zero.
    """
    librosa = _librosa()
    tempo, beats = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length, units="time"
    )
    bpm = float(np.atleast_1d(tempo)[0])
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0
    return bpm, [float(b) for b in np.atleast_1d(beats)]


def transcribe(
    path_or_audio,
    spec: TranscribeSpec,
    source: str = "",
    voice_isolation: bool = False,
    estimate_timing: bool = True,
) -> Transcription:
    """Full front end. Accepts audio samples, a local media file, or a URL."""
    if isinstance(path_or_audio, tuple):
        y, sr = path_or_audio
    else:
        y, sr = load_media(str(path_or_audio), spec.sr)
        source = source or str(path_or_audio)

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if voice_isolation:
        y = isolate_humming(y, int(sr), spec)

    if spec.polyphonic:
        notes = transcribe_polyphonic(y, sr, spec)
    else:
        notes = transcribe_monophonic(y, sr, spec)
    notes = group_chords(notes, spec.chord_window)

    if estimate_timing:
        bpm, beats = estimate_tempo(y, sr, spec.hop_length)
    else:
        bpm, beats = 0.0, []
    return Transcription(
        notes=notes,
        tempo_bpm=bpm,
        beat_times=beats,
        duration=float(len(y) / sr),
        sr=sr,
        source=source,
        polyphonic=spec.polyphonic,
        meta={"n_samples": int(len(y)), "voice_isolated": bool(voice_isolation)},
    )
