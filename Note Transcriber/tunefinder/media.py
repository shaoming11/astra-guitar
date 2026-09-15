"""Media input helpers for local files and online video sources.

The transcription engine works on mono PCM samples.  This module keeps the
container/transport concerns out of the pitch tracker: FFmpeg decodes local
video files (and audio formats it can open), while yt-dlp resolves a YouTube
URL to an audio stream that FFmpeg can decode without saving a permanent copy.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import numpy as np


class MediaInputError(RuntimeError):
    """Raised when a media source cannot be opened or decoded."""


def is_url(source: str | Path) -> bool:
    parsed = urlparse(str(source))
    return parsed.scheme.lower() in {"http", "https"}


def is_youtube_url(source: str | Path) -> bool:
    """Return whether *source* is a YouTube URL supported by yt-dlp."""
    host = urlparse(str(source)).netloc.lower().split(":", 1)[0]
    return host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")


def _read_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 - add a useful media-level error
        raise MediaInputError(f"FFmpeg returned audio that could not be read: {exc}") from exc
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32)
    if not y.size:
        raise MediaInputError("The media source contains no audio.")
    return y, int(sr)


def _decode_with_ffmpeg(
    source: str, sr: int, http_headers: dict[str, str] | None = None
) -> Tuple[np.ndarray, int]:
    header_args: list[str] = []
    if http_headers:
        header_text = "".join(f"{key}: {value}\r\n" for key, value in http_headers.items())
        header_args = ["-headers", header_text]
    command = [
        "ffmpeg", "-nostdin", "-v", "error", *header_args, "-i", source,
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", "pipe:1",
    ]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaInputError(
            "FFmpeg is required for video and URL input. Install it with "
            "Homebrew (`brew install ffmpeg`) or your system package manager."
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise MediaInputError(f"FFmpeg could not decode {source!r}: {detail[-600:]}")
    return _read_wav_bytes(result.stdout)


def _youtube_audio_url(url: str) -> tuple[str, dict[str, str]]:
    try:
        import yt_dlp
    except ImportError as exc:
        raise MediaInputError(
            "YouTube input needs yt-dlp. Install project requirements or run "
            "`pip install yt-dlp`."
        ) from exc

    options = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001 - yt-dlp has many extractor errors
        raise MediaInputError(f"Could not read YouTube URL: {exc}") from exc

    stream_url = info.get("url")
    if not stream_url:
        raise MediaInputError("YouTube did not provide a playable audio stream.")
    headers = {
        str(key): str(value)
        for key, value in (info.get("http_headers") or {}).items()
    }
    return str(stream_url), headers


def load_media(source: str | Path, sr: int) -> Tuple[np.ndarray, int]:
    """Load a local audio/video file or an HTTP(S) source as mono audio.

    YouTube URLs are resolved with yt-dlp.  Other HTTP(S) URLs are passed to
    FFmpeg directly, which also covers many direct MP4/MOV/WebM links.
    """
    source_text = str(source)
    if is_url(source_text):
        if is_youtube_url(source_text):
            decode_source, headers = _youtube_audio_url(source_text)
            return _decode_with_ffmpeg(decode_source, sr, headers)
        return _decode_with_ffmpeg(source_text, sr)

    path = Path(source_text)
    if not path.exists():
        raise MediaInputError(f"Media file not found: {path}")
    video_extensions = {
        ".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".flv", ".wmv", ".mpeg", ".mpg",
        ".ts", ".mts", ".m2ts", ".3gp", ".ogv", ".mxf",
    }
    if path.suffix.lower() in video_extensions:
        return _decode_with_ffmpeg(str(path), sr)

    # Keep the original librosa path for ordinary audio files so existing
    # inputs retain the same decoder behavior and metadata handling.
    import librosa

    try:
        y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        # Some audio containers are only available through FFmpeg/audioread.
        # Falling back here also makes extension-less uploaded video files work.
        try:
            return _decode_with_ffmpeg(str(path), sr)
        except MediaInputError:
            raise MediaInputError(f"Could not open media file {path}: {exc}") from exc
    return np.asarray(y, dtype=np.float32), int(sr_out)
