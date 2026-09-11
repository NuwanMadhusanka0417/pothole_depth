"""FFprobe/ffmpeg helpers when OpenCV VideoCapture cannot read a file."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None and shutil.which("ffmpeg") is not None


def file_container_hint(path: Path) -> str:
    """Quick on-disk checks useful when decoding fails."""
    try:
        size = path.stat().st_size
    except OSError as e:
        return f"stat failed: {e}"
    if size == 0:
        return "file size is 0 bytes (empty or incomplete copy)"
    with path.open("rb") as f:
        head = f.read(64)
    has_ftyp = b"ftyp" in head[:32]
    return f"size={size} bytes, ftyp_in_header={has_ftyp}"


def _parse_fps(rate: str) -> float:
    if not rate or rate == "0/0":
        return 30.0
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        try:
            return float(rate)
        except ValueError:
            return 30.0


def ffprobe_metadata(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Read video metadata via ffprobe.

    Returns (metadata or None, error/diagnostic message).
    """
    if not ffprobe_available():
        return None, "ffprobe/ffmpeg not on PATH (try: module load ffmpeg)"

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,r_frame_rate,avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        return None, err or f"ffprobe exit code {result.returncode}"

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None, "ffprobe returned invalid JSON"

    streams = data.get("streams") or []
    if not streams:
        return None, "ffprobe found no video stream"

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        return None, "ffprobe reported invalid width/height"

    fps = _parse_fps(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "30/1")
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    nb = stream.get("nb_frames")
    if nb is not None and str(nb).isdigit():
        frame_count = int(nb)
    elif duration > 0 and fps > 0:
        frame_count = int(round(duration * fps))
    else:
        frame_count = 0

    codec = str(stream.get("codec_name") or "")

    duration_seconds = duration if duration > 0 else (frame_count / fps if fps > 0 else 0.0)
    return {
        "path": path,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,
        "codec": codec,
    }, ""
