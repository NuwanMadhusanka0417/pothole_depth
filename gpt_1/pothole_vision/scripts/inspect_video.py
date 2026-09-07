#!/usr/bin/env python3
"""Inspect video metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401

from pothole_vision.utils.logging import setup_logging
from pothole_vision.video.reader import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dashcam video metadata")
    parser.add_argument("--video", required=True, type=Path)
    args = parser.parse_args()
    setup_logging("INFO")

    with VideoReader(args.video) as reader:
        meta = reader.metadata
        info = {
            "path": str(meta.path),
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "duration_seconds": meta.duration_seconds,
            "codec": meta.codec,
        }
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
