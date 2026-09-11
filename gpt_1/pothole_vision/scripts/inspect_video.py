#!/usr/bin/env python3
"""Inspect video metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401

from cli_videos import add_video_input_args, resolve_videos

from pothole_vision.utils.config import load_config, resolve_path
from pothole_vision.utils.logging import setup_logging
from pothole_vision.video.reader import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dashcam video metadata")
    add_video_input_args(parser)
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    args = parser.parse_args()
    setup_logging("INFO")

    config = load_config(resolve_path(PROJECT_ROOT, str(args.config)), PROJECT_ROOT)
    videos = resolve_videos(args, config)

    all_info = []
    for video_path in videos:
        with VideoReader(video_path) as reader:
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
            all_info.append(info)
            if len(videos) == 1:
                print(json.dumps(info, indent=2))
            else:
                print(json.dumps(info, indent=2))
                print()

    if len(videos) > 1:
        print(f"--- {len(videos)} videos ---")


if __name__ == "__main__":
    main()
