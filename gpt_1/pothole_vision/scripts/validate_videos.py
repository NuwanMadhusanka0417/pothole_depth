#!/usr/bin/env python3
"""Check that videos in data/input are readable (OpenCV or ffprobe)."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401

from cli_videos import resolve_videos

from pothole_vision.utils.config import load_config, resolve_path
from pothole_vision.video.ffprobe_io import ffprobe_metadata, file_container_hint, ffprobe_available
from pothole_vision.video.reader import open_video_capture


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate video files without full pipeline")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    args = parser.parse_args()

    config = load_config(resolve_path(PROJECT_ROOT, str(args.config)), PROJECT_ROOT)
    class NS:
        video = None
        input_dir = None
    videos = resolve_videos(NS(), config)

    print(f"ffprobe on PATH: {ffprobe_available()}")
    for path in videos:
        print(f"\n{path.name}")
        print(f"  {file_container_hint(path)}")
        try:
            cap = open_video_capture(path)
            cap.release()
            print("  opencv: OK")
        except RuntimeError:
            print("  opencv: FAIL")
        meta, err = ffprobe_metadata(path)
        if meta:
            print(f"  ffprobe: OK {meta['width']}x{meta['height']} @ {meta['fps']:.2f} fps")
        else:
            print(f"  ffprobe: FAIL — {err}")


if __name__ == "__main__":
    main()
