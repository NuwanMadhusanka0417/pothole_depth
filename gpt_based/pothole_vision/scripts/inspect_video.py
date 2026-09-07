#!/usr/bin/env python3
"""Inspect video metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pothole_vision.video.reader import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dashcam video metadata")
    parser.add_argument("--video", required=True, type=Path)
    args = parser.parse_args()

    with VideoReader(args.video) as reader:
        m = reader.metadata
        print(f"Path:      {m.path}")
        print(f"Size:      {m.width} x {m.height}")
        print(f"FPS:       {m.fps:.3f}")
        print(f"Frames:    {m.frame_count}")
        print(f"Duration:  {m.duration_seconds:.2f} s")
        print(f"FOURCC:    {m.fourcc}")


if __name__ == "__main__":
    main()
