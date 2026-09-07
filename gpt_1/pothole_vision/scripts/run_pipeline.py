#!/usr/bin/env python3
"""Main pipeline entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401

from pothole_vision.pipeline.pipeline import PotholePipeline
from pothole_vision.pipeline.stages import PipelineMode
from pothole_vision.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pothole Vision pipeline")
    parser.add_argument("--video", required=True, type=Path, help="Input MP4 video")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--mode", default="detection",
                        choices=["detection", "tracking", "geometry", "full"])
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute cached results")
    args = parser.parse_args()

    mode_map = {
        "detection": PipelineMode.DETECTION,
        "tracking": PipelineMode.TRACKING,
        "geometry": PipelineMode.GEOMETRY,
        "full": PipelineMode.FULL,
    }

    config = load_config(resolve_path(PROJECT_ROOT, str(args.config)), PROJECT_ROOT)
    if args.debug:
        config.pipeline.debug = True

    pipeline = PotholePipeline(
        config, PROJECT_ROOT,
        mode=mode_map[args.mode],
        debug=args.debug,
        force=args.force,
    )
    result = pipeline.run(
        args.video,
        start_frame=args.start,
        duration=args.duration,
        max_frames=args.max_frames,
    )
    print("\n=== Pipeline Complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
