#!/usr/bin/env python3
"""Main pipeline entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pothole_vision.pipeline.pipeline import PotholePipeline
from pothole_vision.pipeline.stages import PipelineMode
from pothole_vision.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monocular dashcam pothole detection and measurement pipeline"
    )
    parser.add_argument("--video", required=True, type=Path, help="Input MP4 video")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument(
        "--mode",
        choices=["detection", "tracking", "geometry", "full"],
        default="detection",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force recompute cache")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.force:
        config.setdefault("pipeline", {})["force_recompute"] = True
    if args.debug:
        config.setdefault("pipeline", {})["debug"] = True

    mode = PipelineMode(args.mode)
    pipeline = PotholePipeline(config, project_root=ROOT)
    result = pipeline.run(
        args.video,
        mode=mode,
        start_frame=args.start,
        max_frames=args.max_frames,
        duration_seconds=args.duration,
        debug=args.debug,
    )
    print(f"\nDone. Output: {result['output_dir']}")
    print(f"Detections: {result['num_detections']}, Tracks: {result['num_tracks']}")


if __name__ == "__main__":
    main()
