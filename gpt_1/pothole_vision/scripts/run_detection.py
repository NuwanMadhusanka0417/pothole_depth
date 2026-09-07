#!/usr/bin/env python3
"""Run detection-only on video."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401

from pothole_vision.pipeline.pipeline import PotholePipeline
from pothole_vision.pipeline.stages import PipelineMode
from pothole_vision.utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pothole detection")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(PROJECT_ROOT, str(args.config))
    config = load_config(config_path, PROJECT_ROOT)
    pipeline = PotholePipeline(config, PROJECT_ROOT, PipelineMode.DETECTION, debug=args.debug)
    result = pipeline.run(args.video, start_frame=args.start, duration=args.duration, max_frames=args.max_frames)
    print(result)


if __name__ == "__main__":
    main()
