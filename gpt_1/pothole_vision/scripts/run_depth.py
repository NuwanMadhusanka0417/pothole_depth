#!/usr/bin/env python3
"""Run depth estimation stage (placeholder CLI)."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401

from pothole_vision.depth.single_frame_depth import FallbackDepthEstimator
from pothole_vision.utils.logging import setup_logging
from pothole_vision.video.reader import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run depth estimation on video frames")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    setup_logging("INFO")

    estimator = FallbackDepthEstimator()
    with VideoReader(args.video) as reader:
        frame = reader.read_frame(args.frame)
        if frame is None:
            raise RuntimeError("Cannot read frame")
        result = estimator.estimate([frame.image])
        print(f"Model: {result.model_name}, metric: {result.is_metric}, scale_conf: {result.scale_confidence}")


if __name__ == "__main__":
    main()
