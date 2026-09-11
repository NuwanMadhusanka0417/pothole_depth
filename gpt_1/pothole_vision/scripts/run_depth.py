#!/usr/bin/env python3
"""Run depth estimation stage (placeholder CLI)."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401

from cli_videos import add_video_input_args, resolve_videos

from pothole_vision.depth.single_frame_depth import FallbackDepthEstimator
from pothole_vision.utils.config import load_config, resolve_path
from pothole_vision.utils.logging import setup_logging
from pothole_vision.video.reader import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run depth estimation on video frames")
    add_video_input_args(parser)
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    setup_logging("INFO")

    config = load_config(resolve_path(PROJECT_ROOT, str(args.config)), PROJECT_ROOT)
    videos = resolve_videos(args, config)
    estimator = FallbackDepthEstimator()

    for video_path in videos:
        with VideoReader(video_path) as reader:
            frame = reader.read_frame(args.frame)
            if frame is None:
                raise RuntimeError(f"Cannot read frame {args.frame} from {video_path}")
            result = estimator.estimate([frame.image])
            print(f"{video_path.name}: model={result.model_name}, metric={result.is_metric}")


if __name__ == "__main__":
    main()
