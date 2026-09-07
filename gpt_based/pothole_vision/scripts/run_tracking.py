#!/usr/bin/env python3
"""Run tracking mode."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = PotholePipeline(config, project_root=ROOT)
    pipeline.run(args.video, mode=PipelineMode.TRACKING, start_frame=args.start, duration_seconds=args.duration)


if __name__ == "__main__":
    main()
