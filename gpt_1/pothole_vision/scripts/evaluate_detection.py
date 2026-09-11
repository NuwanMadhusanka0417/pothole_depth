#!/usr/bin/env python3
"""Evaluate detection against ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection results")
    parser.add_argument(
        "--predictions",
        default=None,
        type=Path,
        help="detections.json path, or use --output-dir to scan per-video folders",
    )
    parser.add_argument("--output-dir", default="data/output", type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path, help="CSV with frame,bbox columns")
    args = parser.parse_args()

    gt = pd.read_csv(args.ground_truth)
    out_root = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    if args.predictions is not None:
        pred_path = args.predictions if args.predictions.is_absolute() else PROJECT_ROOT / args.predictions
        with pred_path.open() as f:
            preds = json.load(f)["detections"]
        print(f"Predictions: {len(preds)}, Ground truth rows: {len(gt)}")
    else:
        json_files = list(out_root.glob("*/detections.json"))
        print(f"Found {len(json_files)} per-video detections.json under {out_root}")
        print(f"Ground truth rows: {len(gt)}")
    print("Full mAP evaluation requires COCO-format annotations — extend this script as needed.")


if __name__ == "__main__":
    main()
