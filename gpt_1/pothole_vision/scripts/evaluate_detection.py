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
    parser.add_argument("--predictions", default="data/output/detections.json", type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path, help="CSV with frame,bbox columns")
    args = parser.parse_args()

    pred_path = args.predictions if args.predictions.is_absolute() else PROJECT_ROOT / args.predictions
    with pred_path.open() as f:
        preds = json.load(f)["detections"]
    gt = pd.read_csv(args.ground_truth)

    print(f"Predictions: {len(preds)}, Ground truth rows: {len(gt)}")
    print("Full mAP evaluation requires COCO-format annotations — extend this script as needed.")


if __name__ == "__main__":
    main()
