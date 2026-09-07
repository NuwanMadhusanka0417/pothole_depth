#!/usr/bin/env python3
"""Evaluate depth measurements against ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate depth measurements")
    parser.add_argument("--events-dir", default="data/events", type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    args = parser.parse_args()

    gt = pd.read_csv(args.ground_truth)
    events_dir = args.events_dir if args.events_dir.is_absolute() else PROJECT_ROOT / args.events_dir

    errors = []
    for _, row in gt.iterrows():
        pid = row.get("pothole_id", "")
        meas_path = events_dir / pid / "measurement.json"
        if not meas_path.exists():
            continue
        with meas_path.open() as f:
            meas = json.load(f)
        if not meas.get("metric_depth_available"):
            continue
        pred = meas.get("maximum_depth_cm")
        true = row.get("true_depth_cm")
        if pred is not None and true is not None:
            errors.append(abs(pred - true))

    if not errors:
        print("No comparable measurements found.")
        return

    errors = np.array(errors)
    print(f"MAE:  {np.mean(errors):.2f} cm")
    print(f"RMSE: {np.sqrt(np.mean(errors**2)):.2f} cm")
    for tol in [0.5, 1.0, 2.0]:
        pct = 100 * np.mean(errors <= tol)
        print(f"Within ±{tol} cm: {pct:.1f}%")


if __name__ == "__main__":
    main()
