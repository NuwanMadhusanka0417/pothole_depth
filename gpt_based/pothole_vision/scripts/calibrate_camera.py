#!/usr/bin/env python3
"""Camera checkerboard calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pothole_vision.geometry.intrinsics import CameraIntrinsics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True, help="Folder of checkerboard images")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--square-size", type=float, default=0.025)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "calibration" / "camera.yaml")
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.cols, 0 : args.rows].T.reshape(-1, 2) * args.square_size

    objpoints, imgpoints = [], []
    img_shape = None
    for img_path in sorted(args.images.glob("*.*")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
            img_shape = gray.shape[::-1]

    if not objpoints:
        print("No checkerboard corners found.")
        return

    ret, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    intrinsics = CameraIntrinsics(
        width=img_shape[0], height=img_shape[1],
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        k1=float(dist[0, 0]), k2=float(dist[0, 1]),
        p1=float(dist[0, 2]), p2=float(dist[0, 3]),
        k3=float(dist[0, 4]) if dist.shape[1] > 4 else 0.0,
        calibrated=True,
    )
    intrinsics.save(args.output)
    print(f"Calibration saved to {args.output}, RMS error: {ret:.4f}")


if __name__ == "__main__":
    main()
