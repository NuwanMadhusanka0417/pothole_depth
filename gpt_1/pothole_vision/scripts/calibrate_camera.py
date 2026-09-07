#!/usr/bin/env python3
"""Checkerboard camera calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from _bootstrap import PROJECT_ROOT
import _bootstrap  # noqa: F401

from pothole_vision.geometry.intrinsics import CameraIntrinsics, save_camera_intrinsics


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate camera from checkerboard images")
    parser.add_argument("--images", required=True, type=Path, help="Folder of checkerboard images")
    parser.add_argument("--cols", type=int, default=9, help="Inner corners columns")
    parser.add_argument("--rows", type=int, default=6, help="Inner corners rows")
    parser.add_argument("--output", default="data/calibration/camera.yaml", type=Path)
    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)
    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)

    obj_points, img_points = [], []
    image_size = None
    images = list(Path(args.images).glob("*.jpg")) + list(Path(args.images).glob("*.png"))

    for img_path in images:
        img = cv2.imread(str(img_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_points.append(objp)
            img_points.append(corners)

    if len(obj_points) < 3:
        raise RuntimeError(f"Need >=3 valid checkerboard images, found {len(obj_points)}")

    ret, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    output = PROJECT_ROOT / args.output if not args.output.is_absolute() else args.output

    intrinsics = CameraIntrinsics(
        image_width=image_size[0],
        image_height=image_size[1],
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        k1=float(dist[0, 0]),
        k2=float(dist[0, 1]),
        p1=float(dist[0, 2]),
        p2=float(dist[0, 3]),
        k3=float(dist[0, 4]) if dist.shape[1] > 4 else 0.0,
        calibrated=True,
    )
    save_camera_intrinsics(output, intrinsics)
    print(f"Calibration RMS error: {ret:.4f}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
