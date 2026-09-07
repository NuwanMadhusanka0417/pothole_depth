"""Camera intrinsics and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    calibrated: bool = False
    camera_height_m: float = 1.2

    @property
    def K(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    @property
    def geometry_confidence_factor(self) -> float:
        return 1.0 if self.calibrated else 0.6


def load_camera_intrinsics(path: Path) -> CameraIntrinsics:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return CameraIntrinsics(
        image_width=int(raw.get("image_width", 1920)),
        image_height=int(raw.get("image_height", 1080)),
        fx=float(raw.get("fx", 1200)),
        fy=float(raw.get("fy", 1200)),
        cx=float(raw.get("cx", 960)),
        cy=float(raw.get("cy", 540)),
        k1=float(raw.get("k1", 0)),
        k2=float(raw.get("k2", 0)),
        p1=float(raw.get("p1", 0)),
        p2=float(raw.get("p2", 0)),
        k3=float(raw.get("k3", 0)),
        calibrated=bool(raw.get("calibrated", False)),
        camera_height_m=float(raw.get("camera_height_m", 1.2)),
    )


def save_camera_intrinsics(path: Path, intrinsics: CameraIntrinsics) -> None:
    data = {
        "calibrated": intrinsics.calibrated,
        "image_width": intrinsics.image_width,
        "image_height": intrinsics.image_height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "k1": intrinsics.k1,
        "k2": intrinsics.k2,
        "p1": intrinsics.p1,
        "p2": intrinsics.p2,
        "k3": intrinsics.k3,
        "camera_height_m": intrinsics.camera_height_m,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False)
