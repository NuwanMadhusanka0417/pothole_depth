"""Camera model and intrinsics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    width: int | None = None
    height: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    calibrated: bool = False
    camera_height_m: float | None = None

    @property
    def K(self) -> np.ndarray | None:
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

    def default_K(self, width: int, height: int) -> np.ndarray:
        """Approximate intrinsics when uncalibrated (lower geometry confidence)."""
        f = max(width, height)
        return np.array([
            [f, 0, width / 2],
            [0, f, height / 2],
            [0, 0, 1],
        ], dtype=np.float64)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CameraIntrinsics:
        return cls(
            width=config.get("image_width"),
            height=config.get("image_height"),
            fx=config.get("fx"),
            fy=config.get("fy"),
            cx=config.get("cx"),
            cy=config.get("cy"),
            k1=config.get("k1", 0.0),
            k2=config.get("k2", 0.0),
            p1=config.get("p1", 0.0),
            p2=config.get("p2", 0.0),
            k3=config.get("k3", 0.0),
            calibrated=bool(config.get("calibrated", False)),
            camera_height_m=config.get("camera_height_m"),
        )

    def save(self, path: Path) -> None:
        data = {
            "image_width": self.width,
            "image_height": self.height,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "k1": self.k1, "k2": self.k2, "p1": self.p1, "p2": self.p2, "k3": self.k3,
            "calibrated": self.calibrated,
            "camera_height_m": self.camera_height_m,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f)
