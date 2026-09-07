"""Camera wrapper."""

from __future__ import annotations

from pothole_vision.geometry.intrinsics import CameraIntrinsics


class Camera:
    def __init__(self, intrinsics: CameraIntrinsics, width: int, height: int) -> None:
        self.intrinsics = intrinsics
        self.width = width
        self.height = height

    @property
    def K(self) -> np.ndarray:
        import numpy as np
        k = self.intrinsics.K
        if k is None:
            return self.intrinsics.default_K(self.width, self.height)
        return k

    @property
    def geometry_confidence_factor(self) -> float:
        return 1.0 if self.intrinsics.calibrated else 0.6
