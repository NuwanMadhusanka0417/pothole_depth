"""Camera model wrapper."""

from __future__ import annotations

from pothole_vision.geometry.intrinsics import CameraIntrinsics


class Camera:
    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics

    @property
    def K(self):
        return self.intrinsics.K

    @property
    def is_calibrated(self) -> bool:
        return self.intrinsics.calibrated
