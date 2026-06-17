"""Camera intrinsics and calibration helpers.

The geometric depth model needs three things from the camera:

  * the **mounting height** above the road (``camera_height_m``),
  * the **focal length in pixels** (``focal_length_px``),
  * the **principal point** (image center, ``cx``, ``cy``).

Optional values:
  * a small ``pitch_deg`` correction for cameras that look slightly down,
  * radial / tangential ``distortion`` coefficients for undistortion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    """Intrinsic parameters and physical mounting of the dash-camera."""

    camera_height_m: float
    focal_length_px: float
    principal_point: Tuple[float, float]
    image_size: Tuple[int, int]              # (width, height)
    pitch_deg: float = 0.0
    distortion: Tuple[float, ...] = field(default_factory=lambda: (0.0,) * 5)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "CameraIntrinsics":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "CameraIntrinsics":
        required = ("camera_height_m", "focal_length_px", "principal_point", "image_size")
        for key in required:
            if key not in data:
                raise KeyError(f"camera config missing '{key}'")
        return cls(
            camera_height_m=float(data["camera_height_m"]),
            focal_length_px=float(data["focal_length_px"]),
            principal_point=tuple(map(float, data["principal_point"])),
            image_size=tuple(map(int, data["image_size"])),
            pitch_deg=float(data.get("pitch_deg", 0.0)),
            distortion=tuple(map(float, data.get("distortion", (0.0,) * 5))),
        )

    def to_dict(self) -> dict:
        return {
            "camera_height_m": self.camera_height_m,
            "focal_length_px": self.focal_length_px,
            "principal_point": list(self.principal_point),
            "image_size": list(self.image_size),
            "pitch_deg": self.pitch_deg,
            "distortion": list(self.distortion),
        }

    # ------------------------------------------------------------------
    # Common conveniences
    # ------------------------------------------------------------------
    @property
    def fx(self) -> float:
        return self.focal_length_px

    @property
    def fy(self) -> float:
        return self.focal_length_px

    @property
    def cx(self) -> float:
        return self.principal_point[0]

    @property
    def cy(self) -> float:
        return self.principal_point[1]

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]

    @property
    def pitch_rad(self) -> float:
        return math.radians(self.pitch_deg)

    def K(self) -> np.ndarray:
        """Return the 3x3 intrinsic matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def dist_coeffs(self) -> np.ndarray:
        return np.asarray(self.distortion, dtype=np.float64)

    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 * self.width / self.fx))

    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 * self.height / self.fy))

    def for_image_size(self, width: int, height: int) -> "CameraIntrinsics":
        """Return intrinsics scaled to match a processed frame size."""
        sx = width / self.width
        sy = height / self.height
        return CameraIntrinsics(
            camera_height_m=self.camera_height_m,
            focal_length_px=self.focal_length_px * sx,
            principal_point=(self.cx * sx, self.cy * sy),
            image_size=(int(width), int(height)),
            pitch_deg=self.pitch_deg,
            distortion=self.distortion,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CameraIntrinsics(h={self.camera_height_m}m, f={self.fx}px, "
            f"pp=({self.cx},{self.cy}), size={self.image_size}, "
            f"pitch={self.pitch_deg}deg)"
        )


def load_camera_calibration(path: str | Path) -> CameraIntrinsics:
    """Convenience wrapper that mirrors the public name."""
    return CameraIntrinsics.from_yaml(path)


# ---------------------------------------------------------------------------
# Optional: estimate focal length from a chessboard sequence
# ---------------------------------------------------------------------------
def estimate_focal_length_from_chessboard(
    images: Sequence[np.ndarray],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run an OpenCV chessboard calibration to estimate intrinsics.

    Returns
    -------
    focal_length_px : float
        Average of (fx, fy).
    K : ndarray (3, 3)
        Camera intrinsic matrix.
    dist : ndarray
        Distortion coefficients.
    """
    import cv2  # local import - keep utility importable without cv2 at top

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: Tuple[int, int] | None = None

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if image_size is None:
            image_size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 1e-3),
        )
        obj_points.append(objp.copy())
        img_points.append(corners)

    if not obj_points:
        raise RuntimeError("no chessboard patterns detected in any image")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    focal = float(0.5 * (K[0, 0] + K[1, 1]))
    return focal, K, dist
