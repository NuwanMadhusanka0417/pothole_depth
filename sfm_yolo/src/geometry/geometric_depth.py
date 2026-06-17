"""Geometric (trigonometric) pothole depth estimator.

This module is the **mathematical core** of the project. Given a
calibrated dash-camera and a pothole bounding box, it returns an
absolute metric depth estimate (no GPS / no stereo required).

Mathematical model
==================

We model the camera as a pinhole at height :math:`h` above a flat road
surface. For any pixel row ``y`` we can compute the angle below the
horizon::

    theta(y) = arctan( (y - cy) / f ) + pitch

The forward distance to the point on the road imaged at row ``y`` is::

    d(y) = h / tan( theta(y) )

The pothole sits *inside* the road surface, so an estimate of the
**pothole depth** (z-distance below the surrounding road) follows from
two horizontal lines that bracket the pothole:

* ``y_road``    — the row where the road plane is intersected if the
  pothole was filled; in practice the *top edge* of the bounding box.
* ``y_bottom``  — the *bottom edge* of the bounding box, where the
  bottom of the pothole is visible.

Plugging both rows into the equations above gives two distances along
the road, ``d_road`` and ``d_bottom``. The vertical drop *into* the
pothole (the depth in the colloquial sense) is::

    depth_along_road = d_road - d_bottom        # >= 0 if pothole is real
    depth_vertical   = depth_along_road * tan(theta_road)

This is the "single-frame" estimate. It is the *most direct* but also
the noisiest, because it depends on a single bbox bottom edge.

Two-frame triangulation
-----------------------

A second frame, separated by camera motion, gives an independent
measurement. Averaging ``depth`` across two frames and comparing the
two values yields both a more accurate estimate **and** a confidence
score (small disagreement => high confidence).

Multi-frame validation
----------------------

Generalising to N frames we keep the median (robust to outliers) and
report a confidence based on the median absolute deviation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.camera_calibration import CameraIntrinsics
from ..utils.logging_utils import get_logger
from .angle_utils import (
    angle_to_distance,
    horizon_pixel,
    median_with_confidence,
    pixel_y_to_angle,
    pixel_y_to_distance,
)

_logger = get_logger("geometry")

BBox = Tuple[float, float, float, float]   # (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class GeometricDepthResult:
    """Result of a geometric depth estimate."""

    depth_m: float                       # estimated pothole depth (metres)
    distance_to_road_m: float            # distance along the road (metres)
    distance_to_bottom_m: float          # distance to bbox bottom (metres)
    theta_road_rad: float
    theta_bottom_rad: float
    confidence: float                    # in [0, 1]
    method: str                          # 'single' | 'two-frame' | 'multi-frame'
    per_frame_depths: List[float] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "depth_m": self.depth_m,
            "distance_to_road_m": self.distance_to_road_m,
            "distance_to_bottom_m": self.distance_to_bottom_m,
            "theta_road_deg": math.degrees(self.theta_road_rad),
            "theta_bottom_deg": math.degrees(self.theta_bottom_rad),
            "confidence": self.confidence,
            "method": self.method,
            "per_frame_depths_m": list(self.per_frame_depths),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------
class GeometricDepthEstimator:
    """Estimate pothole depth from one or more bounding boxes.

    Parameters
    ----------
    intrinsics : CameraIntrinsics
        Calibrated camera (focal length, principal point, mounting height).
    """

    def __init__(self, intrinsics: CameraIntrinsics) -> None:
        self.intrinsics = intrinsics
        self._h = intrinsics.camera_height_m
        self._fy = intrinsics.fy
        self._cy = intrinsics.cy
        self._pitch = intrinsics.pitch_rad

        if self._h <= 0:
            raise ValueError("camera height must be positive")
        if self._fy <= 0:
            raise ValueError("focal length must be positive")

    # ------------------------------------------------------------------
    # Low-level conversions (vectorised, expose the math)
    # ------------------------------------------------------------------
    def pixel_to_angle(self, pixel_y: float | np.ndarray) -> float | np.ndarray:
        """Pixel row -> angle below horizon (radians)."""
        return pixel_y_to_angle(
            pixel_y, cy=self._cy, focal_length_px=self._fy, pitch_rad=self._pitch
        )

    def angle_to_distance(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Angle below horizon -> ground-plane distance (metres)."""
        return angle_to_distance(theta, camera_height_m=self._h)

    def pixel_to_distance(self, pixel_y: float | np.ndarray) -> float | np.ndarray:
        """One-shot pixel row -> distance (metres)."""
        return pixel_y_to_distance(
            pixel_y,
            camera_height_m=self._h,
            cy=self._cy,
            focal_length_px=self._fy,
            pitch_rad=self._pitch,
        )

    @property
    def horizon_pixel(self) -> float:
        return horizon_pixel(
            cy=self._cy, focal_length_px=self._fy, pitch_rad=self._pitch
        )

    # ------------------------------------------------------------------
    # Single-frame estimate
    # ------------------------------------------------------------------
    def single_frame_depth(
        self,
        bbox: BBox,
        *,
        road_pixel_y: Optional[float] = None,
    ) -> GeometricDepthResult:
        """Estimate pothole depth from a single bounding box.

        Parameters
        ----------
        bbox : (x1, y1, x2, y2)
            Pothole bounding box in pixel coordinates (top-left origin).
        road_pixel_y : float, optional
            Row to use as the *road surface* reference. If ``None`` we
            use the top edge ``y1`` of the bounding box (a reasonable
            proxy for the road plane that would exist if the pothole
            were filled).

        Returns
        -------
        GeometricDepthResult
        """
        x1, y1, x2, y2 = self._validate_bbox(bbox)
        y_road = float(road_pixel_y) if road_pixel_y is not None else float(y1)
        y_bot = float(y2)

        if y_road >= y_bot:
            raise ValueError(
                f"road row ({y_road}) must be above the pothole bottom row ({y_bot})"
            )

        theta_road = float(self.pixel_to_angle(y_road))
        theta_bot = float(self.pixel_to_angle(y_bot))

        # Both rows must be below the horizon (positive theta) for the
        # ground-plane projection to be valid.
        if theta_road <= 0 or theta_bot <= 0:
            return GeometricDepthResult(
                depth_m=float("nan"),
                distance_to_road_m=float("nan"),
                distance_to_bottom_m=float("nan"),
                theta_road_rad=theta_road,
                theta_bottom_rad=theta_bot,
                confidence=0.0,
                method="single",
                notes="bbox row above horizon - cannot project to road plane",
            )

        d_road = float(self.angle_to_distance(theta_road))
        d_bot = float(self.angle_to_distance(theta_bot))
        depth_along = max(0.0, d_road - d_bot)
        depth_vertical = depth_along * math.tan(theta_road)

        # Confidence: a simple monotone function of the bbox height in
        # pixels. Tiny boxes are unreliable.
        bbox_h = max(1.0, y_bot - y1)
        confidence = float(np.clip(bbox_h / 60.0, 0.0, 1.0))  # ~60 px = high conf

        return GeometricDepthResult(
            depth_m=depth_vertical,
            distance_to_road_m=d_road,
            distance_to_bottom_m=d_bot,
            theta_road_rad=theta_road,
            theta_bottom_rad=theta_bot,
            confidence=confidence,
            method="single",
            per_frame_depths=[depth_vertical],
            notes=f"bbox_h={bbox_h:.1f}px",
        )

    # ------------------------------------------------------------------
    # Two-frame estimate
    # ------------------------------------------------------------------
    def two_frame_depth(
        self,
        bbox_1: BBox,
        bbox_2: BBox,
        *,
        road_pixel_y_1: Optional[float] = None,
        road_pixel_y_2: Optional[float] = None,
    ) -> GeometricDepthResult:
        """Average two single-frame estimates from frames 1 and 2.

        The two boxes should refer to **the same pothole**, viewed from
        slightly different positions (the camera has moved forward
        between frames). Disagreement between the two estimates yields
        an explicit confidence value.
        """
        r1 = self.single_frame_depth(bbox_1, road_pixel_y=road_pixel_y_1)
        r2 = self.single_frame_depth(bbox_2, road_pixel_y=road_pixel_y_2)

        if not (math.isfinite(r1.depth_m) and math.isfinite(r2.depth_m)):
            return GeometricDepthResult(
                depth_m=float("nan"),
                distance_to_road_m=float("nan"),
                distance_to_bottom_m=float("nan"),
                theta_road_rad=float("nan"),
                theta_bottom_rad=float("nan"),
                confidence=0.0,
                method="two-frame",
                notes="one or both frames invalid",
            )

        depth_avg = 0.5 * (r1.depth_m + r2.depth_m)
        denom = max(1e-6, depth_avg)
        rel_var = abs(r1.depth_m - r2.depth_m) / denom
        confidence_consistency = 1.0 / (1.0 + rel_var)
        confidence = float(
            np.clip(0.5 * (r1.confidence + r2.confidence) * confidence_consistency, 0.0, 1.0)
        )

        return GeometricDepthResult(
            depth_m=depth_avg,
            distance_to_road_m=0.5 * (r1.distance_to_road_m + r2.distance_to_road_m),
            distance_to_bottom_m=0.5 * (r1.distance_to_bottom_m + r2.distance_to_bottom_m),
            theta_road_rad=0.5 * (r1.theta_road_rad + r2.theta_road_rad),
            theta_bottom_rad=0.5 * (r1.theta_bottom_rad + r2.theta_bottom_rad),
            confidence=confidence,
            method="two-frame",
            per_frame_depths=[r1.depth_m, r2.depth_m],
            notes=f"rel_var={rel_var:.3f}",
        )

    # ------------------------------------------------------------------
    # Multi-frame robust estimate
    # ------------------------------------------------------------------
    def multi_frame_validation(
        self,
        bboxes: Sequence[BBox],
        *,
        road_pixel_ys: Optional[Sequence[float]] = None,
    ) -> GeometricDepthResult:
        """Robust depth estimate from N >= 2 same-pothole bounding boxes.

        Uses the **median** of per-frame depths (outlier tolerant) and
        reports a confidence based on the median absolute deviation.
        """
        if len(bboxes) < 2:
            raise ValueError("multi_frame_validation requires at least 2 bboxes")
        if road_pixel_ys is not None and len(road_pixel_ys) != len(bboxes):
            raise ValueError("road_pixel_ys must match bboxes length")

        per_frame_results: List[GeometricDepthResult] = []
        depths: List[float] = []
        for i, bbox in enumerate(bboxes):
            road_y = road_pixel_ys[i] if road_pixel_ys is not None else None
            res = self.single_frame_depth(bbox, road_pixel_y=road_y)
            per_frame_results.append(res)
            if math.isfinite(res.depth_m):
                depths.append(res.depth_m)

        if not depths:
            return GeometricDepthResult(
                depth_m=float("nan"),
                distance_to_road_m=float("nan"),
                distance_to_bottom_m=float("nan"),
                theta_road_rad=float("nan"),
                theta_bottom_rad=float("nan"),
                confidence=0.0,
                method="multi-frame",
                notes="no valid frames",
            )

        median_depth, conf_consistency = median_with_confidence(depths)
        avg_bbox_conf = float(np.mean([r.confidence for r in per_frame_results]))
        confidence = float(np.clip(avg_bbox_conf * conf_consistency, 0.0, 1.0))

        avg_d_road = float(np.mean([r.distance_to_road_m for r in per_frame_results]))
        avg_d_bot = float(np.mean([r.distance_to_bottom_m for r in per_frame_results]))
        avg_th_road = float(np.mean([r.theta_road_rad for r in per_frame_results]))
        avg_th_bot = float(np.mean([r.theta_bottom_rad for r in per_frame_results]))

        return GeometricDepthResult(
            depth_m=float(median_depth),
            distance_to_road_m=avg_d_road,
            distance_to_bottom_m=avg_d_bot,
            theta_road_rad=avg_th_road,
            theta_bottom_rad=avg_th_bot,
            confidence=confidence,
            method="multi-frame",
            per_frame_depths=depths,
            notes=f"n={len(depths)}, mad-confidence={conf_consistency:.3f}",
        )

    # ------------------------------------------------------------------
    # Road-point extraction (for SfM scale recovery)
    # ------------------------------------------------------------------
    def extract_road_points(
        self,
        image_shape: Tuple[int, int],
        *,
        bbox: Optional[BBox] = None,
        n_points: int = 8,
        margin_px: int = 20,
    ) -> List[Tuple[float, float, float]]:
        """Sample N points on the road plane with metric distances.

        These are used as scale references to convert an unscaled SfM
        reconstruction into metres.

        Parameters
        ----------
        image_shape : (H, W)
        bbox : (x1, y1, x2, y2), optional
            If provided, the sampled points are taken just *in front of*
            the pothole's top edge (between ``y1 - margin`` and the
            horizon).
        n_points : int
            Number of samples.
        margin_px : int
            Vertical margin away from horizon and bbox edges.

        Returns
        -------
        list of (pixel_x, pixel_y, distance_m)
        """
        H, W = image_shape[:2]
        horizon = self.horizon_pixel
        y_top = max(horizon + margin_px, 0.0)
        y_bot = float(H - 1)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            y_bot = max(y_top + 1.0, float(y1) - margin_px)

        ys = np.linspace(y_top + 1.0, y_bot, n_points)
        x_center = 0.5 * W
        out: List[Tuple[float, float, float]] = []
        for y in ys:
            d = float(self.pixel_to_distance(y))
            if math.isfinite(d) and d > 0:
                out.append((float(x_center), float(y), d))
        return out

    # ------------------------------------------------------------------
    # Confidence helper (exposed for hybrid fusion)
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_confidence(measurements: Iterable[float]) -> float:
        """Confidence in [0, 1] derived from the spread of measurements."""
        arr = np.asarray(list(measurements), dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return 0.0
        m = float(np.mean(arr))
        if abs(m) < 1e-9:
            return 0.0
        rel_std = float(np.std(arr) / abs(m))
        return float(np.clip(1.0 / (1.0 + rel_std), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_bbox(bbox: BBox) -> Tuple[float, float, float, float]:
        if len(bbox) != 4:
            raise ValueError("bbox must be a 4-tuple (x1, y1, x2, y2)")
        x1, y1, x2, y2 = (float(v) for v in bbox)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"degenerate bbox: {bbox}")
        return x1, y1, x2, y2
