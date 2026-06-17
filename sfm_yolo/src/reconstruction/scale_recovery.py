"""Scale recovery: turn an unscaled SfM point cloud into metres.

Pipeline (see :class:`ScaleRecoverer.recover`):

  1. The geometric depth model produces a list of *road-plane* points
     in front of the camera with known metric distances (from
     ``GeometricDepthEstimator.extract_road_points``).
  2. Each of those points is projected with the (unscaled) SfM camera-1
     pose and matched to the nearest 3D SfM point.
  3. For every successful match we form a per-point scale estimate
     ``lambda_i = d_geometric_i / d_sfm_i``.
  4. The final scale factor is the **median** across points (robust to
     outliers); confidence is derived from the MAD.

This is the bridge that converts the geometric "absolute scale" trick
into a metric SfM cloud, so we get both per-pothole detail (SfM) and
real-world units (geometry) in a single 3D model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..geometry.geometric_depth import GeometricDepthEstimator
from ..utils.logging_utils import get_logger
from . import feature_tracking as ft
from .sfm_runner import SfMResult

_logger = get_logger("reconstruction.scale")


@dataclass
class ScaleRecoveryResult:
    """Outcome of attempting to scale an SfM reconstruction."""

    scale: float
    confidence: float
    num_inliers: int
    per_point_scales: List[float] = field(default_factory=list)
    notes: str = ""

    def is_valid(self) -> bool:
        return (
            np.isfinite(self.scale)
            and self.scale > 0
            and self.num_inliers >= 2
            and self.confidence > 0
        )

    def as_dict(self) -> dict:
        return {
            "scale": self.scale,
            "confidence": self.confidence,
            "num_inliers": self.num_inliers,
            "per_point_scales": list(self.per_point_scales),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Recoverer
# ---------------------------------------------------------------------------
class ScaleRecoverer:
    """Estimate a metric scale factor for an SfM reconstruction."""

    def __init__(
        self,
        geometric: GeometricDepthEstimator,
        *,
        max_pixel_distance: float = 30.0,
        min_inliers: int = 3,
        outlier_mad_threshold: float = 3.0,
    ) -> None:
        self.geometric = geometric
        self.max_pixel_distance = float(max_pixel_distance)
        self.min_inliers = int(min_inliers)
        self.outlier_mad_threshold = float(outlier_mad_threshold)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def recover(
        self,
        sfm: SfMResult,
        *,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        n_road_points: int = 12,
        ref_pose_index: int = 0,
    ) -> ScaleRecoveryResult:
        """Compute a metric scale factor for ``sfm``.

        Parameters
        ----------
        sfm : SfMResult
            Output of :class:`SfMRunner`.
        bbox : (x1, y1, x2, y2), optional
            Pothole bbox in the reference image (used to position the
            road samples *just in front of* the pothole).
        n_road_points : int
            Number of road-plane samples to draw from the geometric model.
        ref_pose_index : int
            Index of the pose to use as reference image (the 0-th pose
            is conventionally the world frame).
        """
        if sfm.K is None or sfm.points_3d.size == 0:
            return ScaleRecoveryResult(
                scale=float("nan"),
                confidence=0.0,
                num_inliers=0,
                notes="empty SfM result",
            )
        if not sfm.poses:
            return ScaleRecoveryResult(
                scale=float("nan"),
                confidence=0.0,
                num_inliers=0,
                notes="no camera poses",
            )

        if sfm.image_size is None:
            # Fall back to the calibration image size
            image_shape = (self.geometric.intrinsics.height, self.geometric.intrinsics.width)
        else:
            w, h = sfm.image_size
            image_shape = (h, w)

        road_pts = self.geometric.extract_road_points(
            image_shape=image_shape,
            bbox=bbox,
            n_points=n_road_points,
        )
        if not road_pts:
            return ScaleRecoveryResult(
                scale=float("nan"),
                confidence=0.0,
                num_inliers=0,
                notes="no geometric road points available",
            )

        ref_pose = sfm.poses[ref_pose_index]

        per_pt_scales: List[float] = []
        for px, py, d_geom in road_pts:
            match = ft.nearest_3d_to_pixel(
                sfm.points_3d,
                target_pixel=(px, py),
                K=sfm.K,
                R=ref_pose.R,
                t=ref_pose.t,
                max_pixel_distance=self.max_pixel_distance,
            )
            if match is None:
                continue
            _, xyz, _ = match
            # Distance from the reference camera centre to this 3D point
            cam_pt = ref_pose.R @ xyz + ref_pose.t
            d_sfm = float(np.linalg.norm(cam_pt))
            if d_sfm <= 1e-6 or not np.isfinite(d_sfm):
                continue
            per_pt_scales.append(d_geom / d_sfm)

        if len(per_pt_scales) < self.min_inliers:
            return ScaleRecoveryResult(
                scale=float("nan"),
                confidence=0.0,
                num_inliers=len(per_pt_scales),
                per_point_scales=per_pt_scales,
                notes=f"too few inliers ({len(per_pt_scales)} < {self.min_inliers})",
            )

        scales = np.asarray(per_pt_scales, dtype=np.float64)
        median = float(np.median(scales))
        mad = float(np.median(np.abs(scales - median))) or 1e-6
        keep = np.abs(scales - median) <= self.outlier_mad_threshold * mad
        kept = scales[keep]
        if kept.size < self.min_inliers:
            kept = scales

        scale_final = float(np.median(kept))
        rel_mad = float(np.median(np.abs(kept - scale_final)) / max(abs(scale_final), 1e-6))
        confidence = float(np.clip(1.0 / (1.0 + rel_mad), 0.0, 1.0))

        _logger.info(
            "Recovered SfM scale=%.4f m/unit (n_in=%d, rel_mad=%.3f, conf=%.3f)",
            scale_final, kept.size, rel_mad, confidence,
        )
        return ScaleRecoveryResult(
            scale=scale_final,
            confidence=confidence,
            num_inliers=int(kept.size),
            per_point_scales=per_pt_scales,
            notes=f"rel_mad={rel_mad:.3f}",
        )

    # ------------------------------------------------------------------
    # Apply scale
    # ------------------------------------------------------------------
    @staticmethod
    def apply_scale(sfm: SfMResult, scale: float) -> SfMResult:
        """Return a new :class:`SfMResult` with metric units applied."""
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid scale {scale}")
        scaled_points = sfm.points_3d * float(scale)
        scaled_poses = []
        for p in sfm.poses:
            scaled_poses.append(
                type(p)(
                    image_id=p.image_id,
                    name=p.name,
                    R=p.R.copy(),
                    t=(p.t.copy() * float(scale)),
                )
            )
        return SfMResult(
            points_3d=scaled_points,
            point_colors=sfm.point_colors,
            poses=scaled_poses,
            K=sfm.K,
            image_paths=list(sfm.image_paths),
            image_size=sfm.image_size,
            method=sfm.method + "+scaled",
            notes=f"scale={scale:.4f}; {sfm.notes}",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def per_point_scale_factors(
        geometric_distances: Sequence[float],
        sfm_distances: Sequence[float],
    ) -> List[float]:
        """Public helper: compute lambda_i = d_geo / d_sfm element-wise."""
        out: List[float] = []
        for dg, ds in zip(geometric_distances, sfm_distances):
            if ds and np.isfinite(dg) and np.isfinite(ds) and ds > 1e-6:
                out.append(float(dg) / float(ds))
        return out
