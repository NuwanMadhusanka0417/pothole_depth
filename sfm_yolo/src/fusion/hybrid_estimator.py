"""Hybrid (Geometric + SfM) depth estimator.

This is the orchestrator that ties the three independently-tested
components together:

  Geometric  ─┐
              ├─→  Scale Recovery  ──→  Scaled SfM cloud
  SfM       ──┘                              │
                                             ├─→  Pothole depth from 3D
                                             │
                            Geometric depth ─┘  (consensus, confidence)

A single :meth:`HybridDepthEstimator.estimate_depth` call returns a
``HybridDepthResult`` containing both the final depth and a per-source
breakdown (so you can see *why* the pipeline reached its answer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..geometry.geometric_depth import (
    BBox,
    GeometricDepthEstimator,
    GeometricDepthResult,
)
from ..reconstruction.scale_recovery import ScaleRecoverer, ScaleRecoveryResult
from ..reconstruction.sfm_runner import SfMResult, SfMRunner
from ..utils.logging_utils import get_logger
from .confidence_scoring import (
    agreement_confidence,
    fuse_confidences,
)

_logger = get_logger("fusion.hybrid")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class HybridDepthResult:
    """Output of a hybrid depth estimate."""

    depth_m: float
    confidence: float
    geometric: Optional[GeometricDepthResult] = None
    sfm_depth_m: float = float("nan")
    sfm_scale: float = float("nan")
    sfm_scale_confidence: float = 0.0
    scale_recovery: Optional[ScaleRecoveryResult] = None
    point_cloud_path: Optional[Path] = None
    method: str = "hybrid"
    notes: str = ""
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "depth_m": self.depth_m,
            "confidence": self.confidence,
            "method": self.method,
            "geometric": self.geometric.as_dict() if self.geometric else None,
            "sfm_depth_m": self.sfm_depth_m,
            "sfm_scale": self.sfm_scale,
            "sfm_scale_confidence": self.sfm_scale_confidence,
            "scale_recovery": self.scale_recovery.as_dict() if self.scale_recovery else None,
            "point_cloud_path": str(self.point_cloud_path) if self.point_cloud_path else None,
            "notes": self.notes,
            "extras": self.extras,
        }


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------
class HybridDepthEstimator:
    """Combine the geometric depth estimator and an SfM runner."""

    def __init__(
        self,
        geometric: GeometricDepthEstimator,
        sfm_runner: Optional[SfMRunner] = None,
        *,
        geometric_weight: float = 0.55,
        sfm_weight: float = 0.45,
        disagreement_threshold: float = 0.30,
        min_confidence: float = 0.35,
    ) -> None:
        self.geometric = geometric
        self.sfm_runner = sfm_runner
        self.scale_recoverer = ScaleRecoverer(geometric)
        self.geometric_weight = float(geometric_weight)
        self.sfm_weight = float(sfm_weight)
        self.disagreement_threshold = float(disagreement_threshold)
        self.min_confidence = float(min_confidence)

    # ------------------------------------------------------------------
    # Geometric-only convenience
    # ------------------------------------------------------------------
    def geometric_only(
        self,
        bboxes: Sequence[BBox],
        *,
        road_pixel_ys: Optional[Sequence[float]] = None,
    ) -> HybridDepthResult:
        """Run only the geometric branch (fast path)."""
        if len(bboxes) == 1:
            geo = self.geometric.single_frame_depth(
                bboxes[0],
                road_pixel_y=(road_pixel_ys[0] if road_pixel_ys else None),
            )
        elif len(bboxes) == 2:
            geo = self.geometric.two_frame_depth(
                bboxes[0],
                bboxes[1],
                road_pixel_y_1=(road_pixel_ys[0] if road_pixel_ys else None),
                road_pixel_y_2=(road_pixel_ys[1] if road_pixel_ys else None),
            )
        else:
            geo = self.geometric.multi_frame_validation(
                bboxes, road_pixel_ys=road_pixel_ys
            )
        return HybridDepthResult(
            depth_m=geo.depth_m,
            confidence=geo.confidence,
            geometric=geo,
            method="geometric-only",
        )

    # ------------------------------------------------------------------
    # Full hybrid pipeline
    # ------------------------------------------------------------------
    def estimate_depth(
        self,
        *,
        bboxes: Sequence[BBox],
        video_path: Optional[str | Path] = None,
        sfm_frames: Optional[Sequence[np.ndarray]] = None,
        sfm_window: Optional[Tuple[int, int]] = None,
        ref_bbox_for_scale: Optional[BBox] = None,
        save_pointcloud: Optional[str | Path] = None,
        sfm_work_dir: Optional[str | Path] = None,
    ) -> HybridDepthResult:
        """Run geometric + SfM + scale recovery and fuse the depths.

        Parameters
        ----------
        bboxes : sequence of (x1,y1,x2,y2)
            One pothole bbox per frame used by the geometric branch.
        video_path : path-like, optional
            Source video for the SfM branch (preferred input).
        sfm_frames : sequence of ndarray, optional
            Pre-loaded frames; used if ``video_path`` is None.
        sfm_window : (start_frame, end_frame), optional
            Frame range inside ``video_path`` to use for SfM.
        ref_bbox_for_scale : (x1,y1,x2,y2), optional
            Bbox in the reference SfM image used to position the road
            scale samples. Defaults to the median bbox.
        save_pointcloud : path-like, optional
            If provided, the scaled point cloud is written as ASCII PLY.
        """
        if not bboxes:
            raise ValueError("bboxes must contain at least one bbox")

        # 1) Geometric branch
        geo = self._run_geometric(bboxes)
        _logger.info(
            "Geometric depth: %.3f m  conf=%.3f  method=%s",
            geo.depth_m, geo.confidence, geo.method,
        )

        # 2) SfM branch (best-effort)
        sfm_result: Optional[SfMResult] = None
        scale_res: Optional[ScaleRecoveryResult] = None
        sfm_depth = float("nan")
        ply_path: Optional[Path] = None
        sfm_notes = ""

        if self.sfm_runner is not None:
            try:
                sfm_result = self._run_sfm(video_path, sfm_frames, sfm_window, sfm_work_dir)
            except Exception as exc:  # pragma: no cover - depends on env
                _logger.warning("SfM stage failed: %s", exc)
                sfm_notes = f"sfm_failed: {exc}"
            else:
                _logger.info(
                    "SfM: method=%s, points=%d", sfm_result.method, sfm_result.num_points
                )
                ref_bbox = ref_bbox_for_scale or self._median_bbox(bboxes)
                scale_res = self.scale_recoverer.recover(sfm_result, bbox=ref_bbox)
                if scale_res.is_valid():
                    scaled = ScaleRecoverer.apply_scale(sfm_result, scale_res.scale)
                    sfm_depth = self._depth_from_pointcloud(scaled, ref_bbox)
                    if save_pointcloud is not None:
                        ply_path = scaled.save_ply(save_pointcloud)
                    sfm_result = scaled
                else:
                    sfm_notes = f"scale_invalid: {scale_res.notes}"

        # 3) Fusion
        depth, confidence, fusion_notes = self._fuse(geo, sfm_depth, scale_res)
        notes = " | ".join(filter(None, (geo.notes, sfm_notes, fusion_notes)))

        return HybridDepthResult(
            depth_m=depth,
            confidence=confidence,
            geometric=geo,
            sfm_depth_m=sfm_depth,
            sfm_scale=scale_res.scale if scale_res else float("nan"),
            sfm_scale_confidence=scale_res.confidence if scale_res else 0.0,
            scale_recovery=scale_res,
            point_cloud_path=ply_path,
            method=("hybrid" if (scale_res and scale_res.is_valid()) else "geometric-fallback"),
            notes=notes,
            extras={
                "n_bboxes": len(bboxes),
                "sfm_method": sfm_result.method if sfm_result else None,
                "sfm_num_points": sfm_result.num_points if sfm_result else 0,
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_geometric(self, bboxes: Sequence[BBox]) -> GeometricDepthResult:
        if len(bboxes) == 1:
            return self.geometric.single_frame_depth(bboxes[0])
        if len(bboxes) == 2:
            return self.geometric.two_frame_depth(bboxes[0], bboxes[1])
        return self.geometric.multi_frame_validation(bboxes)

    def _run_sfm(
        self,
        video_path: Optional[str | Path],
        sfm_frames: Optional[Sequence[np.ndarray]],
        window: Optional[Tuple[int, int]],
        work_dir: Optional[str | Path],
    ) -> SfMResult:
        if self.sfm_runner is None:
            raise RuntimeError("SfMRunner not configured")
        if sfm_frames is not None:
            frames_list = list(sfm_frames)
            if window is not None:
                start_frame, end_frame = window
                frames_list = frames_list[start_frame : end_frame + 1]
            return self.sfm_runner.run_from_frames(frames_list, work_dir=work_dir)
        if video_path is None:
            raise ValueError("Either video_path or sfm_frames must be provided")
        if window is not None:
            start_frame, end_frame = window
        else:
            start_frame, end_frame = 0, None
        return self.sfm_runner.run_from_video(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            work_dir=work_dir,
        )

    def _depth_from_pointcloud(
        self,
        scaled: SfMResult,
        bbox: BBox,
    ) -> float:
        """Estimate pothole depth from the scaled SfM cloud + bbox.

        We project all 3D points into the reference image, keep those
        falling inside the bbox, and look at the spread of their
        camera-1 ``Y`` coordinate (the camera optical axis points along
        ``Z``, but the road plane lies along the camera ``-Y`` axis).
        Pothole depth ~= ``Y_max - Y_min`` of the inliers.
        """
        if not scaled.poses or scaled.K is None or scaled.points_3d.size == 0:
            return float("nan")
        ref_pose = scaled.poses[0]
        cam_pts = (ref_pose.R @ scaled.points_3d.T).T + ref_pose.t  # (N, 3)
        # Project to pixels
        proj = (scaled.K @ cam_pts.T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = proj[:, :2] / proj[:, 2:3]
        x1, y1, x2, y2 = bbox
        in_bbox = (
            (uv[:, 0] >= x1)
            & (uv[:, 0] <= x2)
            & (uv[:, 1] >= y1)
            & (uv[:, 1] <= y2)
            & np.isfinite(uv[:, 0])
            & np.isfinite(uv[:, 1])
        )
        sel = cam_pts[in_bbox]
        if sel.shape[0] < 4:
            return float("nan")

        # In a forward-facing camera, the road lies below the optical
        # centre, so larger y values correspond to closer/lower points.
        # Use the 5/95 percentile spread of y to be robust to outliers.
        y_low, y_high = np.percentile(sel[:, 1], [5, 95])
        depth = float(abs(y_high - y_low))
        if not np.isfinite(depth) or depth <= 0:
            return float("nan")
        return depth

    def _fuse(
        self,
        geo: GeometricDepthResult,
        sfm_depth: float,
        scale_res: Optional[ScaleRecoveryResult],
    ) -> Tuple[float, float, str]:
        """Combine geometric / SfM depths and confidences."""
        if not np.isfinite(sfm_depth) or scale_res is None or not scale_res.is_valid():
            return geo.depth_m, geo.confidence, "fallback: geometric only"

        agreement = agreement_confidence(geo.depth_m, sfm_depth)
        rel_gap = (
            abs(geo.depth_m - sfm_depth) / max(1e-6, 0.5 * (geo.depth_m + sfm_depth))
        )
        confidence = fuse_confidences(
            geo.confidence,
            scale_res.confidence,
            agreement,
            geometric_weight=self.geometric_weight,
            sfm_weight=self.sfm_weight,
        )

        # Weighted average of the two depths, weighted by per-source
        # confidence.
        w_g = max(1e-6, geo.confidence) * self.geometric_weight
        w_s = max(1e-6, scale_res.confidence) * self.sfm_weight
        depth = float((w_g * geo.depth_m + w_s * sfm_depth) / (w_g + w_s))

        notes = f"agreement={agreement:.3f}, rel_gap={rel_gap:.3f}"
        if rel_gap > self.disagreement_threshold:
            notes += " | disagreement>threshold"
            confidence *= 0.5  # penalise

        if confidence < self.min_confidence:
            notes += " | below min_confidence"

        return depth, float(np.clip(confidence, 0.0, 1.0)), notes

    @staticmethod
    def _median_bbox(bboxes: Sequence[BBox]) -> BBox:
        arr = np.asarray(bboxes, dtype=np.float64)
        med = np.median(arr, axis=0)
        return tuple(float(x) for x in med)


# ---------------------------------------------------------------------------
# Optional: pothole surface area / volume from a scaled point cloud
# ---------------------------------------------------------------------------
def estimate_volume_from_pointcloud(
    points_xyz: np.ndarray,
    *,
    method: str = "convex-hull",
) -> Tuple[float, float]:
    """Return ``(surface_area_m2, volume_m3)`` from a metric point cloud.

    ``method='convex-hull'`` uses scipy's QHull and is robust for
    sparse clouds. Returns ``(nan, nan)`` if there are too few points.
    """
    if points_xyz.shape[0] < 4:
        return float("nan"), float("nan")
    if method == "convex-hull":
        try:
            from scipy.spatial import ConvexHull
        except ImportError:  # pragma: no cover
            return float("nan"), float("nan")
        try:
            hull = ConvexHull(points_xyz)
            return float(hull.area), float(hull.volume)
        except Exception:  # pragma: no cover - degenerate clouds
            return float("nan"), float("nan")
    raise ValueError(f"unknown method {method!r}")
