"""Pothole depth as the drop of the floor below a fitted road plane.

This is the shared, source-agnostic core that replaces the flawed bounding-box
trigonometric estimate. It works on any **metric** 3D representation of the
pothole region:

* Path B (learned): a metric depth map from DepthAnything V2 -> back-projected
  to 3D via the camera intrinsics.
* Path A (physics): a metric SfM/MVS point cloud.

Method
------
1. Split the region into a **road ring** (around the pothole) and the
   **interior** (the opening).
2. Fit a plane to the road-ring points with RANSAC (robust to a few outliers
   and to mild road curvature).
3. Orient the plane normal "up" using the IMU **gravity** direction.
4. Depth = the largest perpendicular drop of interior points below that plane
   (a high percentile, to reject speckle). This is the true vertical depth the
   bounding-box method could never see.

Only the road plane and the gravity *sign* come from the IMU, so the result is
insensitive to small errors in the gravity-to-camera convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..utils.logging_utils import get_logger

_logger = get_logger("geometry.plane_depth")

BBox = Tuple[float, float, float, float]


@dataclass
class PlaneDepthResult:
    depth_m: float
    n_road_inliers: int
    n_interior_points: int
    confidence: float
    plane: Optional[Tuple[float, float, float, float]] = None  # (a,b,c,d), normal up
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "depth_m": self.depth_m,
            "n_road_inliers": self.n_road_inliers,
            "n_interior_points": self.n_interior_points,
            "confidence": self.confidence,
            "plane": list(self.plane) if self.plane is not None else None,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# RANSAC plane fit (numpy; no external solver needed)
# ---------------------------------------------------------------------------
def fit_plane_ransac(
    pts: np.ndarray,
    *,
    dist_thresh: float = 0.006,
    iters: int = 300,
    seed: int = 0,
) -> Optional[Tuple[np.ndarray, float, int]]:
    """Fit a plane ``n . x + d = 0`` (unit normal) to ``pts`` (N,3) by RANSAC.

    Returns ``(normal, d, n_inliers)`` or ``None`` if it fails.
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    best: Optional[Tuple[np.ndarray, float]] = None
    best_cnt = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p = pts[idx]
        nrm = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm = nrm / nn
        d = -float(nrm @ p[0])
        cnt = int((np.abs(pts @ nrm + d) < dist_thresh).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (nrm, d)
    if best is None:
        return None
    nrm, d = best
    # Refine on the inlier set via SVD (total least squares).
    inl = pts[np.abs(pts @ nrm + d) < dist_thresh]
    if len(inl) >= 3:
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c)
        nrm = vt[-1]
        nrm = nrm / (np.linalg.norm(nrm) + 1e-12)
        d = -float(nrm @ c)
    return nrm, d, best_cnt


# ---------------------------------------------------------------------------
# Core: depth from road + interior point sets
# ---------------------------------------------------------------------------
def depth_below_plane(
    road_pts: np.ndarray,
    interior_pts: np.ndarray,
    gravity_cam: np.ndarray,
    *,
    dist_thresh: float = 0.006,
    ransac_iters: int = 300,
    drop_percentile: float = 95.0,
) -> PlaneDepthResult:
    """Depth of ``interior_pts`` below the plane fitted to ``road_pts``.

    ``gravity_cam`` is the gravity direction in the SAME camera frame as the
    points (points down); it is used only to orient the plane normal upward and
    to pick the "below road" side.
    """
    road_pts = np.asarray(road_pts, dtype=np.float64).reshape(-1, 3)
    interior_pts = np.asarray(interior_pts, dtype=np.float64).reshape(-1, 3)
    if len(road_pts) < 3:
        return PlaneDepthResult(float("nan"), 0, len(interior_pts), 0.0,
                                notes="too few road points")
    if len(interior_pts) == 0:
        return PlaneDepthResult(float("nan"), 0, 0, 0.0, notes="no interior points")

    fit = fit_plane_ransac(road_pts, dist_thresh=dist_thresh, iters=ransac_iters)
    if fit is None:
        return PlaneDepthResult(float("nan"), 0, len(interior_pts), 0.0,
                                notes="plane fit failed")
    nrm, d, n_in = fit

    # Orient the normal "up" = opposite gravity.
    g = np.asarray(gravity_cam, dtype=np.float64)
    g = g / (np.linalg.norm(g) + 1e-12)
    if float(nrm @ (-g)) < 0:
        nrm, d = -nrm, -d

    # Signed distance: road ~ 0, interior floor (below road, along +gravity) < 0.
    s = interior_pts @ nrm + d
    # Deepest points have the most negative s; take a robust tail.
    drop = -float(np.percentile(s, 100.0 - drop_percentile))
    depth = max(0.0, drop)

    inlier_ratio = n_in / max(1, len(road_pts))
    # Confidence: reward a clean plane and a clear (not noisy) depression.
    below_frac = float((s < -dist_thresh).mean())
    confidence = float(np.clip(inlier_ratio * (0.5 + 0.5 * below_frac), 0.0, 1.0))

    return PlaneDepthResult(
        depth_m=depth,
        n_road_inliers=int(n_in),
        n_interior_points=int(len(interior_pts)),
        confidence=confidence,
        plane=(float(nrm[0]), float(nrm[1]), float(nrm[2]), float(d)),
        notes=f"inlier_ratio={inlier_ratio:.2f}, below_frac={below_frac:.2f}",
    )


# ---------------------------------------------------------------------------
# Convenience: depth straight from a metric depth map + bbox
# ---------------------------------------------------------------------------
def _backproject(us: np.ndarray, vs: np.ndarray, depth: np.ndarray,
                 fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Pixel (u,v) + metric depth Z -> camera-frame XYZ (N,3)."""
    z = depth
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    return np.stack([x, y, z], axis=-1)


def pothole_depth_from_depthmap(
    depth_map: np.ndarray,
    bbox: BBox,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    gravity_cam: np.ndarray,
    ring_margin: float = 0.6,
    interior_shrink: float = 0.15,
    dist_thresh: float = 0.006,
    drop_percentile: float = 95.0,
    min_valid_depth: float = 0.05,
    max_valid_depth: float = 30.0,
) -> PlaneDepthResult:
    """Pothole depth from a metric depth map (Path B, or a projected SfM map).

    Parameters
    ----------
    depth_map : (H, W) float
        Per-pixel metric depth (Z along the optical axis), in metres.
    bbox : (x1, y1, x2, y2)
        Pothole box in the same (upright) image the depth map came from.
    fx, fy, cx, cy : float
        Intrinsics of that (upright) image.
    gravity_cam : (3,)
        Gravity in the camera frame (points down). See
        :func:`..geometry.imu_orientation.gravity_camera_frame`.
    ring_margin : float
        How far to extend past the bbox (as a fraction of its size) to sample
        the surrounding road.
    interior_shrink : float
        Shrink the bbox by this fraction before sampling interior points, to
        avoid the rim.
    """
    H, W = depth_map.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)

    # Outer ROI (road ring lives between ROI and bbox) and inner interior box.
    ox1 = max(0, int(x1 - ring_margin * bw)); oy1 = max(0, int(y1 - ring_margin * bh))
    ox2 = min(W - 1, int(x2 + ring_margin * bw)); oy2 = min(H - 1, int(y2 + ring_margin * bh))
    ix1 = int(x1 + interior_shrink * bw); iy1 = int(y1 + interior_shrink * bh)
    ix2 = int(x2 - interior_shrink * bw); iy2 = int(y2 - interior_shrink * bh)

    uu, vv = np.meshgrid(np.arange(ox1, ox2 + 1), np.arange(oy1, oy2 + 1))
    dd = depth_map[oy1:oy2 + 1, ox1:ox2 + 1].astype(np.float64)
    valid = np.isfinite(dd) & (dd >= min_valid_depth) & (dd <= max_valid_depth)

    in_bbox = (uu >= x1) & (uu <= x2) & (vv >= y1) & (vv <= y2)
    in_interior = (uu >= ix1) & (uu <= ix2) & (vv >= iy1) & (vv <= iy2)

    road_sel = valid & (~in_bbox)          # ring around the pothole
    int_sel = valid & in_interior          # the opening

    road_pts = _backproject(uu[road_sel], vv[road_sel], dd[road_sel], fx, fy, cx, cy)
    int_pts = _backproject(uu[int_sel], vv[int_sel], dd[int_sel], fx, fy, cx, cy)

    return depth_below_plane(
        road_pts, int_pts, gravity_cam,
        dist_thresh=dist_thresh, drop_percentile=drop_percentile,
    )
