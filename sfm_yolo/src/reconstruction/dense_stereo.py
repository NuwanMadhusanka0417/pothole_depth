"""CPU dense stereo: a dense 3D cloud from two posed frames.

Why this exists
---------------
COLMAP's incremental mapping returns a **sparse** cloud (only SIFT keypoints):
on our clips that is ~1200 points for the whole scene and often <50 inside a
pothole -- far too few to fit the floor and measure a few-centimetre drop.
COLMAP's dense MVS (``patch_match_stereo``) would fix that, but it is CUDA-only
and this machine has ``pycolmap.has_cuda == False``.

Why flow instead of rectified SGBM
----------------------------------
The textbook CPU option is ``cv2.stereoRectify`` + ``StereoSGBM``. That assumes
a mostly **sideways** baseline. Our camera moves **forward** toward the pothole,
which puts the epipole inside the image -- rectification is degenerate there and
produces severely distorted images.

So we instead compute **dense optical flow** (Farneback) between two posed
frames and **triangulate every pixel** of the pothole region with the known
relative pose. This needs no rectification, works for arbitrary motion, runs on
CPU, and yields one 3D point per pixel instead of one per keypoint.

Scale: the pose translation comes from SfM, so the cloud is in SfM units. That
is fine -- :func:`..geometry.plane_depth.pothole_depth_from_points` anchors it to
metres using the known camera height.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..utils.logging_utils import get_logger

_logger = get_logger("reconstruction.dense_stereo")

BBox = Tuple[float, float, float, float]


def relative_pose(
    R_ref: np.ndarray, t_ref: np.ndarray,
    R_other: np.ndarray, t_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pose taking points from the reference camera frame to the other camera.

    Both inputs are ``cam_from_world`` (``X_cam = R X_world + t``).
    """
    R_rel = R_other @ R_ref.T
    t_rel = t_other - R_rel @ t_ref
    return R_rel, t_rel.reshape(3)


def dense_cloud_from_pair(
    img_ref: np.ndarray,
    img_other: np.ndarray,
    K: np.ndarray,
    R_rel: np.ndarray,
    t_rel: np.ndarray,
    *,
    roi: Optional[BBox] = None,
    step: int = 1,
    max_reproj_err_px: float = 1.5,
    min_flow_px: float = 0.3,
) -> np.ndarray:
    """Triangulate a dense point cloud (in the reference camera frame).

    Parameters
    ----------
    img_ref, img_other : ndarray
        The two frames (BGR or gray), same size.
    K : (3, 3)
        Intrinsics shared by both frames.
    R_rel, t_rel :
        Relative pose from the reference camera to the other camera.
    roi : (x1, y1, x2, y2), optional
        Only triangulate pixels inside this box (the pothole + surrounding road).
    step : int
        Pixel stride (1 = every pixel).
    max_reproj_err_px : float
        Drop points whose reprojection disagrees with the matched pixels.
    min_flow_px : float
        Drop pixels that barely moved -- their triangulation is ill-conditioned.

    Returns
    -------
    (N, 3) points in the reference camera frame (SfM scale).
    """
    g1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY) if img_ref.ndim == 3 else img_ref
    g2 = cv2.cvtColor(img_other, cv2.COLOR_BGR2GRAY) if img_other.ndim == 3 else img_other

    # Dense flow over the whole frame (needs full context), sampled in the ROI.
    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None,
        pyr_scale=0.5, levels=5, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0,
    )

    H, W = g1.shape[:2]
    if roi is None:
        x1, y1, x2, y2 = 0, 0, W - 1, H - 1
    else:
        x1, y1, x2, y2 = (int(round(v)) for v in roi)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 3), dtype=np.float64)

    us, vs = np.meshgrid(np.arange(x1, x2 + 1, step), np.arange(y1, y2 + 1, step))
    us = us.ravel().astype(np.float64)
    vs = vs.ravel().astype(np.float64)
    fu = flow[vs.astype(int), us.astype(int), 0].astype(np.float64)
    fv = flow[vs.astype(int), us.astype(int), 1].astype(np.float64)

    # Reject near-zero flow (degenerate triangulation) and out-of-frame matches.
    mag = np.hypot(fu, fv)
    u2, v2 = us + fu, vs + fv
    ok = (mag >= min_flow_px) & (u2 >= 0) & (u2 <= W - 1) & (v2 >= 0) & (v2 <= H - 1)
    if ok.sum() < 8:
        return np.empty((0, 3), dtype=np.float64)
    p1 = np.stack([us[ok], vs[ok]])          # (2, N)
    p2 = np.stack([u2[ok], v2[ok]])          # (2, N)

    K = np.asarray(K, dtype=np.float64)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([np.asarray(R_rel, dtype=np.float64),
                        np.asarray(t_rel, dtype=np.float64).reshape(3, 1)])

    pts4 = cv2.triangulatePoints(P1, P2, p1, p2)
    w = pts4[3]
    good = np.abs(w) > 1e-9
    pts3 = (pts4[:3, good] / w[good]).T      # (N, 3) in the reference frame
    p1 = p1[:, good]
    p2 = p2[:, good]
    if len(pts3) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Keep points in front of both cameras.
    z1 = pts3[:, 2]
    pts_c2 = (np.asarray(R_rel) @ pts3.T).T + np.asarray(t_rel).reshape(1, 3)
    z2 = pts_c2[:, 2]
    front = (z1 > 1e-6) & (z2 > 1e-6)

    # Reprojection check in both views.
    def _proj(P, X):
        h = (P @ np.hstack([X, np.ones((len(X), 1))]).T).T
        return h[:, :2] / h[:, 2:3]

    with np.errstate(invalid="ignore", divide="ignore"):
        e1 = np.linalg.norm(_proj(P1, pts3) - p1.T, axis=1)
        e2 = np.linalg.norm(_proj(P2, pts3) - p2.T, axis=1)
    keep = front & np.isfinite(e1) & np.isfinite(e2) & \
        (e1 < max_reproj_err_px) & (e2 < max_reproj_err_px)

    out = pts3[keep]
    _logger.debug("dense stereo: %d/%d points kept", len(out), len(pts3))
    return out
