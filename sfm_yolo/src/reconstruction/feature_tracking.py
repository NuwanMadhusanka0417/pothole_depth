"""Lightweight feature detection / matching / optical-flow utilities.

These helpers are used in two places:

  * the OpenCV fall-back path of :class:`SfMRunner` when COLMAP isn't
    available,
  * the scale-recovery step, where we need to match a 2D pixel from
    the geometric model to the closest projected 3D SfM point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FrameFeatures:
    """Keypoints + descriptors for a single frame."""

    keypoints: np.ndarray   # (N, 2) float32 pixel coords
    descriptors: np.ndarray # (N, D) float32 / uint8 depending on detector
    image_shape: Tuple[int, int]   # (H, W)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_features(
    image: np.ndarray,
    *,
    detector: str = "sift",
    max_features: int = 4000,
) -> FrameFeatures:
    """Detect keypoints and compute descriptors.

    ``detector`` may be ``"sift"`` (default, more robust) or ``"orb"``
    (faster, no contrib package required).
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    if detector.lower() == "sift":
        try:
            algo = cv2.SIFT_create(nfeatures=max_features)
        except AttributeError as exc:  # pragma: no cover
            raise RuntimeError(
                "SIFT not available - install opencv-contrib-python or pass detector='orb'."
            ) from exc
    elif detector.lower() == "orb":
        algo = cv2.ORB_create(nfeatures=max_features)
    else:
        raise ValueError(f"unknown detector {detector!r}")

    kps, desc = algo.detectAndCompute(gray, None)
    if kps is None or desc is None or len(kps) == 0:
        return FrameFeatures(
            keypoints=np.zeros((0, 2), dtype=np.float32),
            descriptors=np.zeros((0, 0), dtype=np.float32),
            image_shape=gray.shape[:2],
        )
    pts = np.array([[kp.pt[0], kp.pt[1]] for kp in kps], dtype=np.float32)
    return FrameFeatures(keypoints=pts, descriptors=desc, image_shape=gray.shape[:2])


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def match_features(
    feats_a: FrameFeatures,
    feats_b: FrameFeatures,
    *,
    ratio: float = 0.75,
    use_flann: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lowe's ratio-test matcher returning matched pixel pairs.

    Returns
    -------
    pts_a : ndarray (K, 2)
    pts_b : ndarray (K, 2)
    """
    if feats_a.descriptors.size == 0 or feats_b.descriptors.size == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty

    is_float = feats_a.descriptors.dtype != np.uint8
    if use_flann and is_float:
        index_params = dict(algorithm=1, trees=5)   # KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        norm = cv2.NORM_L2 if is_float else cv2.NORM_HAMMING
        matcher = cv2.BFMatcher(norm)

    raw = matcher.knnMatch(feats_a.descriptors, feats_b.descriptors, k=2)
    good_a: List[np.ndarray] = []
    good_b: List[np.ndarray] = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_a.append(feats_a.keypoints[m.queryIdx])
            good_b.append(feats_b.keypoints[m.trainIdx])
    if not good_a:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty
    return np.asarray(good_a, dtype=np.float32), np.asarray(good_b, dtype=np.float32)


# ---------------------------------------------------------------------------
# Optical-flow tracking
# ---------------------------------------------------------------------------
def track_points_lk(
    image_a: np.ndarray,
    image_b: np.ndarray,
    points_a: np.ndarray,
    *,
    window: Tuple[int, int] = (21, 21),
    max_level: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lucas-Kanade tracking of ``points_a`` from frame A to B.

    Returns ``(points_b, status)`` where ``status[i] == 1`` means the
    track succeeded.
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY) if image_a.ndim == 3 else image_a
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY) if image_b.ndim == 3 else image_b
    pts = points_a.reshape(-1, 1, 2).astype(np.float32)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, pts, None, winSize=window, maxLevel=max_level
    )
    if next_pts is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.uint8)
    return next_pts.reshape(-1, 2), status.reshape(-1)


# ---------------------------------------------------------------------------
# Two-view triangulation (used by the OpenCV SfM fallback)
# ---------------------------------------------------------------------------
def triangulate_two_views(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    K: np.ndarray,
    *,
    ransac_threshold: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Recover relative pose and triangulate 3D points from two views.

    Parameters
    ----------
    pts_a, pts_b : ndarray (N, 2)
        Matched pixel coordinates.
    K : ndarray (3, 3)
        Camera intrinsic matrix.
    ransac_threshold : float
        RANSAC reprojection threshold (pixels).

    Returns
    -------
    points_3d : ndarray (M, 3) or None
        Triangulated points in camera-1 coordinates, with arbitrary
        (unitless) scale.
    R, t : ndarrays
        Relative rotation / translation of camera 2 with respect to
        camera 1.
    """
    if pts_a.shape != pts_b.shape or pts_a.shape[0] < 8:
        return None, None, None

    E, mask = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_threshold
    )
    if E is None:
        return None, None, None

    pts_a_in = pts_a[mask.ravel().astype(bool)]
    pts_b_in = pts_b[mask.ravel().astype(bool)]
    if len(pts_a_in) < 5:
        return None, None, None

    _, R, t, pose_mask = cv2.recoverPose(E, pts_a_in, pts_b_in, K)

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])

    pose_inliers = pose_mask.ravel().astype(bool)
    pts_a_final = pts_a_in[pose_inliers]
    pts_b_final = pts_b_in[pose_inliers]
    if len(pts_a_final) < 4:
        return None, R, t

    pts4 = cv2.triangulatePoints(P1, P2, pts_a_final.T, pts_b_final.T)
    pts3 = (pts4[:3] / pts4[3:4]).T
    return pts3, R, t


def project_points(
    points_3d: np.ndarray,
    *,
    K: np.ndarray,
    R: np.ndarray | None = None,
    t: np.ndarray | None = None,
) -> np.ndarray:
    """Project 3D points to pixel coordinates."""
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)
    rvec, _ = cv2.Rodrigues(R)
    pts2, _ = cv2.projectPoints(points_3d.reshape(-1, 1, 3), rvec, t.reshape(3), K, None)
    return pts2.reshape(-1, 2)


def nearest_3d_to_pixel(
    points_3d: np.ndarray,
    target_pixel: Sequence[float],
    *,
    K: np.ndarray,
    R: np.ndarray | None = None,
    t: np.ndarray | None = None,
    max_pixel_distance: float = 30.0,
) -> Optional[Tuple[int, np.ndarray, float]]:
    """Find the 3D point whose projection is closest to ``target_pixel``.

    Returns ``(index, point_xyz, pixel_distance)`` or ``None`` if no
    point is within ``max_pixel_distance``.
    """
    if points_3d.size == 0:
        return None
    proj = project_points(points_3d, K=K, R=R, t=t)
    diffs = proj - np.asarray(target_pixel, dtype=np.float32).reshape(1, 2)
    d = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(d))
    if d[idx] > max_pixel_distance:
        return None
    return idx, points_3d[idx], float(d[idx])
