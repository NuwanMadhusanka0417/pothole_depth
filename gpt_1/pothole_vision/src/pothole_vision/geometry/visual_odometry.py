"""Visual odometry via essential matrix."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pothole_vision.geometry.feature_matching import detect_and_match
from pothole_vision.geometry.intrinsics import CameraIntrinsics


@dataclass
class RelativePose:
    R: np.ndarray
    t: np.ndarray
    inlier_count: int
    inlier_ratio: float
    mean_reprojection_error: float
    parallax_deg: float
    valid: bool


def estimate_relative_pose(
    img1: np.ndarray,
    img2: np.ndarray,
    intrinsics: CameraIntrinsics,
    mask1: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
    config: dict | None = None,
) -> RelativePose:
    cfg = config or {}
    min_matches = cfg.get("min_feature_matches", 30)
    min_inlier_ratio = cfg.get("min_inlier_ratio", 0.35)
    min_parallax = cfg.get("min_parallax_deg", 0.5)
    ransac_thresh = cfg.get("ransac_threshold", 1.0)
    detector = cfg.get("feature_detector", "orb")

    pts1, pts2, matches = detect_and_match(
        img1, img2, mask1, mask2, detector_type=detector,
        max_features=cfg.get("max_features", 2000),
    )

    invalid = RelativePose(
        R=np.eye(3), t=np.zeros(3), inlier_count=0, inlier_ratio=0.0,
        mean_reprojection_error=float("inf"), parallax_deg=0.0, valid=False,
    )

    if len(matches) < min_matches:
        return invalid

    E, inliers = cv2.findEssentialMat(
        pts1, pts2, intrinsics.K, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh,
    )
    if E is None or inliers is None:
        return invalid

    inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / len(matches)

    if inlier_ratio < min_inlier_ratio:
        return RelativePose(
            R=np.eye(3), t=np.zeros(3), inlier_count=inlier_count,
            inlier_ratio=inlier_ratio, mean_reprojection_error=float("inf"),
            parallax_deg=0.0, valid=False,
        )

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, intrinsics.K, mask=inliers)

    inlier_pts1 = pts1[inlier_mask]
    inlier_pts2 = pts2[inlier_mask]
    parallax = _estimate_parallax_deg(inlier_pts1, inlier_pts2, intrinsics)

    valid = parallax >= min_parallax

    return RelativePose(
        R=R,
        t=t.ravel(),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        mean_reprojection_error=0.0,
        parallax_deg=parallax,
        valid=valid,
    )


def _estimate_parallax_deg(pts1: np.ndarray, pts2: np.ndarray, intrinsics: CameraIntrinsics) -> float:
    if len(pts1) < 2:
        return 0.0
    disp = np.linalg.norm(pts1 - pts2, axis=1)
    mean_disp_px = float(np.mean(disp))
    f = (intrinsics.fx + intrinsics.fy) / 2
    return float(np.degrees(np.arctan(mean_disp_px / f)))
