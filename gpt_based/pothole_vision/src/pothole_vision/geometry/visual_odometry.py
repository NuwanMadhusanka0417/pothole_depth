"""Visual odometry from feature matches."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pothole_vision.geometry.feature_matching import detect_and_match


@dataclass
class PoseEstimate:
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
    K: np.ndarray,
    mask: np.ndarray | None = None,
    config: dict | None = None,
) -> PoseEstimate:
    cfg = config or {}
    min_matches = int(cfg.get("min_feature_matches", 30))
    min_inlier_ratio = float(cfg.get("min_inlier_ratio", 0.4))
    ransac_thresh = float(cfg.get("ransac_threshold", 1.0))

    pts1, pts2, _ = detect_and_match(
        img1, img2, mask,
        detector=cfg.get("feature_detector", "orb"),
        max_features=int(cfg.get("max_features", 2000)),
    )

    invalid = PoseEstimate(
        R=np.eye(3), t=np.zeros(3),
        inlier_count=0, inlier_ratio=0.0,
        mean_reprojection_error=999.0, parallax_deg=0.0, valid=False,
    )

    if len(pts1) < min_matches:
        return invalid

    E, inliers = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
    if E is None or inliers is None:
        return invalid

    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(pts1)

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=inliers)

    inlier_pts1 = pts1[inliers.ravel() == 1]
    inlier_pts2 = pts2[inliers.ravel() == 1]
    parallax = float(np.mean(np.linalg.norm(inlier_pts2 - inlier_pts1, axis=1))) if len(inlier_pts1) else 0.0
    parallax_deg = np.degrees(np.arctan2(parallax, max(K[0, 0], 1)))

    valid = inlier_count >= min_matches and inlier_ratio >= min_inlier_ratio

    return PoseEstimate(
        R=R, t=t.ravel(),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        mean_reprojection_error=0.0,
        parallax_deg=parallax_deg,
        valid=valid,
    )
