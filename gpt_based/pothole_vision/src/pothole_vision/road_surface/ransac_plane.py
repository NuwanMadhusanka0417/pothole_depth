"""RANSAC plane fitting."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RANSACRegressor


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.02,
    min_samples: int = 3,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Fit ax + by + cz + d = 0 normalized so ||a,b,c||=1.
    Returns (normal_and_d as [a,b,c,d], rmse, inlier_mask).
    """
    if points.shape[0] < min_samples:
        return np.array([0, 0, 1, 0], dtype=np.float64), float("inf"), np.zeros(points.shape[0], dtype=bool)

    X = points[:, :2]
    y = points[:, 2]
    ransac = RANSACRegressor(min_samples=min_samples, residual_threshold=threshold)
    ransac.fit(X, y)
    inliers = ransac.inlier_mask_

    a = -ransac.estimator_.coef_[0]
    b = -ransac.estimator_.coef_[1]
    c = 1.0
    d = -ransac.estimator_.intercept_
    norm = np.sqrt(a * a + b * b + c * c)
    plane = np.array([a, b, c, d]) / norm

    residuals = point_to_plane_distance(points, plane)
    rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2))) if inliers.any() else float("inf")
    return plane, rmse, inliers


def point_to_plane_distance(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    a, b, c, d = plane
    return np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(
        a * a + b * b + c * c
    )
