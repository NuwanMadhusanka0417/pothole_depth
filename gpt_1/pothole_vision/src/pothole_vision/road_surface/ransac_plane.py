"""RANSAC plane fitting for local road surface."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RANSACRegressor


def fit_plane_ransac(points: np.ndarray, threshold: float = 0.02) -> tuple[np.ndarray, float]:
    """Fit ax+by+cz+d=0 via RANSAC. Returns (plane_coeffs [a,b,c,d], rmse)."""
    if len(points) < 4:
        return np.array([0, 0, 1, 0], dtype=np.float64), float("inf")

    X = points[:, :2]
    y = points[:, 2]

    ransac = RANSACRegressor(residual_threshold=threshold, random_state=42)
    try:
        ransac.fit(X, y)
    except ValueError:
        return np.array([0, 0, 1, 0], dtype=np.float64), float("inf")

    a = -ransac.estimator_.coef_[0]
    b = -ransac.estimator_.coef_[1]
    d = -ransac.estimator_.intercept_
    c = 1.0
    norm = np.sqrt(a * a + b * b + c * c)
    plane = np.array([a, b, c, d]) / norm

    residuals = np.abs(points @ plane[:3] + plane[3])
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return plane, rmse
