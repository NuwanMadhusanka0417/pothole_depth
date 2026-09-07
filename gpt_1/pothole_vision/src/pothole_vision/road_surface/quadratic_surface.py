"""Quadratic road surface fitting."""

from __future__ import annotations

import numpy as np


def fit_quadratic_surface(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f. Returns (coeffs, rmse)."""
    if len(points) < 10:
        return np.zeros(6), float("inf")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    A = np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    pred = A @ coeffs
    rmse = float(np.sqrt(np.mean((z - pred) ** 2)))
    return coeffs, rmse


def eval_quadratic(coeffs: np.ndarray, points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    a, b, c, d, e, f = coeffs
    return a * x * x + b * y * y + c * x * y + d * x + e * y + f
