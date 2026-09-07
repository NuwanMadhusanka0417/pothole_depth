"""Quadratic surface fitting."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def fit_quadratic_surface(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Fit z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f.
    Returns coefficients and RMSE.
    """
    if points.shape[0] < 6:
        return np.zeros(6), float("inf")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    A = np.column_stack([x * x, y * y, x * y, x, y, np.ones_like(x)])
    reg = LinearRegression()
    reg.fit(A, z)
    pred = reg.predict(A)
    rmse = float(np.sqrt(np.mean((z - pred) ** 2)))
    return reg.coef_, rmse


def quadratic_height(coeff: np.ndarray, x: float, y: float) -> float:
    a, b, c, d, e, f = coeff
    return a * x * x + b * y * y + c * x * y + d * x + e * y + f
