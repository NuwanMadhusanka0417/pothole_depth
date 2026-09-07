"""Point cloud utilities."""

from __future__ import annotations

import numpy as np


def filter_points(points: np.ndarray, z_min: float = 0.1, z_max: float = 100.0) -> np.ndarray:
    if points.size == 0:
        return points
    valid = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    return points[valid]
