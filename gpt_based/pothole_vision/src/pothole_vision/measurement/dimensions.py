"""Pothole dimension calculation."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def compute_dimensions_2d(points_xy: np.ndarray) -> tuple[float, float, float]:
    """Return length, width, area from 2D boundary points using PCA."""
    if points_xy.shape[0] < 3:
        return 0.0, 0.0, 0.0
    pca = PCA(n_components=2)
    pca.fit(points_xy)
    projected = pca.transform(points_xy)
    length = float(projected[:, 0].max() - projected[:, 0].min())
    width = float(projected[:, 1].max() - projected[:, 1].min())
    area = length * width * 0.785  # approximate ellipse
    return length, width, area
