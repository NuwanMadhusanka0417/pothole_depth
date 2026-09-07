"""Pothole dimension estimation."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def compute_dimensions_2d(boundary_points: np.ndarray, pixel_size_m: float) -> dict[str, float]:
    """PCA-based length/width on road-plane projected boundary."""
    if len(boundary_points) < 3:
        return {"length_m": float("nan"), "width_m": float("nan"), "area_m2": float("nan")}

    pts = boundary_points[:, :2] if boundary_points.shape[1] >= 2 else boundary_points
    pca = PCA(n_components=2)
    pca.fit(pts)
    projected = pca.transform(pts)
    length_px = projected[:, 0].max() - projected[:, 0].min()
    width_px = projected[:, 1].max() - projected[:, 1].min()

    length_m = abs(length_px) * pixel_size_m
    width_m = abs(width_px) * pixel_size_m

    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        area_px2 = hull.volume  # in 2D, volume = area
        area_m2 = area_px2 * pixel_size_m ** 2
    except Exception:
        area_m2 = float("nan")

    return {"length_m": length_m, "width_m": width_m, "area_m2": area_m2}
