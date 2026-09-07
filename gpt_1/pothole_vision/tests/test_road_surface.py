"""Tests for road surface fitting."""

import numpy as np

from pothole_vision.road_surface.ransac_plane import fit_plane_ransac
from pothole_vision.utils.math import point_to_plane_distance


def test_ransac_plane_flat_road():
    rng = np.random.default_rng(42)
    x = rng.uniform(-5, 5, 200)
    y = rng.uniform(-5, 5, 200)
    z = np.zeros_like(x) + rng.normal(0, 0.001, len(x))
    points = np.column_stack([x, y, z])
    plane, rmse = fit_plane_ransac(points, threshold=0.01)
    assert rmse < 0.01
    dists = point_to_plane_distance(points, plane)
    assert np.mean(dists) < 0.01


def test_synthetic_pothole_depth():
    """Pothole 5cm below flat road z=0."""
    plane = np.array([0, 0, 1, 0])
    pothole_points = np.array([
        [0, 0, -0.05],
        [0.1, 0, -0.05],
        [0, 0.1, -0.048],
    ])
    dists = point_to_plane_distance(pothole_points, plane)
    assert abs(np.median(dists) - 0.05) < 0.002
