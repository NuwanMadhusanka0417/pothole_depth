"""Tests for geometry utilities."""

import numpy as np

from pothole_vision.geometry.coordinates import backproject
from pothole_vision.road_surface.ransac_plane import fit_plane_ransac, point_to_plane_distance


def test_backproject():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    pt = backproject(320, 240, 5.0, K)
    np.testing.assert_allclose(pt, [0, 0, 5], atol=1e-4)


def test_plane_distance():
    # z = 0 plane
    plane = np.array([0, 0, 1, 0], dtype=float)
    points = np.array([[0, 0, -0.05], [1, 1, -0.05]])
    dist = point_to_plane_distance(points, plane)
    np.testing.assert_allclose(dist, [0.05, 0.05], atol=1e-6)


def test_ransac_plane_synthetic():
    rng = np.random.default_rng(42)
    x = rng.uniform(-2, 2, 500)
    y = rng.uniform(-2, 2, 500)
    z = 0.01 * x + 0.02 * y + rng.normal(0, 0.001, 500)
    pts = np.column_stack([x, y, z])
    plane, rmse, inliers = fit_plane_ransac(pts, threshold=0.01)
    assert rmse < 0.01
    assert inliers.sum() > 400
