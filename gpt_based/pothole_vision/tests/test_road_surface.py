"""Tests for road surface fitting."""

import numpy as np

from pothole_vision.road_surface.ransac_plane import point_to_plane_distance


def test_synthetic_pothole_depth():
    """Road at z=0, pothole points at z=-0.05 m."""
    plane = np.array([0, 0, 1, 0], dtype=float)
    pothole_pts = np.array([[0, 0, -0.05], [0.1, 0, -0.05], [-0.1, 0.1, -0.048]])
    depths = point_to_plane_distance(pothole_pts, plane)
    assert np.median(depths) == 0.05
    assert abs(np.median(depths) - 0.05) < 0.002
