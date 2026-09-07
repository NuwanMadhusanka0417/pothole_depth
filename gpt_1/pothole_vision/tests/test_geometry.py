"""Tests for geometry utilities."""

import numpy as np

from pothole_vision.geometry.coordinates import remap_bbox_to_original
from pothole_vision.utils.math import point_to_plane_distance


def test_remap_bbox():
    bbox = remap_bbox_to_original((100, 100, 200, 200), 1920, 1080, 640, 640)
    assert bbox[2] > bbox[0]
    assert bbox[3] > bbox[1]


def test_point_to_plane_distance():
    plane = np.array([0, 0, 1, 0])  # z = 0
    points = np.array([[0, 0, 0.05], [0, 0, 0.10]])
    dists = point_to_plane_distance(points, plane)
    np.testing.assert_allclose(dists, [0.05, 0.10], rtol=1e-3)
