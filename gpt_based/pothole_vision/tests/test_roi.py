"""Tests for ROI mask and coordinate conversion."""

import numpy as np

from pothole_vision.roi.fixed_roi import FixedRoadROI, ROIConfig
from pothole_vision.roi.masks import polygon_to_mask
from pothole_vision.utils.math import normalized_to_pixel, pixel_to_normalized


def test_normalized_to_pixel_roundtrip():
    w, h = 1920, 1080
    points = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
    px = normalized_to_pixel(points, w, h)
    back = pixel_to_normalized(px, w, h)
    np.testing.assert_allclose(back, points, atol=1e-6)


def test_polygon_mask_area():
    w, h = 100, 100
    poly = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float)
    mask = polygon_to_mask(poly, h, w)
    assert mask.sum() == 81 * 81  # inclusive pixel boundaries


def test_fixed_roi_static_mask():
    cfg = ROIConfig(road_polygon=[[0.1, 0.9], [0.9, 0.9], [0.5, 0.6]])
    roi = FixedRoadROI(cfg)
    mask = roi.get_static_mask(640, 480)
    assert mask.shape == (480, 640)
    assert mask.dtype == bool
    assert mask.sum() > 0
