"""Tests for ROI normalized coordinate conversion."""

import numpy as np

from pothole_vision.geometry.coordinates import normalized_to_pixel, pixel_to_normalized
from pothole_vision.roi.fixed_roi import FixedRoadROI, ROIConfig


def test_normalized_to_pixel():
    pts = [(0.5, 0.5), (0.0, 1.0)]
    px = normalized_to_pixel(pts, 1920, 1080)
    assert px[0, 0] == 960
    assert px[0, 1] == 540
    assert px[1, 0] == 0
    assert px[1, 1] == 1080


def test_pixel_roundtrip():
    original = [(0.1, 0.2), (0.9, 0.8)]
    px = normalized_to_pixel(original, 1000, 500)
    back = pixel_to_normalized(px, 1000, 500)
    for (ox, oy), (bx, by) in zip(original, back):
        assert abs(ox - bx) < 0.01
        assert abs(oy - by) < 0.01


def test_roi_mask_shape():
    config = ROIConfig(road_polygon=[(0.1, 0.7), (0.9, 0.7), (0.9, 0.95), (0.1, 0.95)])
    roi = FixedRoadROI(config)
    mask = roi.create_mask(640, 480)
    assert mask.shape == (480, 640)
    assert mask.sum() > 0
