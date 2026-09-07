"""Tests for tracking association."""

import numpy as np

from pothole_vision.tracking.association import bbox_iou, mask_iou


def test_mask_iou_identical():
    m = np.zeros((100, 100), dtype=bool)
    m[20:60, 20:60] = True
    assert mask_iou(m, m) == 1.0


def test_mask_iou_disjoint():
    a = np.zeros((50, 50), dtype=bool)
    b = np.zeros((50, 50), dtype=bool)
    a[:25, :] = True
    b[25:, :] = True
    assert mask_iou(a, b) == 0.0


def test_bbox_iou():
    box_a = (0, 0, 100, 100)
    box_b = (50, 50, 150, 150)
    iou = bbox_iou(box_a, box_b)
    assert 0.0 < iou < 1.0
