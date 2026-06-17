"""Tests for data-loader / mask-bbox conversion utilities."""

from __future__ import annotations

import numpy as np

from sfm_yolo.src.utils.data_loader import bboxes_to_yolo, mask_to_bboxes
from sfm_yolo.src.utils.evaluation_metrics import (
    compute_depth_metrics,
    iou_xyxy,
    precision_recall_at_iou,
)


def test_mask_to_bboxes_simple() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 20:40] = 255   # blob 1
    mask[60:80, 70:90] = 255   # blob 2
    boxes = mask_to_bboxes(mask, min_pixels=10)
    assert len(boxes) == 2
    for x1, y1, x2, y2 in boxes:
        assert x2 > x1 and y2 > y1


def test_mask_to_bboxes_min_pixels_filters() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:2, 0:2] = 255  # 4 pixels
    mask[50:80, 50:80] = 255
    boxes = mask_to_bboxes(mask, min_pixels=20)
    assert len(boxes) == 1


def test_bboxes_to_yolo_normalised() -> None:
    lines = bboxes_to_yolo([(10, 20, 30, 40)], image_shape=(100, 100), class_id=0)
    assert len(lines) == 1
    cls, cx, cy, w, h = lines[0].split()
    assert int(cls) == 0
    assert abs(float(cx) - 0.20) < 1e-6
    assert abs(float(cy) - 0.30) < 1e-6
    assert abs(float(w) - 0.20) < 1e-6
    assert abs(float(h) - 0.20) < 1e-6


def test_iou_xyxy_disjoint_zero() -> None:
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_xyxy_identical_one() -> None:
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_precision_recall_basic() -> None:
    res = precision_recall_at_iou(
        pred_boxes=[(0, 0, 10, 10), (50, 50, 60, 60)],
        gt_boxes=[(1, 1, 9, 9)],
        iou_threshold=0.4,
    )
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["fn"] == 0


def test_compute_depth_metrics_perfect() -> None:
    pred = [0.05, 0.10, 0.20]
    truth = pred
    metrics = compute_depth_metrics(pred, truth)
    assert metrics.mae == 0.0
    assert metrics.within_5pct == 1.0
    assert metrics.within_10pct == 1.0


def test_compute_depth_metrics_known_offset() -> None:
    pred = [0.10, 0.20, 0.30]
    truth = [0.11, 0.22, 0.33]
    m = compute_depth_metrics(pred, truth)
    assert 0.01 < m.mae < 0.05
    assert m.bias < 0   # systematic underestimate
