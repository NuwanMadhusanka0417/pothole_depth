"""Evaluation metrics for depth estimation and detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Depth metrics
# ---------------------------------------------------------------------------
@dataclass
class DepthMetrics:
    n: int
    mae: float        # mean absolute error (m)
    rmse: float       # root mean squared error (m)
    mape: float       # mean absolute percentage error (%)
    median_err: float # median absolute error (m)
    bias: float       # mean signed error (pred - truth)
    within_5pct: float
    within_10pct: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "MAE_m": self.mae,
            "RMSE_m": self.rmse,
            "MAPE_pct": self.mape,
            "median_error_m": self.median_err,
            "bias_m": self.bias,
            "within_5pct": self.within_5pct,
            "within_10pct": self.within_10pct,
        }


def compute_depth_metrics(
    pred: Sequence[float] | np.ndarray,
    truth: Sequence[float] | np.ndarray,
    *,
    eps: float = 1e-6,
) -> DepthMetrics:
    """Compute standard depth-error statistics."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {truth.shape}")
    if pred.size == 0:
        raise ValueError("empty inputs")

    err = pred - truth
    abs_err = np.abs(err)
    rel_err = abs_err / np.maximum(np.abs(truth), eps)

    return DepthMetrics(
        n=int(pred.size),
        mae=float(abs_err.mean()),
        rmse=float(np.sqrt((err ** 2).mean())),
        mape=float((rel_err * 100.0).mean()),
        median_err=float(np.median(abs_err)),
        bias=float(err.mean()),
        within_5pct=float((rel_err <= 0.05).mean()),
        within_10pct=float((rel_err <= 0.10).mean()),
    )


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------
def iou_xyxy(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-Union for two ``(x1,y1,x2,y2)`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def precision_recall_at_iou(
    pred_boxes: Iterable[Sequence[float]],
    gt_boxes: Iterable[Sequence[float]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute single-image precision/recall at a given IoU threshold."""
    preds = list(pred_boxes)
    gts = list(gt_boxes)
    matched_gt: set[int] = set()
    tp = 0
    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched_gt:
                continue
            iou = iou_xyxy(p, g)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            matched_gt.add(best_j)
            tp += 1
    fp = max(0, len(preds) - tp)
    fn = max(0, len(gts) - tp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
    }
