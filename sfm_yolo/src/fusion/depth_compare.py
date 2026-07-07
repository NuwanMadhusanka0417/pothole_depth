"""Agreement metrics for validating Path A (physics/SfM) against Path B (learned).

Both paths produce a metric pothole depth per tracked pothole. Because they
share the same below-plane depth core (:mod:`..geometry.plane_depth`), a
disagreement isolates the 3D *source* (SfM vs learned depth), which is exactly
what we want to test. Use :func:`compare_pair` per pothole and
:func:`aggregate` over a dataset. Optionally anchor both against ruler ground
truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class DepthAgreement:
    depth_a_m: float
    depth_b_m: float
    abs_diff_m: float
    rel_gap: float          # |a-b| / mean(a,b)
    agree: bool             # within tolerance

    def as_dict(self) -> dict:
        return {
            "depth_a_m": self.depth_a_m,
            "depth_b_m": self.depth_b_m,
            "abs_diff_m": self.abs_diff_m,
            "rel_gap": self.rel_gap,
            "agree": self.agree,
        }


def compare_pair(
    depth_a: float,
    depth_b: float,
    *,
    abs_tol_m: float = 0.01,
    rel_tol: float = 0.15,
) -> DepthAgreement:
    """Compare one Path A vs Path B depth. Agreement if within abs OR rel tol."""
    a, b = float(depth_a), float(depth_b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return DepthAgreement(a, b, float("nan"), float("nan"), False)
    diff = abs(a - b)
    mean = max(1e-6, 0.5 * (a + b))
    rel = diff / mean
    agree = (diff <= abs_tol_m) or (rel <= rel_tol)
    return DepthAgreement(a, b, diff, rel, bool(agree))


def aggregate(
    depths_a: Sequence[float],
    depths_b: Sequence[float],
    *,
    ground_truth: Optional[Sequence[float]] = None,
) -> dict:
    """Dataset-level A-vs-B metrics (MAE, RMSE, bias, correlation).

    If ``ground_truth`` is given, also reports A-vs-GT and B-vs-GT MAE so the
    A/B agreement is calibrated against truth (agreement != correctness).
    """
    a = np.asarray(depths_a, dtype=np.float64)
    b = np.asarray(depths_b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    out: dict = {"n": int(a.size)}
    if a.size == 0:
        return out
    out["mae_ab_m"] = float(np.mean(np.abs(a - b)))
    out["rmse_ab_m"] = float(np.sqrt(np.mean((a - b) ** 2)))
    out["bias_ab_m"] = float(np.mean(a - b))
    if a.size >= 2 and np.std(a) > 0 and np.std(b) > 0:
        out["pearson_ab"] = float(np.corrcoef(a, b)[0, 1])
    if ground_truth is not None:
        gt = np.asarray(ground_truth, dtype=np.float64)[m]
        gm = np.isfinite(gt)
        if gm.any():
            out["mae_a_gt_m"] = float(np.mean(np.abs(a[gm] - gt[gm])))
            out["mae_b_gt_m"] = float(np.mean(np.abs(b[gm] - gt[gm])))
    return out
