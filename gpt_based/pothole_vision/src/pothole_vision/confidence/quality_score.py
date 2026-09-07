"""Quality score aggregation."""

from __future__ import annotations


def compute_quality_score(sharpness: float, inlier_ratio: float, point_count: int) -> float:
    s1 = min(sharpness / 200.0, 1.0)
    s2 = min(inlier_ratio, 1.0)
    s3 = min(point_count / 500.0, 1.0)
    return 0.3 * s1 + 0.4 * s2 + 0.3 * s3
