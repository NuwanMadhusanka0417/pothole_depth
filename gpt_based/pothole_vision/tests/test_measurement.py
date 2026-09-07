"""Tests for measurement math."""

import numpy as np

from pothole_vision.confidence.uncertainty import multi_frame_consensus
from pothole_vision.measurement.depth_measurement import compute_depth_stats
from pothole_vision.utils.math import mad_outlier_mask, robust_median_with_mad


def test_robust_median_mad():
    values = np.array([5.9, 6.1, 6.0, 14.7, 6.2, 5.8])
    med, disp, n = robust_median_with_mad(values)
    assert abs(med - 6.0) < 0.2
    assert n < len(values)


def test_mad_outlier_rejection():
    values = np.array([1.0, 1.1, 1.0, 10.0, 0.9])
    inliers = mad_outlier_mask(values)
    assert not inliers[3]


def test_depth_percentile():
    depths = np.linspace(0.01, 0.10, 100)
    stats = compute_depth_stats(depths)
    assert stats["maximum_depth_m"] is not None
    assert stats["maximum_depth_m"] < stats["raw_max_m"] or abs(stats["maximum_depth_m"] - stats["raw_max_m"]) < 0.02


def test_multi_frame_consensus():
    med, unc = multi_frame_consensus([5.9, 6.1, 6.0, 14.7, 6.2, 5.8])
    assert abs(med - 6.0) < 0.3
