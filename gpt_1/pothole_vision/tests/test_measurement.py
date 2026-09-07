"""Tests for measurement mathematics."""

import numpy as np

from pothole_vision.depth.temporal_fusion import fuse_depth_observations
from pothole_vision.measurement.depth_measurement import compute_depth_metrics
from pothole_vision.utils.math import mad_outlier_mask, percentile_robust_max, robust_median


def test_robust_median():
    values = np.array([5.9, 6.1, 6.0, 14.7, 6.2, 5.8])
    mask = mad_outlier_mask(values)
    filtered = values[mask]
    med = robust_median(filtered)
    assert 5.8 <= med <= 6.2


def test_fuse_depth_observations():
    obs = [5.9, 6.1, 6.0, 14.7, 6.2, 5.8]
    fused, unc = fuse_depth_observations(obs)
    assert 5.8 <= fused <= 6.2


def test_depth_percentile():
    depths = np.linspace(0.01, 0.10, 100)
    robust_max = percentile_robust_max(depths, percentile=97.5, region_fraction=0.05)
    assert robust_max >= 0.09
    assert robust_max <= 0.10


def test_compute_depth_metrics():
    depths = np.full(50, 0.05) + np.random.default_rng(0).normal(0, 0.002, 50)
    metrics = compute_depth_metrics(depths)
    assert abs(metrics["median_depth_m"] - 0.05) < 0.01
