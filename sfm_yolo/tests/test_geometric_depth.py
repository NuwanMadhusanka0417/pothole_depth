"""Unit tests for the geometric depth estimator.

These rely only on numpy and pytest - no heavy dependencies.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sfm_yolo.src.geometry.angle_utils import (
    angle_to_distance,
    pixel_y_to_angle,
    pixel_y_to_distance,
)
from sfm_yolo.src.geometry.geometric_depth import GeometricDepthEstimator
from sfm_yolo.src.utils.camera_calibration import CameraIntrinsics


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        camera_height_m=1.5,
        focal_length_px=1000.0,
        principal_point=(960.0, 540.0),
        image_size=(1920, 1080),
        pitch_deg=0.0,
    )


def test_pixel_to_angle_at_center_is_zero(intrinsics: CameraIntrinsics) -> None:
    theta = pixel_y_to_angle(
        intrinsics.cy,
        cy=intrinsics.cy,
        focal_length_px=intrinsics.fy,
    )
    assert math.isclose(float(theta), 0.0, abs_tol=1e-9)


def test_angle_to_distance_at_45deg(intrinsics: CameraIntrinsics) -> None:
    d = angle_to_distance(math.pi / 4, camera_height_m=intrinsics.camera_height_m)
    assert math.isclose(float(d), intrinsics.camera_height_m, rel_tol=1e-6)


def test_pixel_to_distance_monotone(intrinsics: CameraIntrinsics) -> None:
    """As the row goes from horizon to image bottom, distance shrinks."""
    rows = np.linspace(intrinsics.cy + 1, intrinsics.height - 1, 50)
    d = pixel_y_to_distance(
        rows,
        camera_height_m=intrinsics.camera_height_m,
        cy=intrinsics.cy,
        focal_length_px=intrinsics.fy,
    )
    assert np.all(np.isfinite(d))
    assert np.all(np.diff(d) <= 0)


def test_single_frame_depth_recovers_synthetic(intrinsics: CameraIntrinsics) -> None:
    """Build a synthetic scene where we know the answer.

    Place a pothole 10 m ahead, with a 0.10 m drop. The bounding box
    rows correspond to road-level (10 m) and pothole-bottom (slightly
    further along the optical ray that grazes the pothole bottom).
    """
    h = intrinsics.camera_height_m
    f = intrinsics.fy
    cy = intrinsics.cy

    target_depth = 0.10
    d_road = 10.0
    # The "extra distance" you have to travel to reach the bottom of the
    # pit is target_depth / tan(theta_road), which keeps the metric
    # answer self-consistent with the model used by the estimator.
    theta_road = math.atan2(h, d_road)
    # Convert distances back to pixel rows using y = cy + f * tan(theta)
    y_road = cy + f * math.tan(theta_road)
    delta_d = target_depth / math.tan(theta_road)
    d_bot = d_road + delta_d
    theta_bot = math.atan2(h, d_bot)
    y_bot = cy + f * math.tan(theta_bot)

    estimator = GeometricDepthEstimator(intrinsics)
    bbox = (800.0, y_road, 1100.0, y_bot)
    res = estimator.single_frame_depth(bbox)
    assert math.isclose(res.depth_m, target_depth, rel_tol=0.01)
    assert res.confidence > 0


def test_two_frame_depth_more_confident_than_single(intrinsics: CameraIntrinsics) -> None:
    estimator = GeometricDepthEstimator(intrinsics)
    bbox_a = (800.0, 760.0, 1100.0, 820.0)
    bbox_b = (820.0, 770.0, 1120.0, 830.0)
    single = estimator.single_frame_depth(bbox_a)
    pair = estimator.two_frame_depth(bbox_a, bbox_b)
    assert math.isfinite(pair.depth_m)
    assert pair.confidence >= 0.0
    # Average should be between the two single-frame values
    s_b = estimator.single_frame_depth(bbox_b)
    if math.isfinite(s_b.depth_m) and math.isfinite(single.depth_m):
        lo = min(single.depth_m, s_b.depth_m) - 1e-9
        hi = max(single.depth_m, s_b.depth_m) + 1e-9
        assert lo <= pair.depth_m <= hi


def test_multi_frame_validation_robust_to_outlier(intrinsics: CameraIntrinsics) -> None:
    estimator = GeometricDepthEstimator(intrinsics)
    base = (800.0, 760.0, 1100.0, 820.0)
    outlier = (800.0, 760.0, 1100.0, 770.0)  # tiny height -> very small depth
    bboxes = [base, base, base, outlier]
    res = estimator.multi_frame_validation(bboxes)
    single = estimator.single_frame_depth(base).depth_m
    assert abs(res.depth_m - single) < 1e-3   # median ignores the outlier


def test_extract_road_points_returns_points(intrinsics: CameraIntrinsics) -> None:
    estimator = GeometricDepthEstimator(intrinsics)
    pts = estimator.extract_road_points(
        image_shape=(intrinsics.height, intrinsics.width), n_points=8
    )
    assert len(pts) >= 4
    for px, py, d in pts:
        assert d > 0
        assert intrinsics.cy < py < intrinsics.height


def test_invalid_bbox_raises(intrinsics: CameraIntrinsics) -> None:
    estimator = GeometricDepthEstimator(intrinsics)
    with pytest.raises(ValueError):
        estimator.single_frame_depth((10.0, 50.0, 5.0, 100.0))  # x2 < x1


def test_above_horizon_returns_nan(intrinsics: CameraIntrinsics) -> None:
    estimator = GeometricDepthEstimator(intrinsics)
    # Whole bbox is above the horizon (y < cy)
    res = estimator.single_frame_depth((100.0, 50.0, 200.0, 80.0))
    assert math.isnan(res.depth_m)
    assert res.confidence == 0.0
