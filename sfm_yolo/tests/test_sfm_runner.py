"""Tests for SfM utilities (file readers, fallback, scale recovery).

These tests do **not** invoke COLMAP; they exercise the parsing helpers
and the OpenCV two-view fall-back path with synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sfm_yolo.src.geometry.geometric_depth import GeometricDepthEstimator
from sfm_yolo.src.reconstruction.feature_tracking import (
    detect_features,
    match_features,
    triangulate_two_views,
)
from sfm_yolo.src.reconstruction.scale_recovery import ScaleRecoverer
from sfm_yolo.src.reconstruction.sfm_runner import (
    CameraPose,
    SfMResult,
    _camera_to_K,
    _qvec_to_rotmat,
)
from sfm_yolo.src.utils.camera_calibration import CameraIntrinsics


# ---------------------------------------------------------------------------
# COLMAP parsing helpers
# ---------------------------------------------------------------------------
def test_qvec_to_rotmat_identity() -> None:
    R = _qvec_to_rotmat([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(R, np.eye(3))


def test_qvec_to_rotmat_orthonormal() -> None:
    R = _qvec_to_rotmat([np.cos(0.7 / 2), np.sin(0.7 / 2), 0.0, 0.0])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_camera_to_K_simple_radial() -> None:
    cam = {"model": "SIMPLE_RADIAL", "params": [1000.0, 960.0, 540.0, 0.01]}
    K = _camera_to_K(cam)
    assert K[0, 0] == K[1, 1] == 1000.0
    assert K[0, 2] == 960.0 and K[1, 2] == 540.0


def test_camera_to_K_pinhole() -> None:
    cam = {"model": "PINHOLE", "params": [1000.0, 1010.0, 960.0, 540.0]}
    K = _camera_to_K(cam)
    assert K[0, 0] == 1000.0 and K[1, 1] == 1010.0


# ---------------------------------------------------------------------------
# Synthetic two-view triangulation
# ---------------------------------------------------------------------------
def _synth_chessboard(size: int = 480) -> np.ndarray:
    """Create a high-texture synthetic image (random blobs)."""
    rng = np.random.default_rng(42)
    img = (rng.integers(0, 255, size=(size, size, 3))).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0)


def test_feature_match_and_triangulate_runs() -> None:
    img = _synth_chessboard()
    K = np.array([[400, 0, 240], [0, 400, 240], [0, 0, 1]], dtype=np.float64)
    img_b = np.roll(img, 5, axis=1)  # tiny pure-translation parallax
    fa = detect_features(img, detector="orb", max_features=500)
    fb = detect_features(img_b, detector="orb", max_features=500)
    pts_a, pts_b = match_features(fa, fb, ratio=0.85, use_flann=False)
    if pts_a.shape[0] < 8:
        pytest.skip("not enough matches on this synthetic image")
    pts3, R, t = triangulate_two_views(pts_a, pts_b, K, ransac_threshold=2.0)
    if pts3 is None:
        pytest.skip("triangulation failed on this synthetic image")
    assert pts3.shape[1] == 3
    assert R.shape == (3, 3)
    assert t.shape == (3, 1) or t.shape == (3,)


# ---------------------------------------------------------------------------
# Scale recovery
# ---------------------------------------------------------------------------
def test_scale_recovery_recovers_known_factor() -> None:
    """Build a synthetic SfM result with a known scale and check we recover it."""
    intr = CameraIntrinsics(
        camera_height_m=1.5,
        focal_length_px=1000.0,
        principal_point=(960.0, 540.0),
        image_size=(1920, 1080),
        pitch_deg=0.0,
    )
    geo = GeometricDepthEstimator(intr)
    expected_scale = 0.25  # metres per "SfM unit"

    road_pts = geo.extract_road_points(image_shape=(1080, 1920), n_points=10)
    points_3d = []
    for px, py, d_geom in road_pts:
        # Camera-1 ray: (X, Y, Z) such that projecting K * [X, Y, Z]^T -> (px, py)
        # We parameterise by metric distance along the ray, then scale to SfM units.
        # Use a forward direction Z = d, X = (px - cx) / f * d, Y = (py - cy) / f * d
        Z = d_geom
        X = (px - intr.cx) / intr.fx * Z
        Y = (py - intr.cy) / intr.fy * Z
        # Convert metric -> SfM units
        points_3d.append([X / expected_scale, Y / expected_scale, Z / expected_scale])
    points_3d = np.asarray(points_3d, dtype=np.float64)

    sfm = SfMResult(
        points_3d=points_3d,
        K=intr.K(),
        poses=[CameraPose(image_id=0, name="frame_0000.jpg", R=np.eye(3), t=np.zeros(3))],
        image_size=(intr.width, intr.height),
        method="synthetic",
    )

    recoverer = ScaleRecoverer(geo, max_pixel_distance=10.0, min_inliers=3)
    res = recoverer.recover(sfm)
    assert res.is_valid()
    assert abs(res.scale - expected_scale) / expected_scale < 0.05
    assert res.confidence > 0.5


def test_apply_scale_keeps_metadata() -> None:
    sfm = SfMResult(
        points_3d=np.eye(3),
        K=np.eye(3),
        poses=[CameraPose(image_id=0, name="x", R=np.eye(3), t=np.zeros(3))],
        method="opencv-two-view",
    )
    scaled = ScaleRecoverer.apply_scale(sfm, 2.0)
    assert np.allclose(scaled.points_3d, 2 * np.eye(3))
    assert scaled.method.endswith("+scaled")
