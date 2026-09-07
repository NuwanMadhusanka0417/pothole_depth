"""Tests for misensorkit IMU record validation."""

from sfm_yolo.src.utils.misensorkit_loader import is_usable_imu_record


def test_usable_imu_with_gravity():
    assert is_usable_imu_record(
        {
            "available": True,
            "gravity_x_g": -0.88,
            "gravity_y_g": -0.01,
            "gravity_z_g": -0.47,
        }
    )


def test_unusable_imu_when_available_false():
    assert not is_usable_imu_record(
        {
            "available": False,
            "frame_index": 159,
            "timestamp_ar": 1.0,
        }
    )


def test_unusable_imu_when_gravity_missing():
    assert not is_usable_imu_record(
        {
            "available": True,
            "gravity_x_g": -0.88,
            "gravity_y_g": -0.01,
        }
    )
