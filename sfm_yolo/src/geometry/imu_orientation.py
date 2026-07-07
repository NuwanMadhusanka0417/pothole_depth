"""Camera orientation from the IMU gravity vector (RGB + IMU pipeline).

This module converts the per-frame gravity vector reported by the phone
IMU into the camera's **pitch below the horizon** -- the single quantity
the geometric depth model (:mod:`geometric_depth`) needs in place of the
fixed calibration ``pitch_deg``.

Why gravity (and not the fused ``attitude_*`` angles)?
    The gravity direction is observed directly from the accelerometer and
    is **drift-free** in pitch and roll (it is anchored to the Earth's
    gravity field). Yaw, by contrast, drifts over time. A ground-plane
    depth model only ever needs pitch (and optionally roll), so gravity is
    the cleanest, most robust source.

Geometry
--------
Let ``o`` be the camera optical-axis unit vector and ``g`` the unit gravity
vector (pointing *down*), both expressed in the device frame. The
depression of the optical axis below the horizontal plane is::

    pitch = arcsin( o . g )

* optical axis horizontal    -> o _|_ g -> pitch = 0
* optical axis straight down  -> o || g -> pitch = +90 deg

This is exactly the angle the geometric model adds to the per-pixel
``arctan((y - cy) / f)`` term, so it drops straight in as ``pitch_rad``.

The optical axis depends on which camera recorded the clip. In the iOS
device frame (+X right, +Y up toward the top of the screen, +Z out of the
screen toward the user)::

    rear  camera -> o = (0, 0, -1)
    front camera -> o = (0, 0, +1)

Roll (rotation about the optical axis) is recovered from the gravity
components that lie in the image plane. The current geometric model is
pitch-only, so roll is reported for diagnostics / frame rejection rather
than used directly -- a large roll means the flat-horizon assumption is
violated for that frame.

Calibration note
----------------
``pitch_offset_rad`` lets you correct a constant mounting/convention bias:
record one frame where the camera is at a *known* angle (e.g. perfectly
level), read the computed pitch, and set the offset so it matches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Optical-axis unit vectors in the iOS device frame.
REAR_CAMERA_AXIS: Tuple[float, float, float] = (0.0, 0.0, -1.0)
FRONT_CAMERA_AXIS: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class CameraOrientation:
    """Camera orientation relative to the (horizontal) ground plane."""

    pitch_rad: float        # depression of the optical axis below horizon (+down)
    roll_rad: float         # rotation about the optical axis (0 = level horizon)
    gravity_norm_g: float   # |gravity| in g-units; should be ~1.0 for a clean read

    @property
    def pitch_deg(self) -> float:
        return math.degrees(self.pitch_rad)

    @property
    def roll_deg(self) -> float:
        return math.degrees(self.roll_rad)

    @property
    def is_valid(self) -> bool:
        return (
            math.isfinite(self.pitch_rad)
            and math.isfinite(self.roll_rad)
            and 0.8 <= self.gravity_norm_g <= 1.2
        )


def gravity_vector_from_imu(imu: dict) -> np.ndarray:
    """Extract the gravity vector (g-units) from a misensorkit IMU record."""
    return np.array(
        [
            float(imu["gravity_x_g"]),
            float(imu["gravity_y_g"]),
            float(imu["gravity_z_g"]),
        ],
        dtype=np.float64,
    )


def orientation_from_gravity(
    gravity: np.ndarray,
    *,
    camera_axis: Tuple[float, float, float] = REAR_CAMERA_AXIS,
    pitch_offset_rad: float = 0.0,
) -> CameraOrientation:
    """Compute camera pitch/roll below the horizon from a gravity vector.

    Parameters
    ----------
    gravity : ndarray, shape (3,)
        Gravity vector in the device frame (g-units). Need not be
        normalised -- it is normalised internally.
    camera_axis : tuple
        Optical-axis unit vector in the device frame. Use
        :data:`REAR_CAMERA_AXIS` (default) for the rear/world camera or
        :data:`FRONT_CAMERA_AXIS` for the front/TrueDepth camera.
    pitch_offset_rad : float
        Constant additive correction for a known mounting/convention bias.

    Returns
    -------
    CameraOrientation
    """
    g = np.asarray(gravity, dtype=np.float64)
    norm = float(np.linalg.norm(g))
    if norm < 1e-6:
        return CameraOrientation(float("nan"), float("nan"), norm)

    g_hat = g / norm
    o = np.asarray(camera_axis, dtype=np.float64)
    o = o / np.linalg.norm(o)

    # Pitch: depression of the optical axis below the horizontal plane.
    pitch = math.asin(float(np.clip(np.dot(o, g_hat), -1.0, 1.0))) + pitch_offset_rad

    # Roll: angle of the in-image gravity direction away from "image down".
    # Image x = device +X (right); image down = device -Y. Roll is 0 when
    # gravity points straight down the image (phone upright in portrait).
    roll = math.atan2(float(g_hat[0]), float(-g_hat[1]))

    return CameraOrientation(pitch_rad=pitch, roll_rad=roll, gravity_norm_g=norm)


def orientation_from_imu(
    imu: dict,
    *,
    camera_axis: Tuple[float, float, float] = REAR_CAMERA_AXIS,
    pitch_offset_rad: float = 0.0,
) -> CameraOrientation:
    """Convenience wrapper: misensorkit IMU record -> :class:`CameraOrientation`."""
    return orientation_from_gravity(
        gravity_vector_from_imu(imu),
        camera_axis=camera_axis,
        pitch_offset_rad=pitch_offset_rad,
    )


# ---------------------------------------------------------------------------
# Upright-rotation detection (for landscape / tilted hand-held capture)
# ---------------------------------------------------------------------------
# The geometric depth model assumes "down in the image" equals "down in the
# world" (roll = 0). When the phone is held in landscape the saved frames are
# rotated 90 deg and that assumption breaks. We detect how many lossless
# 90 deg turns (np.rot90, counter-clockwise) bring the image upright -- i.e.
# make gravity point straight down within the image -- purely from the IMU.
#
# CRITICAL: this only ever touches the gravity components *in the image plane*
# (gx, gy). The pitch the depth model needs is derived from gz (the optical-
# axis component), which an in-plane rotation leaves untouched. So rotating to
# upright cannot corrupt the depth-critical IMU reading.

def _in_image_gravity(gravity: np.ndarray) -> Tuple[float, float]:
    """Gravity components in image axes (x = right, y = down).

    Uses the device->image mapping image_right = +X_device,
    image_down = -Y_device (validated against the sample captures).
    """
    gx, gy, _gz = (float(v) for v in gravity)
    return gx, -gy


def _turn_once(vx: float, vy: float) -> Tuple[float, float]:
    """Transform an in-image vector under one np.rot90 (CCW) turn.

    A pixel displacement (dx, dy) maps to (dy, -dx), the linear part of the
    np.rot90 pixel transform. Gravity-in-image transforms the same way.
    """
    return vy, -vx


def quarter_turns_to_upright(gravity: np.ndarray) -> int:
    """Number of np.rot90 (CCW) turns that make gravity point down in-image.

    Returns one of {0, 1, 2, 3}. 0 means the frame is already upright.
    """
    vx0, vy0 = _in_image_gravity(gravity)
    best_k, best_down = 0, -float("inf")
    for k in range(4):
        vx, vy = vx0, vy0
        for _ in range(k):
            vx, vy = _turn_once(vx, vy)
        if vy > best_down:  # most "down" (largest +y component)
            best_down, best_k = vy, k
    return best_k


def residual_roll_rad(gravity: np.ndarray, k: int) -> float:
    """Roll remaining after applying ``k`` upright turns (radians).

    0 means gravity points exactly down in the rotated image. A few degrees
    of residual is expected from hand wobble; a large value means the snap
    to a 90 deg multiple did not fully level the horizon.
    """
    vx, vy = _in_image_gravity(gravity)
    for _ in range(k % 4):
        vx, vy = _turn_once(vx, vy)
    return math.atan2(vx, vy)  # angle of in-image gravity away from straight-down


def gravity_camera_frame(
    gravity: np.ndarray,
    *,
    camera_axis: Tuple[float, float, float] = REAR_CAMERA_AXIS,
    k: int = 0,
) -> np.ndarray:
    """Gravity in the OpenCV camera frame (x right, y down, z forward) of the
    ``k``-times-upright-rotated image. Points *down*; unit length.

    Needed by :mod:`plane_depth` to orient the road plane. The device frame is
    (x right, y up, z toward the user). For the rear camera the optical axis is
    -Z_device (into the scene); for the front camera it is +Z_device and the
    image is mirrored in x. After mapping to the OpenCV frame we apply the same
    in-plane rotation the image received.
    """
    gx, gy, gz = (float(v) for v in gravity)
    if camera_axis[2] < 0:      # rear / world camera
        cx, cy, cz = gx, -gy, -gz
    else:                        # front camera (mirrored in x)
        cx, cy, cz = -gx, -gy, gz
    for _ in range(k % 4):       # match the image's np.rot90 turns
        cx, cy = _turn_once(cx, cy)
    v = np.array([cx, cy, cz], dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v
