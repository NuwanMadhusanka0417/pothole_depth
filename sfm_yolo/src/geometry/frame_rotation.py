"""Lossless 90-degree frame + intrinsics rotation for the RGB+IMU pipeline.

When the phone is held in landscape, the saved frames are rotated and the
pitch-only geometric model's "image-down = world-down" assumption breaks.
We rotate frames (and their intrinsics) to upright for processing so the
model is valid again and YOLO sees the road the right way up.

A 90-degree rotation via :func:`numpy.rot90` is **lossless** (transpose +
flip, no resampling), so no image quality is lost. The number of turns is
chosen from the IMU gravity vector
(:func:`..geometry.imu_orientation.quarter_turns_to_upright`).

``k`` throughout is the number of counter-clockwise 90-degree turns, matching
``numpy.rot90(img, k)``.
"""

from __future__ import annotations

import numpy as np

from ..utils.camera_calibration import CameraIntrinsics


def rotate_image(img: np.ndarray, k: int) -> np.ndarray:
    """Rotate an image by ``k`` CCW quarter-turns (lossless)."""
    if k % 4 == 0:
        return img
    return np.ascontiguousarray(np.rot90(img, k % 4))


def rotate_intrinsics(intr: CameraIntrinsics, k: int) -> CameraIntrinsics:
    """Rotate camera intrinsics to match :func:`rotate_image` by ``k`` turns.

    For odd ``k`` the image width/height swap and so do (fx, fy). Note
    :class:`CameraIntrinsics` stores a single focal length; for the square,
    fx==fy captures used here that is exact.
    """
    k = k % 4
    w, h = intr.width, intr.height
    fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy

    if k == 0:
        return intr
    if k == 1:        # pixel map (x,y) -> (y, w-1-x); size (w,h) -> (h,w)
        nfx, nfy = fy, fx
        ncx, ncy = cy, (w - 1) - cx
        nw, nh = h, w
    elif k == 2:      # (x,y) -> (w-1-x, h-1-y)
        nfx, nfy = fx, fy
        ncx, ncy = (w - 1) - cx, (h - 1) - cy
        nw, nh = w, h
    else:             # k == 3: (x,y) -> (h-1-y, x); size (w,h) -> (h,w)
        nfx, nfy = fy, fx
        ncx, ncy = (h - 1) - cy, cx
        nw, nh = h, w

    return CameraIntrinsics(
        camera_height_m=intr.camera_height_m,
        focal_length_px=float(0.5 * (nfx + nfy)),
        principal_point=(float(ncx), float(ncy)),
        image_size=(int(nw), int(nh)),
        pitch_deg=intr.pitch_deg,
        distortion=intr.distortion,
    )
