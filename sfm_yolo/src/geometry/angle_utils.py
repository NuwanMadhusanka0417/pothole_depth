"""Pixel <-> angle and angle <-> distance utilities.

These small helpers are factored out from :mod:`geometric_depth` because
they are also useful for visualisations / debugging notebooks.

The geometry assumes a pinhole camera with optical axis pointing forward,
the road plane lying ``h`` metres below the camera, and the image y-axis
pointing **down** (OpenCV convention).

Sign convention for angles
--------------------------
We define ``theta`` as the **angle below the horizon** of the optical
axis, in radians. A point on the road in front of the vehicle has
``pixel_y > cy`` and therefore positive ``theta``.

Formula
-------
.. math::

    \\theta = \\arctan\\!\\left(\\frac{y - c_y}{f}\\right) + \\theta_{\\text{pitch}}

    d = \\frac{h}{\\tan\\theta}
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def pixel_y_to_angle(
    pixel_y: float | np.ndarray,
    *,
    cy: float,
    focal_length_px: float,
    pitch_rad: float = 0.0,
) -> float | np.ndarray:
    """Convert a vertical pixel coordinate to an *angle below horizon*.

    Parameters
    ----------
    pixel_y : float or ndarray
        Pixel row in the image (0 = top, ``H-1`` = bottom).
    cy : float
        Principal point row (image vertical center).
    focal_length_px : float
        Focal length in pixels (vertical).
    pitch_rad : float
        Camera pitch (positive = pointing down). Added to the per-pixel
        angle so a level point in the world has ``theta = pitch_rad``
        when the camera is tilted.

    Returns
    -------
    theta : float or ndarray
        Angle below horizon, in radians.
    """
    return np.arctan2(np.asarray(pixel_y) - cy, focal_length_px) + pitch_rad


def angle_to_distance(
    theta: float | np.ndarray,
    *,
    camera_height_m: float,
    eps: float = 1e-4,
) -> float | np.ndarray:
    """Convert an angle below horizon to a forward distance on the road.

    Implements ``d = h / tan(theta)`` with safe handling of values close
    to zero (which would otherwise blow up to infinity).
    """
    theta = np.asarray(theta, dtype=np.float64)
    safe = np.where(np.abs(theta) < eps, np.sign(theta) * eps + (theta == 0) * eps, theta)
    d = camera_height_m / np.tan(safe)
    d = np.where(np.abs(theta) < eps, np.inf, d)
    if np.isscalar(theta) or theta.ndim == 0:
        return float(d)
    return d


def pixel_y_to_distance(
    pixel_y: float | np.ndarray,
    *,
    camera_height_m: float,
    cy: float,
    focal_length_px: float,
    pitch_rad: float = 0.0,
) -> float | np.ndarray:
    """One-shot helper: pixel row -> ground-plane distance in metres."""
    theta = pixel_y_to_angle(
        pixel_y, cy=cy, focal_length_px=focal_length_px, pitch_rad=pitch_rad
    )
    return angle_to_distance(theta, camera_height_m=camera_height_m)


def horizon_pixel(*, cy: float, focal_length_px: float, pitch_rad: float) -> float:
    """Return the pixel row corresponding to the horizon.

    Useful as a sanity check / visualisation overlay. For ``pitch=0``
    this is simply ``cy``.
    """
    return float(cy - math.tan(pitch_rad) * focal_length_px)


def safe_arctan(values: Iterable[float]) -> np.ndarray:
    """Element-wise ``arctan`` returning radians, handling ``nan/inf``."""
    arr = np.asarray(list(values), dtype=np.float64)
    return np.arctan(arr)


def median_with_confidence(samples: Sequence[float]) -> tuple[float, float]:
    """Return the median of ``samples`` and a confidence in [0, 1].

    Confidence is derived from the median absolute deviation (MAD)
    relative to the median: ``conf = 1 / (1 + MAD/|median|)``.
    """
    arr = np.asarray(list(samples), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if abs(med) < 1e-9:
        return med, 0.0
    return med, float(1.0 / (1.0 + mad / abs(med)))
