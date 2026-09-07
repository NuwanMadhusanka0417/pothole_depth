"""Coordinate transforms preserving original image space."""

from __future__ import annotations

import numpy as np


def normalized_to_pixel(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Convert normalized [0,1] coords to pixel (x, y)."""
    arr = np.array(points, dtype=np.float64)
    arr[:, 0] *= width
    arr[:, 1] *= height
    return arr.astype(np.int32)


def pixel_to_normalized(
    points: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return [(float(x / width), float(y / height)) for x, y in arr]


def remap_points_after_resize(
    points: np.ndarray,
    orig_w: int,
    orig_h: int,
    new_w: int,
    new_h: int,
    offset_x: float = 0,
    offset_y: float = 0,
) -> np.ndarray:
    """Map points from resized/cropped inference space back to original coordinates."""
    sx = orig_w / new_w
    sy = orig_h / new_h
    pts = np.asarray(points, dtype=np.float64).copy()
    pts[:, 0] = pts[:, 0] * sx + offset_x
    pts[:, 1] = pts[:, 1] * sy + offset_y
    return pts


def remap_bbox_to_original(
    bbox: tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
    infer_w: int,
    infer_h: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    sx, sy = orig_w / infer_w, orig_h / infer_h
    return (
        int(x1 * sx),
        int(y1 * sy),
        int(x2 * sx),
        int(y2 * sy),
    )
