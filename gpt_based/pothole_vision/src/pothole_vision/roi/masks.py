"""Mask creation and combination."""

from __future__ import annotations

import cv2
import numpy as np


def polygon_to_mask(polygon_px: np.ndarray, height: int, width: int) -> np.ndarray:
    """Create binary mask from polygon points in pixel coordinates."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = polygon_px.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def combine_masks(*masks: np.ndarray, operation: str = "and") -> np.ndarray:
    """Combine multiple boolean masks."""
    if not masks:
        raise ValueError("At least one mask required")
    result = masks[0].astype(bool).copy()
    for m in masks[1:]:
        if operation == "and":
            result &= m.astype(bool)
        elif operation == "or":
            result |= m.astype(bool)
        else:
            raise ValueError(f"Unknown operation: {operation}")
    return result


def exclude_polygon(mask: np.ndarray, polygon_px: np.ndarray) -> np.ndarray:
    """Remove polygon region from mask."""
    h, w = mask.shape
    exclusion = polygon_to_mask(polygon_px, h, w)
    return mask & ~exclusion


def bonnet_mask(height: int, width: int, bonnet_y_normalized: float | None) -> np.ndarray:
    """Mask out bonnet region below normalized Y threshold."""
    mask = np.ones((height, width), dtype=bool)
    if bonnet_y_normalized is not None:
        y_px = int(bonnet_y_normalized * height)
        mask[y_px:, :] = False
    return mask


def overlay_exclusion_mask(
    height: int, width: int, rect: list[float] | None
) -> np.ndarray:
    """Exclude metadata overlay rectangle [x1,y1,x2,y2] normalized."""
    mask = np.ones((height, width), dtype=bool)
    if rect is None or len(rect) != 4:
        return mask
    x1 = int(rect[0] * width)
    y1 = int(rect[1] * height)
    x2 = int(rect[2] * width)
    y2 = int(rect[3] * height)
    mask[y1:y2, x1:x2] = False
    return mask
