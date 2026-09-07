"""Mask combination utilities."""

from __future__ import annotations

import numpy as np


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    result = masks[0].copy()
    for m in masks[1:]:
        result = np.bitwise_and(result, m)
    return result


def subtract_mask(base: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    out = base.copy()
    out[exclude > 0] = 0
    return out


def geometry_mask(static_mask: np.ndarray, road_mask: np.ndarray, dynamic_mask: np.ndarray) -> np.ndarray:
    """M_geometry = M_static AND M_road AND NOT M_dynamic."""
    combined = combine_masks(static_mask, road_mask)
    return subtract_mask(combined, dynamic_mask)
