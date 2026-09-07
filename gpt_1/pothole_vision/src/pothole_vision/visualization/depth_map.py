"""Depth map visualization."""

from __future__ import annotations

import cv2
import numpy as np


def depth_to_colormap(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    d = depth.copy()
    if valid_mask is not None:
        d[valid_mask == 0] = np.nan
    d_min = np.nanmin(d)
    d_max = np.nanmax(d)
    if not np.isfinite(d_min) or d_max - d_min < 1e-6:
        normalized = np.zeros_like(d, dtype=np.uint8)
    else:
        normalized = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    normalized = np.nan_to_num(normalized, nan=0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
