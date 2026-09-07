"""Depth map visualization."""

from __future__ import annotations

import cv2
import numpy as np


def depth_to_colormap(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    d = depth.copy()
    if valid_mask is not None:
        d[~valid_mask] = np.nan
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    d_min, d_max = np.percentile(finite, [2, 98])
    norm = np.clip((d - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    norm = np.nan_to_num(norm, nan=0.0)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
