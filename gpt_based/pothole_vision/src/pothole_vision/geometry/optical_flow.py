"""Optical flow utilities."""

from __future__ import annotations

import cv2
import numpy as np


def compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    if mask is not None:
        flow[~mask] = 0
    return flow
