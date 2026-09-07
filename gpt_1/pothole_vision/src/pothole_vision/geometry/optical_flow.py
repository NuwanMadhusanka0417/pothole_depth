"""Optical flow utilities."""

from __future__ import annotations

import cv2
import numpy as np


def compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    return flow


def mean_flow_in_region(flow: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    if mask.sum() == 0:
        return (0.0, 0.0)
    fx = flow[mask > 0, 0]
    fy = flow[mask > 0, 1]
    return (float(np.median(fx)), float(np.median(fy)))
