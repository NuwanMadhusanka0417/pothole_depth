"""Temporal depth fusion."""

from __future__ import annotations

import numpy as np


def fuse_depth_maps(maps: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if not maps:
        raise ValueError("No depth maps")
    stack = np.stack(maps, axis=0)
    if weights is None:
        return np.median(stack, axis=0)
    w = np.array(weights) / sum(weights)
    return np.average(stack, axis=0, weights=w)
