"""Coordinate transforms."""

from __future__ import annotations

import numpy as np


def pixel_to_normalized_ray(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Ray direction in camera coordinates for pixel (u,v)."""
    K_inv = np.linalg.inv(K)
    pt = np.array([u, v, 1.0])
    ray = K_inv @ pt
    return ray / np.linalg.norm(ray)


def backproject(u: float, v: float, depth: float, K: np.ndarray) -> np.ndarray:
    K_inv = np.linalg.inv(K)
    pt = np.array([u, v, 1.0])
    return (K_inv @ pt) * depth
