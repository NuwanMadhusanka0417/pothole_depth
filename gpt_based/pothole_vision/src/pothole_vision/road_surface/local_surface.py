"""Local road surface fitting around pothole."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.road_surface.quadratic_surface import fit_quadratic_surface
from pothole_vision.road_surface.ransac_plane import fit_plane_ransac, point_to_plane_distance


def create_road_ring_mask(
    pothole_mask: np.ndarray,
    ring_width: int = 40,
) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_width * 2 + 1, ring_width * 2 + 1)
    )
    dilated = cv2.dilate(pothole_mask.astype(np.uint8), kernel)
    ring = dilated.astype(bool) & ~pothole_mask
    return ring


def fit_local_surface(
    points_3d: np.ndarray,
    pothole_mask_2d: np.ndarray,
    pixel_coords: np.ndarray,
    config: dict | None = None,
) -> dict:
    cfg = config or {}
    model_pref = cfg.get("model", "auto")
    threshold = float(cfg.get("ransac_threshold", 0.02))

    ring_mask = create_road_ring_mask(pothole_mask_2d, cfg.get("ring_width_pixels", 40))
    road_indices = ring_mask.ravel()
    if pixel_coords.shape[0] != points_3d.shape[0]:
        return {"model": "none", "rmse": float("inf"), "confidence": 0.0}

    road_pts = points_3d[road_indices[: points_3d.shape[0]]] if road_indices.any() else points_3d
    if road_pts.shape[0] < cfg.get("min_road_points", 200):
        # use all non-pothole points
        road_pts = points_3d[~pothole_mask_2d.ravel()[: points_3d.shape[0]]]

    plane, plane_rmse, _ = fit_plane_ransac(road_pts, threshold)
    quad_coeff, quad_rmse = fit_quadratic_surface(road_pts)

    if model_pref == "plane" or (model_pref == "auto" and plane_rmse <= quad_rmse * 1.1):
        return {
            "model": "plane",
            "plane": plane,
            "rmse": plane_rmse,
            "confidence": float(np.clip(1.0 - plane_rmse / 0.05, 0, 1)),
        }

    return {
        "model": "quadratic",
        "coeff": quad_coeff,
        "rmse": quad_rmse,
        "confidence": float(np.clip(1.0 - quad_rmse / 0.05, 0, 1)),
    }


def depth_relative_to_surface(points: np.ndarray, surface: dict) -> np.ndarray:
    if surface["model"] == "plane":
        return point_to_plane_distance(points, surface["plane"])
    # Quadratic: distance as z - surface_z
    coeff = surface["coeff"]
    z_surf = np.array([
        coeff[0] * p[0] ** 2 + coeff[1] * p[1] ** 2 + coeff[2] * p[0] * p[1]
        + coeff[3] * p[0] + coeff[4] * p[1] + coeff[5]
        for p in points
    ])
    return np.maximum(z_surf - points[:, 2], 0)
