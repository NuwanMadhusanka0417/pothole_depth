"""Local road surface estimation around pothole."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pothole_vision.road_surface.quadratic_surface import eval_quadratic, fit_quadratic_surface
from pothole_vision.road_surface.ransac_plane import fit_plane_ransac
from pothole_vision.utils.math import point_to_plane_distance


@dataclass
class LocalSurfaceResult:
    model_type: str  # plane | quadratic
    plane: np.ndarray | None
    quadratic_coeffs: np.ndarray | None
    rmse: float
    confidence: float
    ring_point_count: int


class LocalRoadSurface:
    def __init__(
        self,
        model: str = "auto",
        ring_width_pixels: int = 40,
        ransac_threshold: float = 0.02,
        min_road_points: int = 50,
    ) -> None:
        self.model = model
        self.ring_width = ring_width_pixels
        self.ransac_threshold = ransac_threshold
        self.min_road_points = min_road_points

    def extract_ring_mask(self, pothole_mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.ring_width * 2, self.ring_width * 2)
        )
        dilated = cv2.dilate(pothole_mask, kernel)
        ring = cv2.subtract(dilated, pothole_mask)
        return ring

    def fit(
        self,
        points_3d: np.ndarray,
        pothole_mask: np.ndarray,
        geometry_mask: np.ndarray,
    ) -> LocalSurfaceResult:
        ring = self.extract_ring_mask(pothole_mask)
        # Use ring region 3D points from geometry mask
        road_points = points_3d  # caller provides pre-filtered road ring points

        if len(road_points) < self.min_road_points:
            return LocalSurfaceResult(
                model_type="none", plane=None, quadratic_coeffs=None,
                rmse=float("inf"), confidence=0.0, ring_point_count=len(road_points),
            )

        plane, plane_rmse = fit_plane_ransac(road_points, self.ransac_threshold)
        quad_coeffs, quad_rmse = fit_quadratic_surface(road_points)

        use_quad = self.model == "quadratic" or (
            self.model == "auto" and quad_rmse < plane_rmse * 0.8 and len(road_points) >= 30
        )

        if use_quad:
            confidence = max(0.0, min(1.0, 1.0 - quad_rmse / 0.05))
            return LocalSurfaceResult(
                model_type="quadratic", plane=plane, quadratic_coeffs=quad_coeffs,
                rmse=quad_rmse, confidence=confidence, ring_point_count=len(road_points),
            )

        confidence = max(0.0, min(1.0, 1.0 - plane_rmse / 0.05))
        return LocalSurfaceResult(
            model_type="plane", plane=plane, quadratic_coeffs=quad_coeffs,
            rmse=plane_rmse, confidence=confidence, ring_point_count=len(road_points),
        )

    def depth_relative_to_surface(
        self,
        points: np.ndarray,
        surface: LocalSurfaceResult,
    ) -> np.ndarray:
        if surface.model_type == "plane" and surface.plane is not None:
            return point_to_plane_distance(points, surface.plane)
        if surface.model_type == "quadratic" and surface.quadratic_coeffs is not None:
            expected_z = eval_quadratic(surface.quadratic_coeffs, points)
            return np.abs(points[:, 2] - expected_z)
        return np.full(len(points), np.nan)
