"""Full pothole geometry measurement orchestration."""

from __future__ import annotations

import numpy as np

from pothole_vision.measurement.depth_measurement import compute_depth_stats
from pothole_vision.measurement.dimensions import compute_dimensions_2d
from pothole_vision.models import PotholeMeasurement
from pothole_vision.confidence.uncertainty import fuse_confidence, should_reject_measurement


def measure_pothole(
    pothole_id: str,
    depths_m: np.ndarray,
    boundary_xy_m: np.ndarray,
    detection_confidence: float,
    tracking_confidence: float,
    geometry_confidence: float,
    scale_confidence: float,
    surface_confidence: float,
    config: dict | None = None,
) -> PotholeMeasurement:
    cfg = config or {}
    conf_cfg = cfg.get("confidence", {})

    depth_stats = compute_depth_stats(
        depths_m,
        cfg.get("measurement", {}).get("deepest_percentile", 98.0),
        cfg.get("measurement", {}).get("robust_deepest_fraction", 0.03),
    )

    length, width, area = compute_dimensions_2d(boundary_xy_m)

    reject, reason = should_reject_measurement(
        geometry_confidence, scale_confidence, len(depths_m), config=cfg
    )

    overall = fuse_confidence(
        detection_confidence, tracking_confidence, geometry_confidence,
        scale_confidence, surface_confidence,
        depth_confidence=geometry_confidence * scale_confidence,
    )

    metric_available = (
        not reject
        and scale_confidence >= conf_cfg.get("approximate_accept", 0.65)
        and depth_stats["maximum_depth_m"] is not None
    )

    if overall >= conf_cfg.get("metric_accept", 0.85):
        status = "accepted"
    elif overall >= conf_cfg.get("approximate_accept", 0.65):
        status = "approximate"
    else:
        status = "unreliable"
        metric_available = False

    unc_cm = float(np.std(depths_m[np.isfinite(depths_m)]) * 100) if metric_available else None

    return PotholeMeasurement(
        pothole_id=pothole_id,
        metric_depth_available=metric_available,
        maximum_depth_cm=depth_stats["maximum_depth_m"] * 100 if metric_available else None,
        mean_depth_cm=depth_stats["mean_depth_m"] * 100 if metric_available else None,
        median_depth_cm=depth_stats["median_depth_m"] * 100 if metric_available else None,
        depth_uncertainty_cm=unc_cm,
        width_cm=width * 100 if metric_available else None,
        length_cm=length * 100 if metric_available else None,
        area_m2=area if metric_available else None,
        detection_confidence=detection_confidence,
        tracking_confidence=tracking_confidence,
        geometry_confidence=geometry_confidence,
        scale_confidence=scale_confidence,
        road_surface_confidence=surface_confidence,
        depth_confidence=geometry_confidence * scale_confidence,
        overall_confidence=overall,
        measurement_status=status if not reject else "unreliable",
        rejection_reason=reason if reject else None,
    )
