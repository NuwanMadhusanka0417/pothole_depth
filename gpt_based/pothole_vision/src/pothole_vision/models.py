"""Core data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class FrameQuality:
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    blur_score: float = 0.0
    saturation_clip_ratio: float = 0.0
    usable_for_detection: bool = False
    usable_for_depth: bool = False


@dataclass
class FrameData:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray
    width: int
    height: int
    quality_metrics: FrameQuality | None = None


@dataclass
class PotholeDetection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: np.ndarray  # Nx2 float pixel coords
    binary_mask: np.ndarray  # HxW bool, full frame size
    confidence: float
    centroid: tuple[float, float]
    area_pixels: int
    frame_index: int = 0


class TrackStatus(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    READY_FOR_DEPTH = "READY_FOR_DEPTH"
    FINISHED = "FINISHED"
    REJECTED = "REJECTED"


@dataclass
class Track:
    track_id: str
    start_frame: int
    end_frame: int = 0
    detections: list[PotholeDetection] = field(default_factory=list)
    centroid_history: list[tuple[float, float]] = field(default_factory=list)
    mask_history: list[np.ndarray] = field(default_factory=list)
    bbox_history: list[tuple[int, int, int, int]] = field(default_factory=list)
    quality_history: list[FrameQuality] = field(default_factory=list)
    best_frames: list[int] = field(default_factory=list)
    status: TrackStatus = TrackStatus.NEW


@dataclass
class ScaleEstimate:
    scale: float
    confidence: float
    number_of_points: int
    dispersion: float
    source: str


@dataclass
class DepthResult:
    depth_map: np.ndarray
    is_metric: bool
    scale_confidence: float
    model_name: str
    temporal_consistency: float
    valid_mask: np.ndarray


@dataclass
class ReconstructionResult:
    camera_poses: list[np.ndarray]
    point_cloud_xyz: np.ndarray
    point_colors: np.ndarray | None
    frame_depths: dict[int, np.ndarray]
    reprojection_errors: list[float]
    quality_metrics: dict[str, float]
    scale_status: str


@dataclass
class PotholeMeasurement:
    pothole_id: str
    metric_depth_available: bool = False
    maximum_depth_cm: float | None = None
    mean_depth_cm: float | None = None
    median_depth_cm: float | None = None
    depth_uncertainty_cm: float | None = None
    width_cm: float | None = None
    length_cm: float | None = None
    area_m2: float | None = None
    volume_m3: float | None = None
    detection_confidence: float = 0.0
    tracking_confidence: float = 0.0
    geometry_confidence: float = 0.0
    scale_confidence: float = 0.0
    road_surface_confidence: float = 0.0
    depth_confidence: float = 0.0
    overall_confidence: float = 0.0
    measurement_status: str = "pending"
    rejection_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
