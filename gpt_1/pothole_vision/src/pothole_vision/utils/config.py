"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class VideoConfig(BaseModel):
    frame_stride: int = 1
    max_frames: int | None = None
    start_frame: int = 0
    duration_seconds: float | None = None


class PathsConfig(BaseModel):
    output_dir: str = "data/output"
    events_dir: str = "data/events"
    cache_dir: str = "data/output/cache"
    models_dir: str = "data/models"
    calibration_dir: str = "data/calibration"


class QualityConfig(BaseModel):
    min_sharpness: float = 50.0
    max_blur_score: float = 0.65
    min_brightness: float = 30.0
    max_brightness: float = 230.0
    min_contrast: float = 15.0
    max_saturation_clip_ratio: float = 0.15
    min_pothole_pixels: int = 100
    min_visible_ratio: float = 0.5


class ZonesConfig(BaseModel):
    detection_start_y: float = 0.64
    tracking_start_y: float = 0.69
    depth_start_y: float = 0.74
    depth_end_y: float = 0.88


class TrackingConfig(BaseModel):
    max_age: int = 30
    min_track_frames: int = 5
    min_frames_for_depth: int = 10
    max_best_frames: int = 20
    iou_threshold: float = 0.15
    centroid_max_distance: float = 150.0


class RoadSurfaceConfig(BaseModel):
    model: str = "auto"
    ring_width_pixels: int = 40
    ransac_threshold: float = 0.02
    min_road_points: int = 50


class MeasurementConfig(BaseModel):
    deepest_percentile: float = 97.5
    deepest_region_fraction: float = 0.03
    min_pothole_pixels: int = 200


class ConfidenceConfig(BaseModel):
    metric_accept: float = 0.85
    approximate_accept: float = 0.65


class CameraRefConfig(BaseModel):
    config_file: str = "configs/camera.yaml"
    require_calibration: bool = False


class DynamicObjectsConfig(BaseModel):
    enabled: bool = True
    classes: list[str] = Field(default_factory=lambda: [
        "car", "truck", "bus", "motorcycle", "bicycle", "person"
    ])


class LoggingConfig(BaseModel):
    level: str = "INFO"


class PipelineConfig(BaseModel):
    cache_enabled: bool = True
    debug: bool = False


class AppConfig(BaseModel):
    video: VideoConfig = Field(default_factory=VideoConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    roi: dict[str, Any] = Field(default_factory=lambda: {"config_file": "configs/roi.yaml"})
    quality: QualityConfig = Field(default_factory=QualityConfig)
    zones: ZonesConfig = Field(default_factory=ZonesConfig)
    detection: dict[str, Any] = Field(default_factory=lambda: {"config_file": "configs/detection.yaml"})
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    depth: dict[str, Any] = Field(default_factory=lambda: {"config_file": "configs/depth.yaml"})
    reconstruction: dict[str, Any] = Field(default_factory=lambda: {"config_file": "configs/reconstruction.yaml"})
    road_surface: RoadSurfaceConfig = Field(default_factory=RoadSurfaceConfig)
    measurement: MeasurementConfig = Field(default_factory=MeasurementConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    camera: CameraRefConfig = Field(default_factory=CameraRefConfig)
    dynamic_objects: DynamicObjectsConfig = Field(default_factory=DynamicObjectsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(base: Path, relative: str) -> Path:
    p = Path(relative)
    return p if p.is_absolute() else (base / p).resolve()


def load_config(config_path: Path | str | None = None, project_root: Path | None = None) -> AppConfig:
    """Load main application config and resolve nested YAML references."""
    root = project_root or Path(__file__).resolve().parents[3]
    cfg_path = resolve_path(root, str(config_path or "configs/default.yaml"))
    raw = load_yaml(cfg_path)
    return AppConfig.model_validate(raw)


def load_nested_config(config: AppConfig, key: str, project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or Path(__file__).resolve().parents[3]
    section = getattr(config, key)
    if isinstance(section, dict) and "config_file" in section:
        return load_yaml(resolve_path(root, section["config_file"]))
    if hasattr(section, "model_dump"):
        return section.model_dump()
    return dict(section) if isinstance(section, dict) else {}
