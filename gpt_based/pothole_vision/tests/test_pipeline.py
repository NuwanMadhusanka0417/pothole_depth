"""Smoke test for pipeline imports."""

from pathlib import Path

from pothole_vision.pipeline.stages import PipelineMode
from pothole_vision.utils.config import load_config


def test_config_loads():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")
    assert "video" in config
    assert "detection" in config
    assert "road_polygon" in config


def test_pipeline_modes():
    assert PipelineMode.DETECTION.value == "detection"
    assert PipelineMode.FULL.value == "full"
