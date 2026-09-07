"""Smoke test for pipeline imports."""

from pathlib import Path

from pothole_vision.pipeline.stages import PipelineMode
from pothole_vision.utils.config import load_config


def test_config_load():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml", root)
    assert config.video.frame_stride >= 1


def test_pipeline_modes():
    assert PipelineMode.DETECTION.value == "detection"
    assert PipelineMode.FULL.value == "full"
