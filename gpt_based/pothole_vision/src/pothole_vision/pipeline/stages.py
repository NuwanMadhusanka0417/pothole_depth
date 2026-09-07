"""Pipeline stage definitions."""

from __future__ import annotations

from enum import Enum


class PipelineMode(str, Enum):
    DETECTION = "detection"
    TRACKING = "tracking"
    GEOMETRY = "geometry"
    FULL = "full"
