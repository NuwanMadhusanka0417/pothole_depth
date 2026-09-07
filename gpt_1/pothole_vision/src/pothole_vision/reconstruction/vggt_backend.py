"""VGGT adapter (optional)."""

from __future__ import annotations

from pothole_vision.reconstruction.base import ReconstructionBackend, ReconstructionResult
from pothole_vision.reconstruction.triangulation import OpenCVTriangulationBackend
from pothole_vision.utils.logging import log_stage


class VGGTBackend(ReconstructionBackend):
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self._fallback = OpenCVTriangulationBackend(config)
        self._available = False
        try:
            import importlib
            importlib.import_module("vggt")
            self._available = True
            log_stage("RECON", "VGGT module found")
        except ImportError:
            log_stage("RECON", "VGGT not installed — using OpenCV fallback")

    def reconstruct(self, frames, masks, camera) -> ReconstructionResult:
        if not self._available:
            result = self._fallback.reconstruct(frames, masks, camera)
            result.scale_status = "relative (vggt unavailable)"
            return result
        return self._fallback.reconstruct(frames, masks, camera)
