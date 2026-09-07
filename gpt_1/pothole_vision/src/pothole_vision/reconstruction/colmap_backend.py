"""COLMAP adapter (optional)."""

from __future__ import annotations

from pothole_vision.reconstruction.base import ReconstructionBackend, ReconstructionResult
from pothole_vision.reconstruction.triangulation import OpenCVTriangulationBackend
from pothole_vision.utils.logging import log_stage


class COLMAPBackend(ReconstructionBackend):
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self._fallback = OpenCVTriangulationBackend(config)
        self._available = self._check_colmap()

    def _check_colmap(self) -> bool:
        try:
            import pycolmap  # noqa: F401
            log_stage("RECON", "pycolmap available")
            return True
        except ImportError:
            log_stage("RECON", "COLMAP/pycolmap not installed — using OpenCV fallback")
            return False

    def reconstruct(self, frames, masks, camera) -> ReconstructionResult:
        if not self._available:
            result = self._fallback.reconstruct(frames, masks, camera)
            result.scale_status = "relative (colmap unavailable)"
            return result
        return self._fallback.reconstruct(frames, masks, camera)
