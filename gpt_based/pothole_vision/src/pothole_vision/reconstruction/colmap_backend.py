"""Optional COLMAP adapter."""

from __future__ import annotations

from pothole_vision.models import FrameData, ReconstructionResult
from pothole_vision.reconstruction.triangulation import OpenCVTriangulationBackend


class COLMAPBackend(OpenCVTriangulationBackend):
    def reconstruct(self, frames, masks, camera, geometry_mask=None) -> ReconstructionResult:
        raise NotImplementedError(
            "COLMAP backend not installed. See README for setup or use opencv_triangulation."
        )
