"""OpenCV triangulation baseline."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.geometry.feature_matching import detect_and_match
from pothole_vision.geometry.camera import Camera
from pothole_vision.geometry.visual_odometry import estimate_relative_pose
from pothole_vision.models import FrameData, ReconstructionResult
from pothole_vision.reconstruction.base import ReconstructionBackend


class OpenCVTriangulationBackend(ReconstructionBackend):
    def __init__(self, config: dict) -> None:
        self.config = config.get("reconstruction", config)

    def reconstruct(
        self,
        frames: list[FrameData],
        masks: list[np.ndarray] | None,
        camera: Camera,
        geometry_mask: np.ndarray | None = None,
    ) -> ReconstructionResult:
        if len(frames) < 2:
            return ReconstructionResult(
                camera_poses=[np.eye(4)],
                point_cloud_xyz=np.zeros((0, 3)),
                point_colors=None,
                frame_depths={},
                reprojection_errors=[],
                quality_metrics={"valid": 0.0},
                scale_status="insufficient_frames",
            )

        K = camera.K
        pose = estimate_relative_pose(
            frames[0].image, frames[-1].image, K, geometry_mask, self.config
        )

        pts1, pts2, _ = detect_and_match(
            frames[0].image, frames[-1].image, geometry_mask,
            detector=self.config.get("feature_detector", "orb"),
        )

        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([pose.R, pose.t.reshape(3, 1)])

        points_3d = np.zeros((0, 3))
        if len(pts1) >= 8 and pose.valid:
            pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            pts3d = (pts4d[:3] / pts4d[3]).T
            valid_z = pts3d[:, 2] > 0
            points_3d = pts3d[valid_z]

        return ReconstructionResult(
            camera_poses=[np.eye(4)],
            point_cloud_xyz=points_3d,
            point_colors=None,
            frame_depths={},
            reprojection_errors=[],
            quality_metrics={
                "inlier_ratio": pose.inlier_ratio,
                "inlier_count": float(pose.inlier_count),
                "parallax_deg": pose.parallax_deg,
                "valid": float(pose.valid),
            },
            scale_status="unknown_scale",
        )
