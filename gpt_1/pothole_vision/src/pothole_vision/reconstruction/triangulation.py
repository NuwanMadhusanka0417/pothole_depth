"""OpenCV triangulation baseline reconstruction."""

from __future__ import annotations

import cv2
import numpy as np

from pothole_vision.geometry.feature_matching import detect_and_match
from pothole_vision.geometry.visual_odometry import estimate_relative_pose
from pothole_vision.reconstruction.base import ReconstructionBackend, ReconstructionResult
from pothole_vision.utils.logging import log_stage


class OpenCVTriangulationBackend(ReconstructionBackend):
    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    def reconstruct(
        self,
        frames: list[np.ndarray],
        masks: list[np.ndarray],
        camera,
    ) -> ReconstructionResult:
        if len(frames) < 2:
            return ReconstructionResult(scale_status="insufficient_frames")

        intrinsics = camera.intrinsics
        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        poses = [np.eye(4)]
        reproj_errors: list[float] = []

        ref_idx = 0
        P1 = intrinsics.K @ np.hstack([np.eye(3), np.zeros((3, 1))])

        for i in range(1, min(len(frames), 10)):
            pose = estimate_relative_pose(
                frames[ref_idx], frames[i],
                intrinsics, masks[ref_idx], masks[i], self.config,
            )
            log_stage("GEOMETRY", f"pair {ref_idx}-{i} inliers={pose.inlier_count} ratio={pose.inlier_ratio:.2f}")

            if not pose.valid:
                continue

            pts1, pts2, matches = detect_and_match(
                frames[ref_idx], frames[i], masks[ref_idx], masks[i],
                detector_type=self.config.get("feature_detector", "orb"),
            )
            if len(matches) < 8:
                continue

            R, t = pose.R, pose.t.reshape(3, 1)
            P2 = intrinsics.K @ np.hstack([R, t])

            pts4d = cv2.triangulatePoints(
                P1, P2,
                pts1[:50].T, pts2[:50].T,
            )
            pts3d = (pts4d[:3] / pts4d[3]).T
            valid = pts3d[:, 2] > 0
            pts3d = pts3d[valid]

            if len(pts3d) > 0:
                all_points.append(pts3d)
                colors = np.full((len(pts3d), 3), 128, dtype=np.uint8)
                all_colors.append(colors)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = pose.t
            poses.append(T)
            reproj_errors.append(pose.mean_reprojection_error)

        if not all_points:
            return ReconstructionResult(
                camera_poses=poses,
                scale_status="insufficient_parallax",
                quality_metrics={"inlier_ratio": 0.0},
            )

        cloud = np.vstack(all_points)
        colors = np.vstack(all_colors)

        return ReconstructionResult(
            camera_poses=poses,
            point_cloud_xyz=cloud,
            point_colors=colors,
            reprojection_errors=reproj_errors,
            quality_metrics={"point_count": len(cloud), "num_poses": len(poses)},
            scale_status="relative",
        )
