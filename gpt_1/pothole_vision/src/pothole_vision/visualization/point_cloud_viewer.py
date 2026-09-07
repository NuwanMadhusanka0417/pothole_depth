"""Point cloud viewer placeholder."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def show_point_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> None:
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        o3d.visualization.draw_geometries([pcd])
    except ImportError:
        print(f"Open3D not available. Point cloud has {len(points)} points.")
