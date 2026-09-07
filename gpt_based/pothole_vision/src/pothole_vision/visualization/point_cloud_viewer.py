"""Point cloud viewer stub for debug mode."""

from __future__ import annotations

import numpy as np


def save_point_cloud_ply(path: str, xyz: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Minimal PLY writer without requiring open3d at runtime."""
    n = xyz.shape[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x} {y} {z}\n")
