"""Point cloud I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
