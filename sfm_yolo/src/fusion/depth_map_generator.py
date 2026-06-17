"""Depth-map and overlay visualisations.

Two complementary outputs:

  * **Per-frame ground-plane depth map**: an image of the same size as
    the source frame whose pixels encode the metric distance to the
    road *plane*. This requires only the camera calibration; it is
    extremely cheap to compute and useful as a sanity check.

  * **Pothole depth overlay**: takes a list of detections + their final
    depth estimates and renders the values on top of the frame so a
    human can validate the pipeline visually.

  * **Sparse depth from a scaled SfM cloud**: project the 3D points
    onto a frame and colour each pixel by depth.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..detection.yolo_detector import Detection
from ..geometry.geometric_depth import GeometricDepthEstimator
from ..reconstruction.sfm_runner import SfMResult


# ---------------------------------------------------------------------------
# Ground-plane depth map (geometric)
# ---------------------------------------------------------------------------
def ground_plane_depth_map(
    image_shape: Tuple[int, int],
    geometric: GeometricDepthEstimator,
    *,
    max_distance_m: float = 80.0,
) -> np.ndarray:
    """Return an ``(H, W)`` float32 array of metric distances to the road.

    Pixels above the horizon are set to ``np.nan``.
    """
    H, W = image_shape[:2]
    ys = np.arange(H, dtype=np.float64).reshape(-1, 1)
    distances = geometric.pixel_to_distance(ys)  # (H, 1) - same value across cols
    distances = np.tile(distances, (1, W))
    distances = np.where(np.isfinite(distances) & (distances > 0), distances, np.nan)
    distances = np.where(distances > max_distance_m, np.nan, distances)
    return distances.astype(np.float32)


def colorize_depth(
    depth_map: np.ndarray,
    *,
    cmap: int = cv2.COLORMAP_TURBO,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Return a BGR uint8 image visualising ``depth_map``."""
    valid = np.isfinite(depth_map)
    if not valid.any():
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    finite = depth_map[valid]
    if vmin is None:
        vmin = float(np.percentile(finite, 2))
    if vmax is None:
        vmax = float(np.percentile(finite, 98))
    vmax = max(vmax, vmin + 1e-6)
    norm = np.clip((depth_map - vmin) / (vmax - vmin), 0.0, 1.0)
    norm = np.where(valid, norm, 0.0)
    norm_u8 = (norm * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(norm_u8, cmap)
    coloured[~valid] = (0, 0, 0)
    return coloured


# ---------------------------------------------------------------------------
# Pothole depth overlay
# ---------------------------------------------------------------------------
def annotate_depth_overlay(
    frame: np.ndarray,
    detections: Sequence[Detection],
    depths_m: Sequence[float],
    confidences: Sequence[float] | None = None,
    *,
    bbox_color: Tuple[int, int, int] = (0, 0, 255),
    text_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Draw bbox + ``depth=X.XXm`` label per detection."""
    if confidences is None:
        confidences = [float("nan")] * len(detections)
    if len(depths_m) != len(detections) or len(confidences) != len(detections):
        raise ValueError("depths_m / confidences length mismatch")

    out = frame.copy()
    for det, depth, conf in zip(detections, depths_m, confidences):
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), bbox_color, 2)
        if not np.isfinite(depth):
            label = f"{det.class_name}: depth?"
        else:
            label = f"{det.class_name}: {depth*100:.1f} cm"
            if np.isfinite(conf):
                label += f"  conf={conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), bbox_color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, max(th + 3, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Sparse depth from a scaled SfM cloud
# ---------------------------------------------------------------------------
def sparse_depth_from_pointcloud(
    sfm: SfMResult,
    image_shape: Tuple[int, int],
    *,
    pose_index: int = 0,
    radius: int = 3,
) -> np.ndarray:
    """Project a scaled SfM cloud onto a frame and return a depth map.

    The map has ``np.nan`` everywhere there is no projected point.
    """
    H, W = image_shape[:2]
    out = np.full((H, W), np.nan, dtype=np.float32)
    if sfm.K is None or sfm.points_3d.size == 0 or not sfm.poses:
        return out
    pose = sfm.poses[pose_index]
    cam_pts = (pose.R @ sfm.points_3d.T).T + pose.t
    z = cam_pts[:, 2]
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        return out
    cam_pts = cam_pts[valid]

    proj = (sfm.K @ cam_pts.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    depths = np.linalg.norm(cam_pts, axis=1)

    for (u, v), d in zip(uv, depths):
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        ui = int(round(u))
        vi = int(round(v))
        if 0 <= ui < W and 0 <= vi < H:
            cv2.circle(out, (ui, vi), radius, float(d), -1)
    return out


# ---------------------------------------------------------------------------
# Convenience: side-by-side debug image
# ---------------------------------------------------------------------------
def make_debug_panel(
    frame: np.ndarray,
    overlay: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Glue the original frame, overlay and (optional) depth map together."""
    parts: List[np.ndarray] = [frame, overlay]
    if depth_map is not None:
        parts.append(colorize_depth(depth_map))
    h = max(p.shape[0] for p in parts)
    resized = []
    for p in parts:
        scale = h / p.shape[0]
        new_w = int(round(p.shape[1] * scale))
        resized.append(cv2.resize(p, (new_w, h)))
    return np.hstack(resized)


def write_image(image: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path
