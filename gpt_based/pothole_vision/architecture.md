# Architecture: Monocular Dashcam Pothole Vision

## Overview

Pothole Vision is a modular research pipeline that processes monocular dashcam `.mp4` video to detect potholes, track them across frames, reconstruct 3D geometry, estimate metric depth, and report measurements with explicit uncertainty.

**Input:** single fixed-mount dashcam video  
**Output:** annotated video, JSON/CSV results, per-pothole event folders, SQLite database

## Design Principles

1. **Original coordinates preserved** — masks and transforms map back to full-frame pixel space.
2. **Modular backends** — detection, depth, reconstruction are swappable via interfaces.
3. **Scientific honesty** — metric depth is never fabricated; low-confidence results are flagged.
4. **Two-stage processing** — cheap continuous detection/tracking vs expensive per-event geometry.
5. **Config-driven** — thresholds live in YAML, not scattered constants.

## Pipeline Stages

```
MP4 → VideoReader → FrameQuality → ROIMask
  → PotholeDetector → Tracker → FrameSelector
  → DynamicObjectMask → VisualOdometry → DepthEstimator
  → ReconstructionBackend → ScaleEstimation → RoadSurfaceFit
  → PotholeMeasurement → Uncertainty → Storage + Visualization
```

### Stage 1 (Continuous, Cheap)

| Module | Responsibility |
|--------|----------------|
| `video/` | Read frames, timestamps, metadata |
| `roi/` | Static normalized polygon → binary mask |
| `video/quality` | Sharpness, blur, exposure filters |
| `detection/` | Instance segmentation in ROI |
| `tracking/` | Multi-frame association, track lifecycle |
| `visualization/` | Annotated output video |

### Stage 2 (Per-Event, Expensive)

| Module | Responsibility |
|--------|----------------|
| `geometry/` | Feature matching, essential matrix, poses |
| `depth/` | Temporal monocular depth + fusion |
| `reconstruction/` | Triangulation / COLMAP / VGGT adapters |
| `road_surface/` | Local plane or quadratic fit |
| `measurement/` | Depth, width, length, volume |
| `confidence/` | Multi-factor uncertainty scoring |
| `storage/` | SQLite, JSON, event folders |

## Data Models

### FrameData
- `frame_index`, `timestamp_seconds`, `image`, `width`, `height`, `quality_metrics`

### PotholeDetection
- `bbox`, `polygon`, `binary_mask`, `confidence`, `centroid`, `area_pixels`

### Track
- `track_id`, lifecycle status, detection/mask/bbox histories, `best_frames`

### DepthResult
- `depth_map`, `is_metric`, `scale_confidence`, `valid_mask`

### ScaleEstimate
- `scale`, `confidence`, `dispersion`, `source`

### Measurement
- depth stats, dimensions, `metric_depth_available`, `measurement_status`, confidences

## Processing Zones

Normalized Y thresholds define three zones:

| Zone | Purpose |
|------|---------|
| A (far) | Detection only |
| B (middle) | Detection + tracking |
| C (near) | Depth/geometry candidates |

## Mask Composition

```
M_static = M_road_roi AND NOT M_bonnet AND NOT M_overlay AND NOT M_manual
M_geometry = M_static AND M_road_semantic AND NOT M_dynamic
```

Road segmentation (`M_road_semantic`) is optional; v1 uses static ROI only.

## Backend Interfaces

```python
class PotholeDetector:
    def predict(frame, roi_mask) -> list[PotholeDetection]: ...

class DepthEstimator:
    def estimate(frames, masks, camera) -> DepthResult: ...

class ReconstructionBackend:
    def reconstruct(frames, masks, camera) -> ReconstructionResult: ...

class RoadSegmentation:
    def segment(frame) -> np.ndarray: ...
```

## Coordinate System

- All detections stored in **original image coordinates** (full resolution).
- Resize/crop for inference uses stored affine transform for inverse mapping.
- 3D points in camera frame; metric scale applied via `ScaleEstimate`.

## Confidence Model

Separate scores merged into `overall_confidence`:

- detection, tracking, geometry, scale, road_surface, depth

Publication rules (configurable):

| Score | Status |
|-------|--------|
| ≥ 0.85 | Metric depth accepted |
| 0.65–0.85 | Approximate |
| < 0.65 | Detection only / unreliable |

## Caching

Intermediate results under `data/output/cache/<video_id>/`:

- detections, tracks, selected frames, depth maps, poses, point clouds

Use `--force` to recompute.

## Database Schema (SQLite)

- `videos` — input metadata
- `potholes` — track summaries
- `observations` — per-frame detections
- `measurements` — final geometry results

## Future Extensions

- `CameraProfile` for multi-dashcam support
- Automatic road segmentation backend
- PostgreSQL/PostGIS migration
- GPS/speed fusion (optional, not required v1)

## Directory Layout

See `README.md` for full tree. Source lives in `src/pothole_vision/` with one module per pipeline concern.
