# Pothole Vision — System Architecture

## Overview

Monocular dashcam video pipeline for pothole detection, multi-frame tracking, 3D reconstruction, and metric depth estimation. Input is **only** an `.mp4` file from a fixed dashcam. No LiDAR, stereo, IMU, CAN-bus, or external speed input.

## Design Principles

1. **Modularity** — Every algorithm stage implements a replaceable interface.
2. **Coordinate preservation** — Original image coordinates are never lost; masks are used instead of destructive crops.
3. **Scientific honesty** — Metric depth is reported only when scale and geometry confidence exceed thresholds; otherwise status is `unreliable` or `relative_only`.
4. **Two-stage processing** — Cheap continuous detection/tracking (Stage 1) vs expensive per-event geometry (Stage 2).
5. **Config-driven** — All thresholds live in YAML, not scattered magic numbers.

## Pipeline Stages

```
MP4 → VideoReader → FrameData (+ timestamps)
    → Static ROI mask (normalized polygon)
    → Frame quality assessment
    → Instance segmentation (YOLO)
    → Multi-frame tracking (IoU + optical flow)
    → Best-frame selection (quality + parallax diversity)
    → Dynamic object masking
    → Visual odometry / essential matrix
    → Monocular / video depth estimation
    → Multi-view triangulation (OpenCV baseline; optional COLMAP/VGGT)
    → Metric scale estimation (robust median of Z_metric/Z_sfm)
    → Local road surface fitting (RANSAC plane / quadratic)
    → Pothole depth & dimensions (perpendicular to local surface)
    → Uncertainty & confidence scoring
    → SQLite + JSON/CSV + annotated video + per-event folders
```

## Module Map

| Module | Responsibility |
|--------|----------------|
| `video/` | Frame reading, sampling, quality metrics |
| `roi/` | Normalized polygon ROI, bonnet/overlay exclusions, zones |
| `detection/` | YOLO instance segmentation with coordinate remap |
| `tracking/` | Track lifecycle, association, best-frame selection |
| `geometry/` | Camera model, features, essential matrix, VO |
| `depth/` | Depth estimator adapters, temporal fusion, scale |
| `reconstruction/` | Triangulation baseline, optional COLMAP/VGGT |
| `road_surface/` | Local plane/quadratic fit around pothole ring |
| `measurement/` | Depth percentiles, PCA dimensions, volume |
| `confidence/` | Per-stage and overall confidence, rejection rules |
| `visualization/` | Overlays, debug artifacts |
| `storage/` | SQLite DB, event writer, results export |
| `pipeline/` | Orchestration, caching, stage wiring |

## Data Models

- **FrameData** — `frame_index`, `timestamp_seconds`, `image`, `width`, `height`, `quality_metrics`
- **PotholeDetection** — `bbox`, `polygon`, `binary_mask`, `confidence`, `centroid`, `area_pixels`
- **Track** — lifecycle states: NEW → ACTIVE → READY_FOR_DEPTH → FINISHED / REJECTED
- **DepthResult** — `depth_map`, `is_metric`, `scale_confidence`, `valid_mask`
- **ScaleEstimate** — `scale`, `confidence`, `number_of_points`, `dispersion`, `source`
- **ReconstructionResult** — poses, point cloud, reprojection errors, `scale_status`

## Processing Zones (normalized Y)

| Zone | Purpose |
|------|---------|
| A (far) | Detection only |
| B (middle) | Detection + tracking |
| C (near) | High-res geometry / depth candidate |

## Confidence Rules

| Overall confidence | Action |
|--------------------|--------|
| ≥ 0.85 | Metric depth accepted |
| 0.65 – 0.85 | Approximate metric depth |
| < 0.65 | Detection only; depth not reported |

## Caching

Intermediate results stored under `data/output/cache/<video_id>/`:
detections, tracks, selected frames, depth maps, poses, point clouds. Use `--force` to recompute.

## Future Extensions

- `CameraProfile` for multi-dashcam support
- Semantic road segmentation replacing static ROI
- PostgreSQL/PostGIS backend
- Ground-truth calibration for uncertainty models

## Backend Adapters

| Interface | Default | Optional |
|-----------|---------|----------|
| `PotholeDetector` | YOLO seg | Custom weights |
| `DepthEstimator` | Single-frame fallback | Video Depth Anything |
| `ReconstructionBackend` | OpenCV triangulation | COLMAP, VGGT |
| `RoadSegmentation` | Static ROI | Semantic model |

Advanced backends fail gracefully; pipeline continues with fallbacks and reduced confidence.
