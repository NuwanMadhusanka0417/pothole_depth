# Implementation Plan

## Phase 1 — Detection Pipeline (CURRENT)

**Goal:** Runnable `--mode detection` on dashcam MP4.

- [x] Project structure, configs, logging
- [x] Video reader + frame timestamps
- [x] Normalized ROI mask + configure_roi tool
- [x] Frame quality metrics
- [x] YOLO segmentation detector interface
- [x] Detection-only pipeline + annotated video
- [x] Unit tests for ROI, quality, coordinate mapping

**Demo command:**
```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode detection
```

**Outputs:** `annotated.mp4`, `detections.json`, ROI debug image, event snapshot images

---

## Phase 2 — Tracking

- Multi-frame tracker (mask IoU + centroid + optical flow hint)
- Track lifecycle: NEW → ACTIVE → READY_FOR_DEPTH → FINISHED
- Best-frame selection with quality + parallax diversity
- Event clip extraction
- Cache detections/tracks

**Demo:**
```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode tracking
```

---

## Phase 3 — Geometry Baseline

- Checkerboard camera calibration script
- ORB feature matching + essential matrix + recoverPose
- OpenCV triangulation backend
- Dynamic object masking for geometry
- Debug visualizations (matches, trajectory, point cloud)

**Demo:**
```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode geometry --debug
```

---

## Phase 4 — Depth + Metric Scale

- DepthEstimator interface
- Single-frame fallback depth (relative)
- Video Depth Anything adapter (optional)
- Robust scale estimation (median + MAD)
- Temporal depth fusion (weighted)

---

## Phase 5 — Measurement

- Local road surface (RANSAC plane + quadratic)
- Perpendicular depth to fitted surface
- Robust depth (percentile, not single max point)
- Width/length via PCA, volume integration
- Multi-frame consensus + uncertainty
- Confidence scoring + refusal logic

**Demo:**
```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode full
```

---

## Phase 6 — Advanced Backends

- COLMAP adapter
- VGGT adapter
- SuperPoint/LightGlue feature adapter
- Bundle adjustment hook

Each optional; pipeline falls back to OpenCV baseline.

---

## Phase 7 — Evaluation

- Detection: precision, recall, F1, mAP
- Depth: MAE, RMSE, % within tolerance
- Ground-truth CSV format support

---

## Testing Strategy

| Layer | Approach |
|-------|----------|
| Math (ROI, plane distance, robust stats) | Synthetic unit tests |
| Tracking association | Simulated trajectories |
| Measurement | Synthetic pothole on plane z=0, depth=5cm |
| Integration | Short video clip, smoke test |

---

## Dependencies Rollout

| Phase | Required | Optional |
|-------|----------|----------|
| 1 | opencv, numpy, ultralytics, pydantic | — |
| 3 | scipy, scikit-image | SIFT |
| 4 | torch | Video Depth Anything |
| 5 | scikit-learn | — |
| 6 | open3d | COLMAP, VGGT |

---

## Milestone Checklist

- [ ] Phase 1 demo on real dashcam video
- [ ] Phase 2 track IDs + event clips
- [ ] Phase 3 point cloud for one event
- [ ] Phase 4 scale confidence reported
- [ ] Phase 5 metric depth with uncertainty (when geometry passes)
- [ ] Phase 7 evaluation scripts
