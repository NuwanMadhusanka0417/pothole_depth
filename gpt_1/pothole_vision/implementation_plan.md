# Implementation Plan

## Phase 1 — Detection Pipeline (Milestone 1) ✅ Target

**Goal:** Runnable detection-only pipeline on dashcam MP4.

- [x] Project structure, configs, dependency files
- [x] Config loader (Pydantic + YAML)
- [x] Video reader with FrameData and timestamps
- [x] Interactive normalized ROI tool (`configure_roi.py`)
- [x] Fixed ROI mask + processing zones
- [x] Frame quality metrics (sharpness, blur, brightness, contrast)
- [x] YOLO segmentation detector with coordinate preservation
- [x] Detection pipeline + annotated video output
- [x] Unit tests for ROI, quality, coordinate mapping

**Command:**
```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode detection
```

**Outputs:** `annotated.mp4`, `detections.json`, ROI debug image, event preview images.

---

## Phase 2 — Tracking (Milestone 2)

- [x] Track dataclass and lifecycle states
- [x] Mask IoU + centroid + optical-flow association
- [x] Best-frame selection (quality + parallax diversity)
- [x] Event folder creation with clips
- [x] Detection cache

**Command:**
```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode tracking
```

---

## Phase 3 — Geometry (Milestone 3)

- [x] Camera intrinsics model + checkerboard calibration script
- [x] ORB feature matching with dynamic mask exclusion
- [x] Essential matrix + recoverPose
- [x] OpenCV triangulation baseline
- [x] Debug visualizations (matches, trajectory, point cloud)

**Command:**
```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode geometry --debug
```

---

## Phase 4 — Depth & Scale

- [x] DepthEstimator interface + single-frame fallback
- [x] Video Depth Anything adapter (optional, install instructions)
- [x] Metric scale estimation (robust median, MAD filtering)
- [x] Weighted depth fusion architecture

---

## Phase 5 — Measurement

- [x] RANSAC plane + quadratic surface fitting
- [x] Perpendicular depth to local road surface
- [x] Robust depth percentiles (not single deepest point)
- [x] PCA dimensions, area, volume
- [x] Multi-frame consensus (median + MAD)
- [x] Uncertainty propagation
- [x] Rejection rules (insufficient parallax, scale, etc.)

**Command:**
```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode full
```

---

## Phase 6 — Advanced Backends

- [x] COLMAP adapter interface + install docs
- [x] VGGT adapter interface + install docs
- [x] Optional dynamic object masking (YOLO COCO classes)

---

## Phase 7 — Evaluation

- [x] Detection evaluation (Precision, Recall, F1, mAP)
- [x] Depth evaluation (MAE, RMSE, ±cm bands)
- [x] Ground-truth CSV format support

---

## Testing Strategy

| Test file | Coverage |
|-----------|----------|
| `test_roi.py` | Normalized ↔ pixel conversion, mask ops |
| `test_tracking.py` | IoU association, track lifecycle |
| `test_geometry.py` | Essential matrix helpers, coordinate transforms |
| `test_road_surface.py` | RANSAC plane, point-to-plane distance |
| `test_measurement.py` | Synthetic pothole depth (5 cm on z=0 plane) |
| `test_pipeline.py` | End-to-end smoke test with synthetic frames |

---

## Dependencies

Core: Python 3.11, PyTorch, OpenCV, NumPy, SciPy, scikit-learn, scikit-image, ultralytics, Open3D, pandas, matplotlib, pydantic, tqdm, PyYAML, loguru.

Optional: Video Depth Anything, COLMAP, VGGT — installed separately; adapters degrade gracefully.

---

## Data Layout

Place one or more `.mp4` files in `data/input/`. Run without `--video` to process all files.
Outputs land in `data/output/<video_stem>/` and `data/events/<video_stem>/`.
