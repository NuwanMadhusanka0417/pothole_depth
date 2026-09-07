# Pothole Vision

Research-grade monocular dashcam pipeline for **pothole detection, tracking, 3D reconstruction, and metric depth estimation**.

Input is only an ordinary `.mp4` dashcam video — no LiDAR, stereo, IMU, CAN-bus, or external speed.

## Features

- Normalized road ROI with interactive configuration tool
- YOLO instance segmentation (custom or fallback weights)
- Multi-frame tracking with mask IoU + optical flow association
- Visual odometry + OpenCV triangulation baseline
- Optional COLMAP / VGGT / Video Depth Anything adapters
- Local road surface fitting (RANSAC plane / quadratic)
- Robust depth measurement with uncertainty and confidence scoring
- Refuses fake metric depth when geometry is insufficient
- SQLite database + JSON/CSV + annotated video + per-event folders

## Requirements

- Python 3.11+
- CUDA GPU recommended (CPU works for short clips)

## Installation

```bash
cd gpt_1/pothole_vision

# Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode (optional)
pip install -e ".[dev]"
```

### Custom pothole segmentation model

Place trained weights at:

```
data/models/pothole_seg.pt
```

If missing, the pipeline falls back to `yolov8n-seg.pt` (general COCO segmentation — **not** pothole-specific). Train a custom YOLO-seg model on pothole datasets for production use.

### Optional advanced backends

| Backend | Install |
|---------|---------|
| Video Depth Anything | See [official repo](https://github.com/DepthAnything/Video-Depth-Anything) |
| COLMAP | `pip install pycolmap` or system COLMAP |
| VGGT | See official VGGT repository |

Advanced backends are optional; the pipeline continues with OpenCV/fallback implementations.

## Project Structure

```
pothole_vision/
├── configs/          # YAML configuration
├── data/
│   ├── input/        # Place MP4 videos here
│   ├── output/       # Annotated video, JSON, cache
│   └── events/       # Per-pothole event folders
├── scripts/          # CLI tools
├── src/pothole_vision/
└── tests/
```

## Quick Start

### 1. Link or copy dashcam video

Videos from the parent project are at `../../data/dashcam/`:

```bash
# Windows (PowerShell) — symlink example
New-Item -ItemType SymbolicLink -Path "data\input\dashcam.mp4" -Target "..\..\data\dashcam\20260725_120804_NF.mp4"

# Or copy
copy "..\..\data\dashcam\20260725_120804_NF.mp4" "data\input\dashcam.mp4"
```

### 2. Inspect video

```bash
python scripts/inspect_video.py --video data/input/dashcam.mp4
```

### 3. Configure ROI (recommended)

```bash
python scripts/configure_roi.py --video data/input/dashcam.mp4
```

Click polygon vertices on a representative frame. Press `s` to save normalized coordinates to `configs/roi.yaml`.

### 4. Run detection pipeline (Milestone 1)

```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode detection --debug
```

**Outputs:**
- `data/output/annotated.mp4`
- `data/output/detections.json`
- `data/output/roi_debug.jpg`

### 5. Run tracking (Milestone 2)

```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode tracking
```

### 6. Run geometry (Milestone 3)

```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode geometry --debug
```

### 7. Limit processing for testing

```bash
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode detection --start 30 --duration 20 --max-frames 600
```

## Camera Calibration

```bash
python scripts/calibrate_camera.py --images path/to/checkerboard/images --cols 9 --rows 6
```

Saves intrinsics to `data/calibration/camera.yaml`. Pipeline continues without calibration but marks geometry confidence lower.

## Configuration

All thresholds in `configs/default.yaml` and nested configs:

| File | Purpose |
|------|---------|
| `configs/roi.yaml` | Normalized road polygon |
| `configs/detection.yaml` | YOLO model and thresholds |
| `configs/depth.yaml` | Depth backend selection |
| `configs/reconstruction.yaml` | SfM backend and RANSAC params |
| `configs/camera.yaml` | Intrinsics and distortion |

## Event Output

Each pothole track creates `data/events/PH_XXXXXX/`:

```
metadata.json
best_frame.jpg
annotated_frame.jpg
mask.png
measurement.json   (when geometry passes thresholds)
point_cloud.ply    (geometry mode)
```

## Ground Truth Evaluation

CSV format:

```csv
pothole_id,video,frame,true_depth_cm,true_width_cm,true_length_cm
PH_000001,dashcam.mp4,1320,6.5,55.0,78.0
```

```bash
python scripts/evaluate_depth.py --ground-truth data/ground_truth/measurements.csv
```

## Running Tests

```bash
pytest tests/ -v
```

## Scientific Rules

1. Detection confidence ≠ depth confidence
2. Relative neural depth ≠ guaranteed metric depth
3. Monocular SfM has unknown scale unless recovered
4. Depth measured perpendicular to local road surface
5. Never use single deepest point as final depth
6. System rejects unreliable measurements rather than fabricating values

## Known Limitations (v0.1)

- Default YOLO weights are not pothole-trained — provide `data/models/pothole_seg.pt`
- Full metric measurement requires custom model + calibration + sufficient parallax
- Video Depth Anything / COLMAP / VGGT need separate installation
- Dynamic object masking uses general YOLO classes when enabled

## Dataset Recommendations

Include hard negatives in training data: shadows, puddles, manholes, asphalt patches, cracks, road joints, leaves, stains, repair patches. Do not classify potholes by darkness alone.

## License

MIT
