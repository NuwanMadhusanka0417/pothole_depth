# Pothole Vision

Research-grade monocular dashcam pipeline for **pothole detection, tracking, 3D reconstruction, and metric depth estimation**.

## Requirements

- Python 3.11+
- CUDA optional (speeds up YOLO inference)

## Installation

**Windows note:** If your project lives under a long OneDrive path, PyTorch may fail to install due to Windows path-length limits. Either:
- Enable [Long Path support](https://pip.pypa.io/warnings/enable-long-paths) in Windows, or
- Create the virtual environment in a short path, e.g. `C:\pv\venv`, then run scripts from the project folder.

```powershell
cd pothole_vision
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install torch ultralytics
$env:PYTHONPATH = "src"
```

## Input Video

Place dashcam MP4 files in `data/input/` or point `--video` to your file.

This project links sample videos from the parent dataset:

```powershell
# From pothole_vision directory
New-Item -ItemType Directory -Force -Path data\input
Copy-Item "..\..\data\dashcam\20260725_120804_NF.mp4" data\input\dashcam.mp4
```

Or use any video directly:

```powershell
python scripts/run_pipeline.py --video "..\..\data\dashcam\20260725_120804_NF.mp4" --mode detection
```

## Model Setup

1. **Custom pothole weights (recommended):** place trained weights at `data/models/pothole_seg.pt`
2. **Fallback:** if custom weights are missing, YOLOv8n-seg is downloaded automatically (general segmentation — train custom weights for potholes)

### Training recommendations

Include hard negatives in training data:

- shadows, puddles, manholes, asphalt patches, cracks, road joints, leaves, stains, repair patches

Do not classify potholes by darkness alone.

## Quick Start

### Inspect video

```powershell
python scripts/inspect_video.py --video data/input/dashcam.mp4
```

### Configure ROI (interactive)

```powershell
python scripts/configure_roi.py --video data/input/dashcam.mp4
```

Click polygon points, `s` to save normalized coords to `configs/roi.yaml`, `r` reset, `q` quit.

### Phase 1 — Detection only

```powershell
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode detection
```

**Outputs** (`data/output/<video_id>/`):

- `annotated.mp4` — ROI, zones, detections
- `detections.json`
- `roi_debug.jpg`
- `data/events/detection_snapshots/` — per-detection images

### Phase 2 — Tracking

```powershell
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode tracking
```

### Limit processing window

```powershell
python scripts/run_pipeline.py --video data/input/dashcam.mp4 --mode detection --start 30 --duration 20
```

## Camera Calibration

```powershell
python scripts/calibrate_camera.py --images data/calibration/checkerboard/ --output data/calibration/camera.yaml
```

Pipeline continues without calibration but marks geometry confidence lower.

## Configuration

Main config: `configs/default.yaml`

| File | Purpose |
|------|---------|
| `roi.yaml` | Normalized road polygon, bonnet exclusion |
| `camera.yaml` | Intrinsics (optional) |
| `detection.yaml` | YOLO model paths, thresholds |
| `depth.yaml` | Depth backend selection |
| `reconstruction.yaml` | SfM backend, feature matching |

## Project Structure

```
pothole_vision/
├── configs/          YAML configuration
├── data/
│   ├── input/        MP4 videos
│   ├── output/       Results and cache
│   ├── events/       Per-pothole folders
│   └── models/       YOLO weights
├── scripts/          CLI entry points
├── src/pothole_vision/  Library code
└── tests/            Unit tests
```

See `architecture.md` and `implementation_plan.md` for design details.

## Advanced Backends (Optional)

| Backend | Config key | Notes |
|---------|------------|-------|
| Video Depth Anything | `depth.backend=video_depth_anything` | See `configs/depth.yaml` |
| COLMAP | `reconstruction.backend=colmap` | Install COLMAP separately |
| VGGT | `reconstruction.backend=vggt` | See README when available |

If unavailable, pipeline uses OpenCV triangulation and relative depth fallback.

## Testing

```powershell
pytest tests/ -v
```

## Known Limitations (v0.1)

- Phase 1 detection pipeline is fully operational
- Tracking mode is implemented; geometry/depth/measurement modules are scaffolded for Phases 3–5
- Metric depth is **never reported** without passing confidence thresholds
- Fallback YOLO model is not pothole-specific — train `pothole_seg.pt` for production use

## Scientific Rules

1. Detection confidence ≠ depth confidence
2. Relative neural depth ≠ guaranteed metric depth
3. Monocular SfM has unknown scale unless recovered
4. System refuses unreliable measurements rather than fabricating values

## License

MIT
