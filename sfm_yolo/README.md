# sfm_yolo — Pothole detection & metric depth estimation

Production-ready pipeline that combines:

1. **YOLOv8** for real-time pothole detection (fine-tuned on the
   Mendeley pothole video dataset),
2. **Custom geometric triangulation** that turns a single dash-camera
   into an absolute scale reference (no GPS, no stereo),
3. **Structure-from-Motion (COLMAP)** for dense relative 3-D
   reconstruction of each pothole,
4. **Hybrid fusion** that uses (2) to scale (3) into metres and reach
   the **±5–8 % depth-accuracy** target.

```
Video frame ──► YOLO ──► bbox ──► Geometric depth ──┐
                                                    ├──► Hybrid depth
        Multi-frame stack ──► COLMAP ──► 3-D cloud ─┘   (metric ±5-8%)
```

The math behind step (2): a pothole bbox top edge falls on the road
plane at row `y_top`, the bottom edge at row `y_bot`. With camera
height `h`, focal length `f` and principal point `cy`:

```
theta(y) = atan((y - cy) / f) + pitch
d(y)     = h / tan(theta(y))
depth    = (d(y_top) - d(y_bot)) * tan(theta(y_top))
```

This single trick unlocks absolute scale with only a tape-measure
calibration — see [`src/geometry/geometric_depth.py`](src/geometry/geometric_depth.py)
for the full derivation in docstrings.

---

## 1. Layout

```
sfm_yolo/
├── configs/                # YAML configs (camera, YOLO, SfM, pipeline)
├── data/                   # working dirs (gitignored except .gitkeep)
├── models/                 # trained YOLO weights
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── geometric_method_demo.ipynb
│   └── accuracy_analysis.ipynb
├── outputs/                # per-clip results / overlay videos / PLYs
├── src/
│   ├── detection/
│   │   ├── yolo_detector.py     # YOLOv8 wrapper + IoU tracker
│   │   ├── train_yolo.py        # Mendeley video -> YOLO format + fine-tune
│   │   └── inference.py         # CLI for video inference
│   ├── geometry/
│   │   ├── angle_utils.py
│   │   └── geometric_depth.py   # CORE math model
│   ├── reconstruction/
│   │   ├── feature_tracking.py  # SIFT/ORB + LK + 2-view triangulation
│   │   ├── sfm_runner.py        # COLMAP wrapper + OpenCV fallback
│   │   └── scale_recovery.py    # geometric -> SfM scale
│   ├── fusion/
│   │   ├── hybrid_estimator.py  # geometric + SfM fusion
│   │   ├── confidence_scoring.py
│   │   └── depth_map_generator.py
│   ├── utils/
│   │   ├── camera_calibration.py
│   │   ├── data_loader.py
│   │   ├── evaluation_metrics.py
│   │   └── logging_utils.py
│   └── main.py             # end-to-end CLI (process / batch)
├── tests/
└── requirements.txt
```

---

## 2. Setup

```bash
# from the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate      # Linux/macOS

pip install -r sfm_yolo/requirements.txt
```

Optional (highly recommended for the SfM branch):

* Install **COLMAP** (https://colmap.github.io/install.html) so the
  `colmap` binary is in `PATH`. On Windows the easiest path is the
  pre-built CUDA bundle.
* If you don't install COLMAP the pipeline still runs, using the
  built-in OpenCV two-view fallback (less accurate).

---

## 3. Calibrate the camera (one-off)

Edit [`sfm_yolo/configs/camera_calib.yaml`](configs/camera_calib.yaml):

```yaml
camera_height_m: 1.52        # measure from ground to lens once
focal_length_px: 1400.0      # see helper below if unknown
principal_point: [960, 540]  # default = image center for 1920x1080
image_size: [1920, 1080]
pitch_deg: 0.0
```

If you don't know the focal length, you can either:

* run the OpenCV chessboard helper (see
  `src/utils/camera_calibration.estimate_focal_length_from_chessboard`),
  or
* approximate `focal_length_px ≈ 0.5 * image_width / tan(HFOV/2)`
  using the camera spec sheet.

---

## 4. Build the YOLO dataset & train the detector

The Mendeley dataset is shipped as paired video clips. We convert it
to standard YOLO format (one frame -> one image + one `.txt` of
bboxes derived from the mask) and fine-tune YOLOv8n on it:

```bash
# from the repo root
python -m sfm_yolo.src.detection.train_yolo --config sfm_yolo/configs/yolo_config.yaml --build
python -m sfm_yolo.src.detection.train_yolo --config sfm_yolo/configs/yolo_config.yaml --train
# both at once:
python -m sfm_yolo.src.detection.train_yolo --build --train
```

After training, copy `outputs/yolo_runs/pothole_yolov8n/weights/best.pt`
to `sfm_yolo/models/pothole_yolov8n.pt` (or update the path in
`pipeline.yaml`).

Tweak `configs/yolo_config.yaml` for stride, epochs, augmentations.

---

## 5. Run the end-to-end pipeline

### Single video

```bash
python -m sfm_yolo.src.main process \
    --config sfm_yolo/configs/pipeline.yaml \
    --video data/pothole_video/pothole_video/test/rgb/0001.mp4 \
    --output sfm_yolo/outputs/runs/0001
```

If you want to test the pipeline before having trained YOLO, you can
fall back to the dataset's mask video as the "detector":

```bash
python -m sfm_yolo.src.main process \
    --video data/pothole_video/pothole_video/test/rgb/0001.mp4 \
    --mask-video data/pothole_video/pothole_video/test/mask/0001.mp4 \
    --output sfm_yolo/outputs/runs/0001
```

Add `--no-sfm` to skip the COLMAP/SfM stage (geometric only — much
faster, still produces metric depth at ±10–15 % expected error).

### Whole split

```bash
python -m sfm_yolo.src.main batch \
    --root data/pothole_video/pothole_video \
    --split test \
    --output sfm_yolo/outputs/runs/test_batch \
    --max-clips 20
```

### Outputs

For every clip you get:

```
outputs/runs/<clip>/
├── summary.json            # tracks, depths, confidences
├── overlay.mp4             # annotated video with bbox + depth labels
├── ground_plane_depth.png  # geometric distance map (sanity check)
└── track_<id>.ply          # scaled SfM point cloud (per pothole)
```

The batch command also writes a top-level `_index.json` summarising
all clips.

---

## 6. Validate accuracy

If you have hand-measured ground-truth depths, save them as
`data/ground_truth.csv` with columns `clip_id,depth_m`, then open

```
sfm_yolo/notebooks/accuracy_analysis.ipynb
```

It computes MAE / RMSE / MAPE and a "predicted vs truth" scatter
with the ±5 % accuracy band overlaid.

---

## 7. Run the unit tests

```bash
pytest sfm_yolo/tests -q
```

The tests cover the geometric math (synthetic round-trips), the
COLMAP file readers, the OpenCV fallback, scale recovery on
analytically-known data, and the evaluation metrics.

---

## 8. Programmatic API

```python
from sfm_yolo.src.utils.camera_calibration import load_camera_calibration
from sfm_yolo.src.geometry.geometric_depth import GeometricDepthEstimator
from sfm_yolo.src.reconstruction.sfm_runner import SfMRunner
from sfm_yolo.src.fusion.hybrid_estimator import HybridDepthEstimator

intr = load_camera_calibration("sfm_yolo/configs/camera_calib.yaml")
geo  = GeometricDepthEstimator(intr)
sfm  = SfMRunner(intr)                       # uses COLMAP if available
hyb  = HybridDepthEstimator(geo, sfm)

result = hyb.estimate_depth(
    bboxes=[(800, 720, 1100, 820), (820, 730, 1120, 830), ...],
    video_path="my_clip.mp4",
)
print(result.depth_m, result.confidence)
```

---

## 9. Accuracy targets

| Mode                         | Expected error |
|------------------------------|----------------|
| Geometric only (single)      | ±10–15 %       |
| Geometric only (multi-frame) | ±8–12 %        |
| Hybrid (geometric + SfM)     | ±5–8 % ✅      |

The pipeline never silently throws away the SfM branch — it falls back
to geometric whenever scale recovery fails or disagreement exceeds
`fusion.disagreement_threshold` in `pipeline.yaml`.

---

## 10. References

* Hartley & Zisserman, *Multiple View Geometry in Computer Vision*.
* COLMAP: https://colmap.github.io/
* Ultralytics YOLOv8: https://docs.ultralytics.com/
* Mendeley Pothole Video Dataset: https://data.mendeley.com/datasets/5bwfg4v4cd/3
