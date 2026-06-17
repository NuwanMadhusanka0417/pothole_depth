# COMPREHENSIVE CURSOR PROMPT FOR POTHOLE DETECTION & DEPTH ESTIMATION PROJECT

## PROJECT BRIEF

I am building a professional pothole detection and depth estimation system using:
- **YOLO v8** for real-time pothole detection
- **Structure from Motion (SfM)** via COLMAP for 3D reconstruction
- **Custom geometric triangulation model** for absolute metric depth calculation

**Goal**: Achieve ±5-8% depth estimation accuracy without requiring GPS or stereo cameras, using only dash camera video sequences.

**Dataset**: Mendeley Pothole Dataset (https://data.mendeley.com/datasets/5bwfg4v4cd/3) - Already downloaded and ready to use.

---

## TECHNICAL SPECIFICATIONS

### Method Overview
```
Input Video (30 FPS, 1080p)
        ↓
[YOLO v8 Detection] → Pothole bounding boxes
        ↓
    ┌───┴────────────────┬──────────────────┐
    ↓                     ↓                  ↓
[Geometric Method]  [SfM (COLMAP)]   [Feature Tracking]
Angles→Distance     Multi-view 3D    Camera motion
Camera height       Relative points   Optical flow
    ↓                     ↓
  Absolute         Relative
  Distances        Reconstruction
    │                    │
    └────────┬───────────┘
             ↓
    [Scale Recovery]
    Geometric validates SfM
    Multi-point consensus
             ↓
    [Depth Estimation]
    Dense 3D metric depth map
    ±5-8% accuracy
```

### Core Components

**1. YOLO Detection Component**
- Input: Video frames (RGB, 1920×1080)
- Model: YOLOv8n (nano) for speed
- Training: Fine-tune on Mendeley pothole dataset
- Output: Pothole bounding boxes with confidence scores
- Target: 90%+ mAP50 on pothole detection

**2. Geometric Triangulation Model (Custom)**
- Input: Pothole bbox, camera angles, camera height
- Camera height: Fixed parameter (measured once, ~1.5m)
- Formula: depth = h / tan(θ), where:
  - h = camera height (meters)
  - θ = angle below horizontal (from pixel-to-angle conversion)
- Multi-frame approach:
  - Frame 1: angle_road_1, angle_pothole_1
  - Frame 2: angle_road_2, angle_pothole_2
  - Depth = (d_road_1 - d_pothole_1 + d_road_2 - d_pothole_2) / 2
- Output: Absolute metric depth (meters) with confidence
- Math: Full trigonometric derivation provided

**3. Structure from Motion (SfM) Module**
- Tool: COLMAP (external, integrated via Python subprocess)
- Input: 5-10 video frames of pothole
- Process:
  1. Feature extraction and matching
  2. Camera pose estimation
  3. Triangulation of 3D points
  4. Dense reconstruction (optional)
- Output: Relative 3D point cloud, camera poses
- Purpose: Get detailed 3D pothole shape/surface

**4. Scale Recovery & Fusion**
- Input: SfM reconstruction (unitless), geometric distances (meters)
- Method:
  1. Extract road surface points from geometric method
  2. Find same points in SfM reconstruction
  3. Calculate scale factor λ = d_geometric / d_sfm
  4. Apply λ to entire SfM point cloud
  5. Use multi-point median for robustness
- Output: Scaled metric 3D model with confidence

**5. Depth Map Generation**
- Input: Scaled SfM 3D point cloud
- Extract: Points in pothole region
- Calculate:
  - Road surface height (max z)
  - Pothole bottom (min z)
  - Depth = max - min
- Validate: Against geometric estimate (variance check)
- Output: Final pothole depth, 3D model, confidence score

---

## DATASET SPECIFICATIONS

**Mendeley Pothole Dataset**
- Format: Video sequences + labeled pothole annotations
- Structure: Needs analysis - determine if:
  - Bounding boxes (for YOLO training)
  - Video sequences (for SfM)
  - Ground truth depth (for validation)
- Expected: Multi-frame sequences of roads with potholes
- Use: Train YOLO, validate depth estimation

---

## IMPLEMENTATION REQUIREMENTS

### Code Structure
```
pothole_project/
├── data/
│   ├── mendeley_raw/          # Downloaded dataset
│   ├── processed/             # Organized for YOLO
│   ├── videos/                # For SfM processing
│   └── ground_truth.csv       # Depth annotations (if available)
├── src/
│   ├── detection/
│   │   ├── yolo_detector.py   # YOLO v8 integration
│   │   ├── train_yolo.py      # Fine-tuning pipeline
│   │   └── inference.py       # Real-time detection
│   ├── geometry/
│   │   ├── geometric_depth.py # Your mathematical model
│   │   │   ├── pixel_to_angle()
│   │   │   ├── angle_to_distance()
│   │   │   ├── single_frame_depth()
│   │   │   ├── two_frame_depth()
│   │   │   └── multi_frame_validation()
│   ├── reconstruction/
│   │   ├── sfm_runner.py      # COLMAP integration
│   │   ├── feature_tracking.py
│   │   └── scale_recovery.py
│   ├── fusion/
│   │   ├── hybrid_estimator.py   # Geometric + SfM fusion
│   │   ├── confidence_scoring.py
│   │   └── depth_map_generator.py
│   ├── utils/
│   │   ├── camera_calibration.py
│   │   ├── data_loader.py
│   │   └── evaluation_metrics.py
│   └── main.py                 # End-to-end pipeline
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── geometric_method_demo.ipynb
│   └── accuracy_analysis.ipynb
├── configs/
│   ├── yolo_config.yaml
│   ├── sfm_config.yaml
│   └── camera_calib.json
├── requirements.txt
└── README.md
```

### Key Functions to Implement

**geometric_depth.py**
```python
class GeometricDepthEstimator:
    def __init__(self, camera_height, focal_length, principal_point, frame_height)
    def pixel_to_angle(pixel_y) -> float
    def angle_to_distance(angle) -> float
    def single_frame_depth(frame, bbox) -> dict
    def two_frame_depth(frame1, frame2, bbox) -> dict
    def multi_frame_validation(frames, bbox, num_frames=5) -> dict
    def extract_road_points(frame) -> list
    def calculate_confidence(measurements) -> float
```

**sfm_runner.py**
```python
class SfMRunner:
    def run_colmap(video_path, output_dir) -> dict
    def extract_frames(video_path, frame_indices) -> list
    def read_colmap_model(colmap_path) -> dict
    def get_3d_points() -> ndarray
    def get_camera_poses() -> list
    def triangulate_points(frame1, frame2) -> ndarray
```

**hybrid_estimator.py**
```python
class HybridDepthEstimator:
    def __init__(self, geometric_estimator, sfm_runner)
    def estimate_depth_hybrid(video_path, pothole_bbox) -> dict
    def recover_scale(sfm_result, geometric_distances) -> float
    def apply_scale_to_sfm(sfm_points, scale_factor) -> ndarray
    def compute_pothole_depth_from_3d(scaled_sfm, bbox) -> float
    def generate_confidence_score(geometric_var, sfm_var, scale_var) -> float
```

**yolo_detector.py**
```python
class YOLODetector:
    def __init__(self, model_path, device='cpu')
    def detect_potholes(frame) -> list[Detection]
    def process_video(video_path, confidence_threshold=0.5) -> list
    def draw_detections(frame, detections) -> ndarray
    def track_across_frames(frames) -> list[Track]
```

---

## IMPLEMENTATION DETAILS

### 1. Camera Calibration
- Read/prompt for camera height (critical parameter)
- Calculate focal length from Mendeley dataset metadata OR estimate from frame dimensions
- Principal point: Assume center of image (cx = width/2, cy = height/2)
- Store in config file for reproducibility

### 2. YOLO Training Pipeline
- Load Mendeley dataset
- Convert annotations to YOLO format
- Split: 70% train, 20% val, 10% test
- Train on pothole detection (fine-tune YOLOv8n)
- Validation: mAP50, precision, recall
- Save best model weights

### 3. Geometric Method
- Implement pixel-to-angle conversion: arctan((pixel_y - cy) / focal_length)
- Single frame: Extract road point and pothole point, calculate depth
- Two-frame method: Use consistency between frames as validation
- Multi-frame: Average depths from multiple frame pairs, compute variance
- Confidence: Inverse of variance (consistent = confident)

### 4. SfM Integration
- Call COLMAP via subprocess for each video
- Parse COLMAP output (cameras.txt, images.txt, points3D.txt)
- OR use pycolmap wrapper if available
- Extract 3D points and camera poses
- Handle COLMAP failures gracefully

### 5. Scale Recovery
- Extract N road points using geometric method (angles + distances)
- Find same points in SfM reconstruction (closest 3D point to projected pixel)
- Calculate N scale factors: scale_i = d_geometric_i / d_sfm_i
- Use median scale for robustness
- Compute variance to assess reliability
- Apply scale to all SfM points: points_metric = scale * points_sfm

### 6. Hybrid Fusion
- Run geometric method → confidence_geometric
- Run SfM → unscaled reconstruction
- Apply scale → metric reconstruction
- Extract pothole depth from 3D model
- Confidence_final = weighted_average(confidence_geometric, confidence_sfm)
- If confidence_final > threshold: Accept result
- Otherwise: Rerun with more frames or flag as uncertain

### 7. Validation & Metrics
- If Mendeley has ground truth: Compare predictions vs truth
- Calculate: MAE, RMSE, MAPE, median_error
- Accuracy target: ±5-8% mean absolute percentage error
- Generate visualization: 3D models, depth maps, comparison charts

---

## INPUT/OUTPUT SPECIFICATIONS

### Input
```
Video file: .mp4, .avi, or .mov
Format: RGB, any resolution (will resize to 1920×1080)
Frame rate: 24+ FPS preferred
Duration: 3-10 seconds per pothole
Metadata: Camera height (meters), focal length (pixels)
```

### Output
```
Per pothole:
├─ detection_bbox: [x1, y1, x2, y2]
├─ depth_meters: float (±uncertainty)
├─ method: "geometric" | "hybrid"
├─ confidence: float (0-1)
├─ 3d_model: point cloud (if SfM used)
├─ surface_area_m2: float (optional)
└─ volume_m3: float (optional)

Summary:
├─ num_potholes_detected: int
├─ mean_depth: float
├─ std_depth: float
├─ confidence_distribution: histogram
└─ accuracy_vs_groundtruth: dict (if available)
```

---

## TECHNICAL STACK

**Required Libraries**
```
numpy              # Mathematical operations
opencv-python     # Image processing, feature tracking
ultralytics       # YOLOv8
pycolmap          # SfM (optional, can use subprocess)
scipy              # Optimization, geometry
scikit-image      # Image processing
pandas             # Data handling
matplotlib        # Visualization
torch              # YOLO backbone
torchvision       # Image transforms
```

**External Tools**
```
COLMAP            # Structure-from-Motion (install separately)
```

---

## ACCURACY TARGETS

```
Depth Estimation Accuracy:
├─ Geometric method alone: ±10-15%
├─ SfM alone: ±12-18%
└─ Hybrid (geometric + SfM): ±5-8% ✓ TARGET

Detection Accuracy:
└─ YOLO on Mendeley: 90%+ mAP50 ✓ TARGET

Processing Speed:
├─ Per-frame detection: <50ms
├─ SfM (per pothole): ~2-5 seconds
└─ End-to-end: <30 seconds per pothole
```

---

## SPECIAL NOTES FOR CURSOR

### Geometric Model - Critical Mathematics

The geometric depth calculation is the core innovation:

**Single Frame (from bounding box + pixel location):**
```
d_road = h / tan(|θ_road|)
d_pothole = h / tan(|θ_pothole|)
depth = d_road - d_pothole
```

**Two-Frame Triangulation (recommended):**
```
Frame 1:
  d_road_1 = h / tan(θ_r1)
  d_pothole_1 = h / tan(θ_p1)
  depth_1 = d_road_1 - d_pothole_1

Frame 2:
  d_road_2 = h / tan(θ_r2)
  d_pothole_2 = h / tan(θ_p2)
  depth_2 = d_road_2 - d_pothole_2

Final:
  depth = (depth_1 + depth_2) / 2
  variance = |depth_1 - depth_2| / depth
  confidence = 1 / (1 + variance)
```

**Camera Height Invariance:**
- h (camera height) is the absolute scale reference
- All angles are measured from pixels via focal length
- Result automatically in METERS (not unitless)
- No GPS needed (h replaces GPS scale)

### SfM Integration Notes

COLMAP can be called as:
```bash
colmap feature_extractor --database_path db.db --image_path frames/
colmap exhaustive_matcher --database_path db.db
colmap mapper --database_path db.db --image_path frames/ --output_path sparse/
```

Parse output:
```
sparse/0/
├─ cameras.txt    → camera parameters
├─ images.txt     → camera poses, frame info
└─ points3D.txt   → 3D points with color
```

### Mendeley Dataset Integration

- Explore dataset structure first (identify if videos exist)
- Create data loader to read Mendeley format
- Verify if ground truth depth available for validation
- If only images: Synthetic video generation (shift + interpolate)
- If videos: Use directly for SfM

---

## PROJECT GOALS

✓ Production-ready pothole detection system  
✓ Accurate depth estimation (±5-8%)  
✓ No GPU required (geometric method can run on CPU)  
✓ No GPS dependency (uses camera height)  
✓ Works with single dash camera  
✓ Self-validating (multi-frame consistency)  
✓ Handles edge cases gracefully  
✓ Comprehensive logging and visualization  

---

## IMPORTANT: CODE QUALITY

- **Modular design**: Each component independent, testable
- **Error handling**: Graceful failures, informative messages
- **Logging**: DEBUG, INFO, WARNING levels throughout
- **Documentation**: Docstrings for all functions
- **Type hints**: Python type annotations for clarity
- **Unit tests**: For geometric method, SfM wrapper
- **Configuration**: YAML files for all parameters
- **Reproducibility**: Save configs with results

---

## START HERE

Begin with:
1. **Dataset exploration** - understand Mendeley structure
2. **Camera calibration module** - get focal length, camera height
3. **Geometric method implementation** - core algorithm
4. **YOLO detector** - pothole detection
5. **SfM integration** - COLMAP wrapper
6. **Hybrid fusion** - combine all components
7. **Validation pipeline** - measure accuracy

---

## REFERENCES

The geometric and SfM methods are well-documented:
- Mathematical derivations provided separately
- COLMAP documentation: https://colmap.github.io/
- YOLOv8: https://docs.ultralytics.com/
- Camera geometry: "Multiple View Geometry" (Hartley & Zisserman)

---

END OF PROMPT
