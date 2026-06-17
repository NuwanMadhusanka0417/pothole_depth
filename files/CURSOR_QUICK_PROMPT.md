# QUICK CURSOR PROMPT (Copy-Paste Ready)

## SHORT VERSION FOR IMMEDIATE USE

---

I'm building a pothole detection and depth estimation system using YOLOv8, Structure from Motion (SfM), and a custom geometric triangulation mathematical model.

**Key Technologies:**
- YOLO v8 for pothole detection (fine-tune on Mendeley Pothole Dataset - already downloaded)
- COLMAP for Structure-from-Motion (multi-frame 3D reconstruction)
- Custom geometric triangulation using camera height and pixel angles for absolute metric depth
- Hybrid fusion of SfM + geometric method for ±5-8% depth accuracy

**Dataset:** Mendeley Pothole Dataset (https://data.mendeley.com/datasets/5bwfg4v4cd/3) - already downloaded

**Overall Algorithm:**
1. YOLO detects potholes in video frame → bounding boxes
2. Geometric method: Calculate depth from camera height, focal length, and angle below horizontal
   - Formula: depth = h / tan(θ), where h=camera_height, θ=angle to pothole
   - Multi-frame: Compare angles across frames to validate and improve accuracy
3. SfM (COLMAP): Reconstruct 3D points from video frames
4. Scale recovery: Use geometric distances to convert SfM reconstruction from unitless to meters
5. Fusion: Combine geometric confidence + SfM confidence → final depth estimate

**Required Output Structure:**
```
pothole_project/
├── data/
│   ├── mendeley_raw/
│   ├── processed/
│   ├── videos/
│   └── ground_truth.csv
├── src/
│   ├── detection/
│   │   ├── yolo_detector.py
│   │   ├── train_yolo.py
│   │   └── inference.py
│   ├── geometry/
│   │   ├── geometric_depth.py  # CORE: Your mathematical model
│   │   │   ├── pixel_to_angle()
│   │   │   ├── angle_to_distance()
│   │   ├── single_frame_depth()
│   │   ├── two_frame_depth()
│   │   └── multi_frame_validation()
│   ├── reconstruction/
│   │   ├── sfm_runner.py        # COLMAP integration
│   │   ├── feature_tracking.py
│   │   └── scale_recovery.py
│   ├── fusion/
│   │   ├── hybrid_estimator.py  # Geometric + SfM fusion
│   │   └── depth_map_generator.py
│   ├── utils/
│   │   ├── camera_calibration.py
│   │   ├── data_loader.py
│   │   └── evaluation_metrics.py
│   └── main.py
├── notebooks/
│   ├── dataset_exploration.ipynb
│   └── accuracy_analysis.ipynb
└── requirements.txt
```

**Critical Mathematical Model (Geometric Depth Estimation):**

Single frame (simple):
```
d_road = h / tan(θ_road)
d_pothole = h / tan(θ_pothole)
depth = d_road - d_pothole
```

Two-frame triangulation (recommended - more accurate):
```
Frame 1: d_road_1, d_pothole_1, depth_1
Frame 2: d_road_2, d_pothole_2, depth_2
depth_final = (depth_1 + depth_2) / 2
confidence = 1 / (1 + variance)
```

Where:
- h = camera_height (meters, ~1.5m for dash cam)
- θ = angle below horizontal = arctan((pixel_y - cy) / focal_length)
- cy = principal point y (image center)
- focal_length = from camera calibration

**Implementation Requirements:**

1. **Geometric Depth Module (geometric.py)**
   - Implement pixel_to_angle(pixel_y) → converts pixel to angle
   - Implement angle_to_distance(angle) → d = h / tan(angle)
   - Implement single_frame_depth(frame, bbox) → extracts angles, calculates depth
   - Implement two_frame_depth(frame1, frame2, bbox) → triangulation with validation
   - Implement multi_frame_validation(frames, bbox) → averaged result with confidence
   - Auto-extract road surface points (for scale reference)

2. **YOLO Detection (yolo_detector.py)**
   - Load YOLOv8n model
   - Fine-tune on Mendeley dataset
   - Process video frames, return pothole bounding boxes
   - Track potholes across frames (for multi-frame depth)

3. **SfM Integration (sfm_runner.py)**
   - Call COLMAP via subprocess
   - Parse COLMAP output (cameras.txt, images.txt, points3D.txt)
   - Return 3D points and camera poses
   - Handle failures gracefully

4. **Scale Recovery (scale_recovery.py)**
   - Extract N road points from geometric method
   - Find same points in SfM reconstruction
   - Calculate scale_factor = d_geometric / d_sfm
   - Use median for robustness

5. **Hybrid Fusion (hybrid_estimator.py)**
   - Run geometric method → d_geometric, conf_geo
   - Run SfM → unscaled 3D model
   - Apply scale → metric 3D model
   - Extract depth from 3D model
   - Confidence_final = fusion(conf_geo, conf_sfm)

6. **Main Pipeline (main.py)**
   - Load video
   - For each pothole detected:
     a. Run geometric method (fast)
     b. Run SfM (slower, 5-10 frames)
     c. Recover scale
     d. Output final depth + confidence

**Accuracy Targets:**
- Geometric only: ±10-15%
- SfM only: ±12-18%
- Hybrid (combined): ±5-8% ← TARGET
- YOLO detection: 90%+ mAP50

**Key Parameters:**
```yaml
camera_height_m: 1.52  # CRITICAL - measured once
focal_length_px: 500.0  # From calibration
principal_point: [640, 360]  # Image center
frame_height: 1080
confidence_threshold: 0.7
min_frames_for_sfm: 5
```

**Dependencies:**
```
numpy scipy opencv-python ultralytics torch pycolmap pandas matplotlib scikit-image
```

**External:** COLMAP (install separately)

Please create a complete, production-ready implementation with:
- Modular components (each file independent)
- Type hints on all functions
- Docstrings explaining mathematical basis
- Error handling and logging
- Unit test examples
- Configuration files (YAML)
- Visualization of results (3D models, depth maps)

Start with geometric_depth.py as the core, then build other components around it.

---

## ALTERNATIVE PROMPT (Ultra-Concise)

I need to build a pothole depth detection system:

**Method:** YOLO (detection) + SfM (COLMAP, 3D) + Custom geometry (angles + camera height → absolute depth)

**Innovation:** Use geometric triangulation to scale SfM reconstruction (no GPS needed)

**Formula:** depth = camera_height / tan(angle_below_horizontal)

**Data:** Mendeley Pothole Dataset (downloaded)

**Target:** ±5-8% depth accuracy

Create modular, production-grade Python code:
- yolo_detector.py (pothole detection)
- geometric_depth.py (angle→distance math model)
- sfm_runner.py (COLMAP wrapper)
- hybrid_estimator.py (geometry + SfM fusion)
- main.py (end-to-end pipeline)

Full type hints, docstrings, error handling, config files, validation metrics.

---

END QUICK PROMPT
