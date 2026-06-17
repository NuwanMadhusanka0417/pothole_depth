# CURSOR PROMPTS - FINAL SUMMARY & ACTION ITEMS

## What You Have

```
Three professionally crafted Cursor prompts for your pothole project:

1. CURSOR_COMPREHENSIVE_PROMPT.md
   - 1500+ lines of detailed specifications
   - Complete technical requirements
   - Mathematical formulas included
   - Full code structure planned
   - Best for: Complete project setup

2. CURSOR_QUICK_PROMPT.md
   - 300-400 lines, concise
   - Copy-paste ready
   - Core concepts explained
   - Best for: Fast implementation, iterative work

3. HOW_TO_USE_CURSOR_PROMPTS.md
   - Step-by-step guide
   - Pattern examples
   - Tips for effective prompting
   - Verification checklist
```

---

## YOUR TECHNOLOGY STACK

```
Detection:        YOLOv8 (fine-tuned on Mendeley dataset)
3D Reconstruction: COLMAP (Structure-from-Motion)
Depth Math:       Custom geometric triangulation
Fusion:           Geometric distances scale SfM reconstruction

Key Innovation:   Camera height acts as absolute scale reference
                  NO GPS needed
                  ±5-8% accuracy target
```

---

## QUICK START (DO THIS NOW)

### Step 1: Choose Your Prompt
```
If first time with Cursor:
  → Use CURSOR_COMPREHENSIVE_PROMPT.md

If experienced with Cursor:
  → Use CURSOR_QUICK_PROMPT.md

If unsure:
  → Use CURSOR_QUICK_PROMPT.md (faster)
```

### Step 2: Prepare Your Information
```
Have ready:
- Camera height measurement (in meters)
- Approximate focal length (or "don't know, will calibrate")
- Mendeley dataset location path
- GPU available? (yes/no)
- Target accuracy: ±5-8%
```

### Step 3: Copy & Paste to Cursor
```
1. Open Cursor IDE
2. Create new project (or open existing)
3. Open the prompt file
4. Copy ALL content
5. Paste into Cursor chat
6. Add context: "I have Mendeley dataset at [path]. Camera height: [measurement]"
7. Press Enter
8. Answer Cursor's clarifying questions
```

### Step 4: Cursor Will Generate
```
- Complete project structure
- All Python files with full implementations
- Configuration templates
- Requirements.txt
- Basic examples
```

### Step 5: Iterate
```
- Review generated code
- Point out any issues
- Ask for improvements
- Test and validate
```

---

## THE MATHEMATICAL CORE

Your innovation is this formula (explain to Cursor clearly):

```
Geometric Depth Calculation:

Single Frame:
  depth = h / tan(θ_road) - h / tan(θ_pothole)

Two-Frame Triangulation (recommended):
  Frame 1: depth₁
  Frame 2: depth₂
  depth_final = (depth₁ + depth₂) / 2
  confidence = 1 / (1 + variance)

Where:
  h = camera_height (meters) - FIXED, measured once
  θ = angle below horizontal = arctan((pixel_y - cy) / focal_length)
  
KEY: h is the absolute scale reference
     No GPS needed
     All depths automatically in meters
```

---

## MENDELEY DATASET NOTE

When Cursor asks about the dataset, explain:

```
"The Mendeley Pothole Dataset contains:
- Videos of dash camera footage (3-10 seconds each)
- Visible potholes in the road
- [Bounding boxes / Annotations / Ground truth depth - describe what you have]

I need you to:
1. Create a data loader for this format
2. Extract frames from videos
3. Convert to YOLO training format if needed
4. Process frames for SfM
5. Handle both single images and video sequences"
```

---

## CORE COMPONENTS TO BUILD

### 1. geometric_depth.py (MOST IMPORTANT)
```python
class GeometricDepthEstimator:
    def pixel_to_angle(pixel_y) → float
    def angle_to_distance(angle) → float
    def single_frame_depth(frame, bbox) → depth
    def two_frame_depth(frame1, frame2, bbox) → depth
    def multi_frame_validation(frames, bbox) → depth_with_confidence
```

### 2. yolo_detector.py
```python
class YOLODetector:
    def train(mendeley_dataset)
    def detect(frame) → [pothole_boxes]
    def process_video(video_path) → detections_over_time
```

### 3. sfm_runner.py
```python
class SfMRunner:
    def run_colmap(frames) → 3d_points
    def get_camera_poses() → poses
    def triangulate() → point_cloud
```

### 4. hybrid_estimator.py
```python
class HybridEstimator:
    def recover_scale(geometric_dist, sfm_dist) → scale_factor
    def estimate_depth_hybrid() → final_depth_with_confidence
    def generate_depth_map() → 3d_model
```

### 5. main.py
```python
def main():
    1. Load video
    2. Detect potholes (YOLO)
    3. For each pothole:
       - Geometric method
       - SfM method
       - Scale recovery
       - Fusion
    4. Output results
```

---

## WHAT CURSOR WILL DO FOR YOU

✓ Create complete file structure
✓ Implement all classes and functions
✓ Add type hints and docstrings
✓ Include error handling
✓ Write configuration files
✓ Create basic unit tests
✓ Provide usage examples
✓ Add logging throughout

---

## WHAT YOU NEED TO DO

1. ✓ Download Mendeley dataset (done)
2. ✓ Measure camera height (do this before Cursor)
3. ✓ Prepare one prompt file
4. ✓ Paste to Cursor
5. ✓ Answer Cursor's questions
6. ✓ Test generated code
7. ✓ Adjust as needed
8. ✓ Validate accuracy on 20-30 manual measurements

---

## SUCCESS TIMELINE

```
Day 1: Setup with Cursor
  - Copy prompt to Cursor
  - Generate project structure
  - First set of files created

Day 2-3: Implementation
  - Cursor builds each module
  - You test and iterate
  - Fix any issues

Day 4-5: Integration & Testing
  - Connect all components
  - Test on Mendeley dataset
  - Measure accuracy

Week 2+: Refinement
  - Improve accuracy
  - Add more features
  - Production hardening
```

---

## KEY PARAMETERS TO MEASURE/PROVIDE

```
CRITICAL (measure/find before using Cursor):
- camera_height_m: 1.52 (example - yours will be different)
- focal_length_px: 500 (from calibration or estimate)
- principal_point: [width/2, height/2] (image center)

OPTIONAL (Cursor can help find):
- Mendeley dataset path
- COLMAP installation path
- GPU availability
- Ground truth depths (for validation)
```

---

## TROUBLESHOOTING

### If Cursor doesn't understand the math:
```
"The core formula is: depth = camera_height / tan(angle_to_pothole)

This is basic trigonometry. 
The angle comes from pixels: θ = arctan((pixel_y - cy) / focal_length)

Can you implement this function and explain the math in docstrings?"
```

### If Cursor asks which framework:
```
"Use PyTorch for YOLO (via ultralytics library).
Use subprocess to call COLMAP (it's an external tool).
Use OpenCV for image processing and feature tracking.
Use NumPy for all mathematical operations."
```

### If Cursor generates incomplete code:
```
"Please add error handling for [specific case].
Include logging statements with DEBUG, INFO, WARNING levels.
Add docstrings with mathematical explanation.
Include unit tests."
```

---

## AFTER CURSOR GENERATES CODE

### Validation Checklist
```
Code Quality:
✓ All imports present
✓ Type hints on all functions
✓ Docstrings with examples
✓ Error handling (try/except)
✓ Logging statements
✓ Configuration files

Functionality:
✓ geometric_depth.py works correctly
✓ YOLO detection runs on Mendeley data
✓ SfM integrates with COLMAP
✓ Scale recovery works
✓ Fusion produces final depth

Testing:
✓ Can process single frame
✓ Can process video sequence
✓ Can validate with ground truth
✓ Accuracy matches targets (±5-8%)
```

---

## FILE ORGANIZATION IN YOUR PROJECT

```
After Cursor creates everything, you should have:

pothole_detection/
├── data/
│   ├── mendeley_raw/           ← Your downloaded dataset
│   ├── processed/              ← Processed by code
│   └── ground_truth.csv        ← Your measurements
├── src/
│   ├── detection/
│   │   ├── yolo_detector.py
│   │   ├── train_yolo.py
│   │   └── inference.py
│   ├── geometry/
│   │   ├── geometric_depth.py  ← CORE ALGORITHM
│   │   ├── calibration.py
│   │   └── angle_utils.py
│   ├── reconstruction/
│   │   ├── sfm_runner.py
│   │   ├── feature_tracking.py
│   │   └── scale_recovery.py
│   ├── fusion/
│   │   ├── hybrid_estimator.py
│   │   ├── confidence_scoring.py
│   │   └── depth_map_generator.py
│   ├── utils/
│   │   ├── data_loader.py
│   │   ├── camera_calibration.py
│   │   └── evaluation_metrics.py
│   └── main.py
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── geometric_demo.ipynb
│   └── accuracy_analysis.ipynb
├── configs/
│   ├── camera_calib.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── tests/
│   ├── test_geometric_depth.py
│   └── test_sfm_runner.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## FINAL CHECKLIST BEFORE CURSOR

- [ ] Read both prompt files (comprehensive and quick)
- [ ] Downloaded Mendeley dataset
- [ ] Know your camera height (measured in meters)
- [ ] Have COLMAP installed or know how to install
- [ ] Understand the geometric formula (depth = h/tan(θ))
- [ ] Ready to iterate with Cursor
- [ ] Have about 4-6 hours for initial setup
- [ ] Can measure 20-30 potholes manually for validation

**If all checked: Ready to go!**

---

## WHICH PROMPT TO USE?

### Use COMPREHENSIVE if:
- First time with Cursor
- Want Cursor to explain everything
- Need detailed technical specs
- Want to understand full architecture first
- Have time to read/review everything

### Use QUICK if:
- Experienced with Cursor
- Know what you need
- Want faster iteration
- Already understand the concepts
- Want to build incrementally

---

## NEXT STEP RIGHT NOW

1. Open `/mnt/user-data/outputs/CURSOR_QUICK_PROMPT.md` (or COMPREHENSIVE)
2. Copy entire content
3. Open Cursor IDE
4. Paste into chat
5. Add: "I have Mendeley dataset. Camera height: [your measurement]m"
6. Press Enter

**That's it! Cursor will take it from there.**

---

## GOOD LUCK! 🚀

You have everything you need:
- ✓ Complete algorithm design
- ✓ Mathematical foundation
- ✓ Dataset guides
- ✓ Carefully crafted prompts
- ✓ Implementation instructions

**Now go build something amazing!**
