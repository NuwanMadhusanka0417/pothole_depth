# HOW TO USE CURSOR PROMPTS FOR YOUR POTHOLE PROJECT

## Two Prompts Available

### 1. CURSOR_COMPREHENSIVE_PROMPT.md
**Best for:** Complete project setup with detailed specifications
- 1500+ lines of detailed requirements
- Mathematical formulas and explanations
- Complete code structure specifications
- Technical stack details
- All edge cases and error handling
- Use this if you want Cursor to understand EVERYTHING

**When to use:** First time, need complete project understanding

**Recommended workflow:**
```
1. Open CURSOR_COMPREHENSIVE_PROMPT.md
2. Copy entire contents
3. Paste into Cursor with: "@files Create all the project structure and core files"
4. Let Cursor analyze and ask clarifying questions
5. Provide Mendeley dataset information
6. Cursor will generate complete skeleton/boilerplate
```

### 2. CURSOR_QUICK_PROMPT.md
**Best for:** Fast implementation, already understand the concepts
- Concise and focused (~300 lines)
- Copy-paste ready
- Best for implementation, not architecture
- Use this for "build this specific component"

**When to use:** Quick components, specific modules, iterative development

**Recommended workflow:**
```
1. Copy CURSOR_QUICK_PROMPT.md (shorter version)
2. Paste into Cursor with: "@files Create the core implementation"
3. Iterate on specific components
4. Use for bug fixes and optimizations
```

---

## HOW TO USE WITH CURSOR (Step-by-Step)

### Option A: Create Entire Project at Once

```
1. Open Cursor IDE
2. Start new project or use your existing project directory
3. In Cursor chat:
   
   PASTE THIS (use comprehensive prompt):
   ================================================
   I'm building a pothole detection and depth estimation system.
   I have downloaded the Mendeley Pothole Dataset and need you to create
   a complete production-ready implementation.
   
   The system uses:
   - YOLO v8 for pothole detection
   - COLMAP for Structure-from-Motion (3D reconstruction)
   - Custom geometric triangulation for absolute metric depth
   
   [PASTE FULL CONTENT OF CURSOR_COMPREHENSIVE_PROMPT.md HERE]
   
   Please create:
   1. Full project structure with all files
   2. Core implementations with type hints and docstrings
   3. Configuration templates
   4. Basic unit tests
   ================================================
   
4. Cursor will:
   - Ask about your Mendeley dataset structure
   - Ask about hardware/GPU availability
   - Generate complete boilerplate
   - Create file structure
   
5. You fill in:
   - Mendeley dataset path
   - Camera height (measured value)
   - Expected focal length or calibration method
```

### Option B: Build Components Iteratively

```
1. Use CURSOR_QUICK_PROMPT.md for initial guidance

2. Ask Cursor to build step-by-step:
   - "Create geometric_depth.py with pixel_to_angle(), angle_to_distance(), two_frame_depth()"
   - "Create yolo_detector.py wrapper for YOLOv8"
   - "Create sfm_runner.py to integrate COLMAP"
   - "Create hybrid_estimator.py to fuse both methods"

3. For each component:
   - Ask: "Implement [function] with error handling"
   - Cursor will generate with full docstrings
   - You review and adjust camera height/calibration values

4. Once all components done:
   - Ask: "Create main.py that ties everything together"
   - Ask: "Create unit tests for geometric_depth.py"
```

### Option C: Hybrid Approach (RECOMMENDED)

```
1. Use comprehensive prompt to establish project structure
   → Cursor creates folders, configs, skeleton files

2. Use quick prompt to implement core logic
   → Cursor implements each module

3. For each file, provide specific requirements:
   - "Implement geometric_depth.py with these functions..."
   - "Add error handling for when COLMAP fails..."
   - "Create evaluation_metrics.py to measure accuracy against ground truth..."
```

---

## EFFECTIVE CURSOR PROMPTING PATTERNS

### Pattern 1: Full File Generation
```
Create a file: src/geometry/geometric_depth.py

Requirements:
- Class: GeometricDepthEstimator
- Methods:
  * pixel_to_angle(pixel_y: int) -> float
  * angle_to_distance(angle: float) -> float
  * two_frame_depth(frame1, frame2, bbox) -> dict
- Include: Type hints, docstrings, error handling
- Mathematical basis in docstrings with formulas
```

### Pattern 2: Specific Functionality
```
In hybrid_estimator.py, implement:

def recover_scale(self, geometric_distances, sfm_distances):
    """
    Calculate scale factor using multiple road points.
    
    Args:
        geometric_distances: list of distances from geometric method
        sfm_distances: list of distances from SfM
    
    Returns:
        scale_factor: median ratio for robustness
        confidence: based on variance of ratios
    
    Mathematical basis:
    λ = d_geometric / d_sfm (for each point)
    λ_final = median(λ₁, λ₂, ..., λₙ)
    """
```

### Pattern 3: Integration
```
Create main.py that:
1. Loads video from path
2. Initializes YOLODetector with Mendeley-trained weights
3. For each pothole detected:
   - Runs GeometricDepthEstimator
   - Runs SfMRunner on 5-10 frames
   - Runs scale recovery
   - Outputs final depth + confidence
4. Generates summary report with accuracy metrics
```

### Pattern 4: Debugging/Fixing
```
I'm getting [error message]. 

The issue is in [component]. 
The code currently [does this].
It should [do this instead].

Please fix the [function] to handle [edge case].
```

---

## WHAT TO TELL CURSOR ABOUT MENDELEY DATASET

When Cursor asks about the dataset, provide:

```
The Mendeley Pothole Dataset is structured as:
[Describe what you see - videos? images? bounding boxes?]

Typical structure:
- Videos folder: video_001.mp4, video_002.mp4, etc.
- Each video is 3-10 seconds of dash camera footage
- Potholes visible in frames
- [Do you have annotations/ground truth depths?]

For training YOLO:
- Extract frames from videos
- Convert to YOLO format (center_x, center_y, width, height)
- Normalize by image dimensions

I have:
- Camera height: [YOUR MEASUREMENT in meters]
- Focal length: [approximate, e.g., 500 pixels]
- OR: Checkerboard images for calibration

I need Cursor to:
1. Understand Mendeley structure
2. Create data loader for it
3. Generate YOLO training pipeline
```

---

## EXAMPLE: Using the Prompt

### Comprehensive Approach Example

```
YOU (in Cursor):
===============

I'm building a pothole detection system using YOLO, SfM, and geometric triangulation.

[PASTE ENTIRE CURSOR_COMPREHENSIVE_PROMPT.md]

Please help me implement this. Here's my setup:
- Mendeley dataset downloaded to ./data/mendeley_raw/
- Camera height: 1.52 meters (measured)
- Focal length: approximately 500 pixels (1920×1080 camera)
- I have COLMAP installed
- Target: ±5-8% depth accuracy
- Want production-ready code with full error handling


CURSOR (responds):
==================

Thanks for the detailed spec! I can see you need:

1. Project structure setup
2. Geometric depth math implementation
3. YOLO detector training
4. SfM integration
5. Hybrid fusion

I have a few questions:
- Does Mendeley dataset have bounding box annotations for YOLO training?
- Do you have ground truth depths to validate against?
- Preferred output format?

I'll start by creating:
- Full project structure
- geometric_depth.py with your mathematical model
- Data loader for Mendeley format
- Configuration templates

Ready?


YOU:
====

Yes! The Mendeley dataset has:
- Videos in ./videos/ folder
- Bounding box XML annotations in ./annotations/
- No ground truth depths (we'll measure 20-30 manually for validation)

Start with project structure and geometric_depth.py


CURSOR:
=======

[Generates complete file structure]
[Creates geometric_depth.py with full implementation]
[Creates requirements.txt, config.yaml]
[Provides next steps]

```

---

## TIPS FOR SUCCESS

### DO
✓ Be specific about your requirements
✓ Provide mathematical formulas when relevant
✓ Explain why (SfM for detail, geometry for scale)
✓ Ask for error handling and edge cases
✓ Request type hints and docstrings
✓ Ask for unit tests or examples
✓ Request visualization/debugging helpers

### DON'T
✗ Be vague about algorithms
✗ Skip explaining the math
✗ Ask for everything in one message (break into steps)
✗ Ignore Cursor's clarifying questions
✗ Assume Cursor knows your dataset structure
✗ Skip configuration/requirements

---

## VERIFICATION CHECKLIST

After Cursor generates code, verify:

```
✓ All imports present
✓ Type hints on functions
✓ Docstrings with formula explanations
✓ Error handling with try/except
✓ Logging statements (DEBUG, INFO, WARNING)
✓ Configuration file support
✓ Unit tests or examples
✓ README with usage examples
✓ Requirements.txt complete
✓ Modular (each file independent)
```

---

## NEXT ACTIONS

1. **Immediately:**
   - Choose Comprehensive or Quick prompt
   - Copy entire prompt to Cursor
   - Paste with: "@files Please create this project"

2. **Wait for Cursor to ask questions:**
   - Describe Mendeley dataset structure
   - Provide camera measurements
   - Clarify any requirements

3. **Iterate:**
   - Cursor generates code
   - You test and provide feedback
   - Cursor refines

4. **Build incrementally:**
   - Start with geometric_depth.py
   - Add YOLO training
   - Add SfM integration
   - Add fusion/main pipeline

---

## SUCCESS CRITERIA

When done, you should have:

```
✓ src/geometry/geometric_depth.py
  - pixel_to_angle() ✓
  - angle_to_distance() ✓
  - two_frame_depth() ✓

✓ src/detection/yolo_detector.py
  - Fine-tuned on Mendeley ✓

✓ src/reconstruction/sfm_runner.py
  - COLMAP integration ✓

✓ src/fusion/hybrid_estimator.py
  - Scale recovery ✓
  - Depth computation ✓

✓ src/main.py
  - End-to-end pipeline ✓

✓ Validation
  - ±5-8% depth accuracy ✓
```

---

## FINAL CHECKLIST

Before pasting prompt to Cursor:

- [ ] You've read both prompts (comprehensive and quick)
- [ ] You have Mendeley dataset downloaded
- [ ] You know your camera height (measured)
- [ ] You have COLMAP installed (or know how to install)
- [ ] You understand the geometric formula
- [ ] You have GPU available (or OK with CPU-only)
- [ ] You're ready to iterate with Cursor

If all checked → **Ready to go!**

**Choose prompt (comprehensive if first time, quick if iterating) and paste into Cursor now.**

Good luck! 🚀
