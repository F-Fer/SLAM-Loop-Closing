# SLAM Loop Closing System

A monocular visual SLAM implementation with loop closure detection and pose graph optimization, developed for a computer vision course.

## Overview

This project implements a Loop Closing system for SLAM. The system identifies similar frames throughout a video sequence – when the camera returns to a previously visited location. This detection is used to correct accumulated drift and improve 3D reconstruction.

---

## Implementation Steps

### ✅ 1. Recording a Suitable Video Sequence
Multiple video sequences were recorded with an iPhone, where the camera returns to its starting position at the end (loop). The camera was calibrated beforehand.

### ✅ 2. Sequential Feature Detection, Feature Matching, and Reconstruction
- **Feature Detection**: SIFT (Scale-Invariant Feature Transform) for robust feature detection
- **Feature Matching**: FLANN-based matching with Lowe's Ratio Test (0.8)
- **Keyframe Selection**: Dynamic selection based on parallax (20-150 pixel median displacement)
- **Pose Estimation**: 
  - Primary: PnP (Perspective-n-Point) for scale-consistent poses
  - Fallback: Essential Matrix for new scenes without 3D points
- **Triangulation**: With quality checks (depth, parallax angle, reprojection error)

### ✅ 3. Loop Closure Check After Each Frame
After complete sequence processing, the **best** loop closure is searched:
- Minimum gap: Half of the trajectory (prevents false positive detection)
- Geometric verification with Essential Matrix (RANSAC)
- Strict criteria: >200 inliers and >60% inlier ratio

### ✅ 4. Re-matching and Reconstruction
After loop closure detection:
1. **Pose Graph Optimization**: Correction of drift (rotation + translation)
2. **Re-Triangulation**: Recalculation of all 3D points with corrected poses
3. **Bundle Adjustment**: Optional refinement (skipped when loop closure is applied)

---

## Additional Implementations (Beyond Minimum Requirements)

| Feature | Description |
|---------|-------------|
| **PnP Pose Estimation** | Avoids scale drift through direct pose estimation from 2D-3D correspondences |
| **Dynamic Keyframe Selection** | Automatic selection based on parallax instead of fixed intervals |
| **Two PGO Methods** | Linear Interpolation and Gauss-Newton optimization compared |
| **Huber Loss in BA** | Robust loss function to handle outliers |
| **Camera Calibration** | Custom calibration with checkerboard pattern |
| **Visualization** | Loop closure cameras are highlighted in OBJ files (green color) |

---

## Experimental Results

### Comparison: With vs. Without Loop Closure

Tested on video `IMG_0282.MOV` (camera returns to starting position):

| Metric | Without Loop Closure | With Loop Closure (Linear) | With Loop Closure (Gauss-Newton) |
|--------|----------------------|----------------------------|----------------------------------|
| Rotation Drift | ~15-20° | ~0° | ~0° |
| Translation Drift | ~2.5 units | ~0 units | ~0 units |
| Reprojection Error | 2.8 px | 3.2 px | 3.1 px |
| Loop Cameras Overlap | ❌ No | ✅ Yes | ✅ Yes |

### Comparison: Pose Graph Optimization Methods

| Aspect | Linear Interpolation | Gauss-Newton |
|--------|---------------------|--------------|
| **Complexity** | O(n) | O(n² × iterations) |
| **Accuracy** | Good for small drift | Better for large drift |
| **Computation Time** | ~1ms | ~50-100ms |
| **Recommendation** | Real-time applications | Offline reconstruction |

### Key Insights

1. **Bundle Adjustment after Loop Closure is problematic**: BA only minimizes reprojection error and has no knowledge of loop closure constraints. In my experiments, BA partially undid the loop closure correction. **Solution**: BA is skipped when loop closure is applied.

2. **PnP is essential for scale consistency**: Without PnP, monocular SLAM accumulates scale drift because the translation from `recoverPose()` has unit norm only.

3. **Linear Interpolation is sufficient**: For typical loop closure scenarios (camera returns to start), simple linear interpolation delivers comparable results to Gauss-Newton, with significantly lower computational cost.

4. **Re-Triangulation is necessary**: After PGO, the 3D points must be recalculated because they were triangulated with the old (incorrect) poses.

---

## Project Structure

```
SLAM-Loop-Closing/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This documentation
├── data/
│   ├── calibration/            # Calibration images
│   ├── extracted_frames/       # Extracted frames (generated)
│   ├── reconstruction/         # Comparison reconstructions
│   │   ├── reconstruction_no_loop_closing.obj
│   │   ├── reconstruction_linear_interpolation.obj
│   │   └── reconstruction_gauss_newton.obj
│   └── *.MOV                   # Input videos
├── include/
│   └── extract_images.hpp      # Frame extraction header
└── src/
    ├── calibrate.cpp           # Camera calibration
    ├── extract_images_from_mov.cpp
    └── main.cpp                # Main SLAM pipeline
```

---

## Configuration Options

The following parameters can be adjusted in `src/main.cpp`:

```cpp
// Select video
std::string VIDEO_FILENAME = "IMG_0282.MOV";

// Enable/disable loop closure (for comparison)
const bool ENABLE_LOOP_CLOSURE = true;

// Pose Graph Optimization method
const PoseGraphMethod POSE_GRAPH_METHOD = PoseGraphMethod::GAUSS_NEWTON;
// Options: SIMPLE_LINEAR, GAUSS_NEWTON

// Bundle Adjustment with Huber Loss (robust against outliers)
const bool USE_HUBER_LOSS = true;
const double HUBER_DELTA = 2.0;

// Save scene points (false = camera trajectory only)
const bool SAVE_SCENE_POINTS = true;
```

---

## Build & Run

### Prerequisites
- CMake 3.10+
- C++17 Compiler
- OpenCV 4.x

### Compile

```bash
mkdir build && cd build
cmake ..
make -j4
```

### Run

```bash
# From the build directory
./LoopClosing
```

The program:
1. Extracts frames from the video (if not already present)
2. Runs the SLAM pipeline
3. Saves the result as an OBJ file in `data/reconstruction/`

### Visualization

The OBJ files can be opened with MeshLab or similar tools:
- **White points**: 3D scene points
- **Blue points**: Regular camera positions
- **Green points**: Loop closure cameras (start and end position)

---

## Algorithm Details

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend                                 │
├─────────────────────────────────────────────────────────────────┤
│  Frame → SIFT → Match → Keyframe? → PnP/Essential → Triangulate │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Loop Detection                              │
├─────────────────────────────────────────────────────────────────┤
│  For all keyframe pairs: Match → RANSAC → Find best loop        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Backend                                  │
├─────────────────────────────────────────────────────────────────┤
│  Pose Graph Optimization → Re-Triangulation → [Bundle Adjust.]  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                        OBJ Export
```

### Pose Graph Optimization

**Linear Interpolation:**
- Calculates rotation and translation drift between loop closure frames
- Distributes the correction linearly across all frames in between
- Simple and fast, but only suitable for a single loop closure

**Gauss-Newton:**
- Optimizes all poses simultaneously based on:
  - Sequential constraints (odometry)
  - Loop closure constraint (higher weighted)
- Iterative minimization of the pose graph error
- Can handle multiple loop closures (in this implementation: one)

### Bundle Adjustment

Alternating BA with Huber Loss:
1. Fix 3D points → optimize camera poses
2. Fix poses → optimize 3D points
3. Repeat for N iterations

Huber Loss reduces the influence of outliers:
```
L(r) = { 0.5 * r²           if |r| ≤ δ
       { δ * (|r| - 0.5δ)   if |r| > δ
```

---

## Comparison to ORB-SLAM

| Aspect | Our Implementation | ORB-SLAM |
|--------|-------------------|----------|
| Features | SIFT | ORB |
| Loop Detection | Brute-force Matching | Bag-of-Words (DBoW2) |
| Pose Graph | Linear / Gauss-Newton | g2o Library |
| Threading | Single-threaded | Multi-threaded (3 Threads) |
| BA + Loop | Sequential (BA skip) | Parallel with Loop Constraints |

Our implementation is simpler but demonstrates all essential concepts of loop closing.

---

## References

- [Monocular SLAM in Python (LearnOpenCV)](https://learnopencv.com/monocular-slam-in-python/)
- [ORB-SLAM Paper](https://arxiv.org/abs/1502.00956)
- OpenCV Documentation: `findEssentialMat`, `recoverPose`, `solvePnPRansac
