# SLAM Loop Closing System

This project implements a Loop Closing system for SLAM (Simultaneous Localization and Mapping) as part of a computer vision course assignment.

## Overview

Loop closing is essential for SLAM systems. It identifies similar frames throughout a video sequence - when the camera returns to a previously visited location and the same scene becomes visible again. This identification is based on camera rotation and translation, and can be used to generate new feature matches and improve reconstruction quality.

## Features

- **Frame Extraction**: Extract frames from video files (.MOV)
- **Feature Detection**: Uses ORB (Oriented FAST and Rotated BRIEF) features for robust detection
- **Feature Matching**: BFMatcher with Hamming distance for efficient matching
- **Pose Estimation**: Computes relative camera pose using Essential Matrix decomposition
- **Loop Detection**: Automatically detects when the camera revisits previous locations
- **3D Reconstruction**: Triangulates 3D points from matched features
- **Visualization**: Generates visualizations of matches and loop closures

## Requirements

- CMake 3.10 or higher
- C++17 compiler (Clang/GCC)
- OpenCV 4.x
- macOS, Linux, or Windows

## Project Structure

```
SLAM-Loop-Closing/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── data/
│   ├── IMG_0242.MOV        # Input video
│   ├── extracted_frames/    # Extracted frames (generated)
│   └── loop_closing_results/ # Results and visualizations (generated)
├── include/
│   ├── extract_images.hpp   # Frame extraction header
│   └── loop_closing.hpp     # Loop closing system header
└── src/
    ├── extract_images_from_mov.cpp  # Frame extraction implementation
    ├── loop_closing.cpp     # Loop closing system implementation
    └── main.cpp             # Main program
```

## Building the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

The program supports three modes:

### 1. Extract Frames Only

```bash
./build/LoopClosing extract
```

This extracts all frames from the video file at `data/IMG_0242.MOV` and saves them to `data/extracted_frames/`.

### 2. Run Loop Closing Only

```bash
./build/LoopClosing loop
```

This runs the loop closing algorithm on previously extracted frames. Make sure frames have been extracted first.

### 3. Run Both (Extract + Loop Closing)

```bash
./build/LoopClosing all
```

This runs both extraction and loop closing in sequence.

### Default Behavior

Running without arguments defaults to loop closing mode:

```bash
./build/LoopClosing
```

## Algorithm Details

### Workflow (Arbeitsschritte)

1. **Video Capture**: Record a suitable video sequence (can be done offline) ✓
2. **Sequential Processing**: 
   - Pairwise feature detection between consecutive frames
   - Feature matching using ORB descriptors
   - Camera pose estimation via Essential Matrix
   - 3D reconstruction through triangulation
   - Loop closure check after each frame ✓
3. **Loop Closure Matching**: Re-match features on identified loop frames ✓
4. **Reconstruction Update**: Repeat triangulation with additional matches from loops ✓

### Parameters

You can adjust the following parameters in `src/main.cpp`:

- `loop_threshold`: Similarity threshold for loop detection (default: 0.15)
- `min_loop_gap`: Minimum frame gap for considering loop closure (default: 30)
- `frame_skip`: Process every Nth frame to speed up computation (default: 3)

### Feature Detection

- **Detector**: ORB with 2000 maximum features per frame
- **Descriptor**: 256-bit binary descriptor
- **Matching**: Brute-force matching with Hamming distance
- **Filtering**: Distance-based filtering (threshold: 2× minimum distance)

### Loop Detection

Loop closure is detected when:
1. Current frame is compared against all frames at least `min_loop_gap` frames ago
2. Feature match similarity exceeds `loop_threshold`
3. At least 50 good matches are found

Similarity score = (number of matches) / min(features in frame1, features in frame2)

### Pose Estimation

- Essential Matrix computation using RANSAC
- Pose recovery from Essential Matrix
- Minimum 8-point correspondence required

### 3D Reconstruction

- Triangulation using projection matrices
- Camera intrinsics assumed (fx=fy=800, cx=640, cy=360)
- Point filtering: rejects points behind camera or too far (>100 units)

## Output

Results are saved to `data/loop_closing_results/`:

- `loop_closures.txt`: Text file listing all detected loop closures with statistics
- `matches_X_Y.png`: Visualizations of matches between consecutive frames (every 10th frame)
- `loop_X_Y.png`: Visualizations of matches for each detected loop closure

### Example Output

```
=== Processing Complete ===
Total frames processed: 97
Loop closures detected: 45

Loop Closures Detected:
======================

Frame 93 <-> Frame 0
  Matches: 434
  Similarity: 0.2085

Frame 96 <-> Frame 0
  Matches: 236
  Similarity: 0.217
...
```

## Performance Notes

- Processing every 3rd frame (default) provides good balance between speed and accuracy
- Reducing image size by 50% speeds up feature detection significantly
- For long videos (>500 frames), consider increasing `frame_skip` to 5 or 10

## Troubleshooting

### "Frames directory not found"

Run frame extraction first:
```bash
./build/LoopClosing extract
```

### Slow Processing

Adjust these parameters in `main.cpp`:
- Increase `frame_skip` (e.g., to 5 or 10)
- Reduce image size more aggressively (e.g., resize to 0.33)
- Reduce ORB features (e.g., to 1000)

### No Loop Closures Detected

- Ensure your video revisits previous locations
- Try lowering `loop_threshold` (e.g., to 0.10)
- Reduce `min_loop_gap` (e.g., to 20)

## Assignment Requirements

This implementation fulfills the course requirements:

- ✅ Video sequence capture (provided as `data/IMG_0242.MOV`)
- ✅ Sequential pairwise feature detection and matching
- ✅ Pose estimation between frames
- ✅ Loop closure detection after each frame
- ✅ Additional feature matching on loop-identified frames
- ✅ Triangulation and 3D reconstruction
- ✅ Updated reconstruction with loop closure matches

## License

Academic project for computer vision course.
