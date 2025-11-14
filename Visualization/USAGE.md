# Quick Start Guide

## 1. Try the Sample Data

The easiest way to get started is to use the provided sample trajectory:

1. Open `camera_trajectory_3d.html` in your web browser (Chrome/Firefox recommended)
2. Click "Choose File" and select `sample_trajectory.txt`
3. Explore the 3D visualization!

## 2. Load Your Own Data

### From MAC-VO Experiment Results

After running a MAC-VO experiment, you can export the trajectory:

```bash
# Export to TXT format (human-readable)
python Visualization/export_trajectory.py \
    --sandbox /path/to/your/experiment/sandbox \
    --output my_trajectory.txt

# Or export to NPY format (binary, smaller file size)
python Visualization/export_trajectory.py \
    --sandbox /path/to/your/experiment/sandbox \
    --output my_trajectory.npy
```

### From Custom Data

If you have pose data in NumPy format:

```python
import numpy as np

# Your poses: (N, 7) array with [tx, ty, tz, qx, qy, qz, qw]
# or (N, 8) array with [timestamp, tx, ty, tz, qx, qy, qz, qw]
poses = ...  # Your pose data

# Save it
np.save('my_trajectory.npy', poses)
```

Or create a TXT file manually:

```
# Format: FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ
0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
1, 0, 0.999, 0.001, 0.002, 0.001, 0.1, 0.05, 0.02
...
```

## 3. Using the Visualizer

### Basic Controls

**Mouse Navigation:**
- Left-click + drag: Rotate view
- Right-click + drag: Pan view
- Scroll wheel: Zoom in/out

**Playback:**
1. Click "▶ Play" to start animation
2. Click "⏸ Pause" to pause
3. Click "↺ Reset" to go back to start
4. Adjust speed with the "Speed" slider (1-100 FPS)

**Frame Selection:**
1. Set "Start Frame" and "End Frame"
2. Click "Apply Range"
3. The visualizer will only show/play the selected range
4. "Range Distance" shows the distance traveled in that range

### Advanced Features

**Toggle Visibility:**
- Click "📷 Cameras" to show/hide camera markers
- Click "📐 Axes" to show/hide coordinate axes

**Reset View:**
- Click "🔄 Reset View" to automatically fit the trajectory in the viewport

**Distance Analysis:**
- "Total Distance" shows the cumulative trajectory length
- "Range Distance" updates based on your frame selection
- Use this to analyze specific segments of your trajectory

**Pose Information:**
- The "Current Pose" panel shows real-time data for the current frame
- Includes position (X, Y, Z) and quaternion (W, X, Y, Z)

## 4. Common Use Cases

### Analyzing a Specific Segment

1. Load your trajectory
2. Use the frame slider or set start/end frames to find the segment
3. Click "Apply Range" to isolate it
4. Check "Range Distance" to see the distance traveled
5. Use playback to review the motion

### Comparing Different Runs

1. Load first trajectory and take a screenshot
2. Load second trajectory and take another screenshot
3. Compare visually or use image comparison tools

(Future versions may support multiple trajectory overlay)

### Quality Control

1. Load your estimated trajectory
2. Play through the frames
3. Look for:
   - Smooth motion (sudden jumps indicate errors)
   - Consistent camera orientation
   - Reasonable distances between frames

## 5. Troubleshooting

**File won't load:**
- Check the file format (should be .txt or .npy)
- For TXT files, ensure format matches: `FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ`
- For NPY files, ensure shape is (N, 7) or (N, 8)

**Visualization looks wrong:**
- Click "🔄 Reset View" to refit the camera
- Check coordinate system matches your data
- Verify quaternion order (should be QW, QX, QY, QZ for TXT or QX, QY, QZ, QW for NPY)

**Performance issues:**
- Reduce playback speed
- Use Chrome or Firefox for best performance
- Close other browser tabs

## 6. Tips & Tricks

- **Large trajectories:** The visualizer automatically samples markers for display efficiency
- **Smooth playback:** Adjust speed based on your hardware (30 FPS is usually smooth)
- **Distance validation:** Use known distances in your environment to validate measurements
- **Frame-by-frame:** Use the frame slider for precise frame selection

## Need More Help?

- Read the full documentation: [`README.md`](README.md)
- Check the MAC-VO repository for issues and discussions
- Try the sample data first to ensure the tool works correctly
