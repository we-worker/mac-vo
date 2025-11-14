# Camera Trajectory 3D Visualization

An interactive web-based application for visualizing and analyzing camera pose trajectories in 3D space.

## Features

### 📁 Data Loading
- **File Upload**: Supports both `.txt` and `.npy` file formats
- **TXT Format**: CSV-like format with columns: `FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ`
- **NPY Format**: NumPy binary format with shape `(N, 7)` or `(N, 8)` containing SE3 poses

### 🎨 3D Visualization
- **Interactive Scene**: Built with Three.js for smooth 3D rendering
- **Camera Path**: Visualized as a connected line showing the trajectory
- **Camera Markers**: Simplified camera geometries showing pose at keyframes
- **Current Camera**: Highlighted 3D camera model showing current frame position and orientation
- **Grid & Axes**: Reference grid and coordinate axes for spatial orientation

### 🎮 Controls
- **Mouse Controls**:
  - Left-click + drag: Rotate view
  - Right-click + drag: Pan view
  - Scroll wheel: Zoom in/out
- **Camera Visibility**: Toggle camera markers on/off
- **Axes Visibility**: Toggle coordinate axes on/off
- **Reset View**: Automatically fit trajectory in viewport

### 📊 Distance Calculation
- **Total Distance**: Calculates cumulative distance traveled across entire trajectory
- **Range Distance**: Calculates distance for selected frame range
- **Real-time Updates**: Distance updates as frame range changes

### ▶️ Playback Features
- **Play/Pause**: Animate through trajectory frames
- **Speed Control**: Adjustable playback speed (1-100 FPS)
- **Frame Slider**: Manually scrub through frames
- **Progress Bar**: Visual indication of playback progress
- **Reset**: Jump back to start frame

### 📍 Pose Information
Real-time display of current frame's pose data:
- Frame ID
- Position (X, Y, Z)
- Quaternion (W, X, Y, Z)

### 🎯 Frame Range Selection
- Set custom start and end frames
- Calculate distance for specific trajectory segments
- Playback within selected range

## Usage

### Opening the Application

1. Open `camera_trajectory_3d.html` in a modern web browser (Chrome, Firefox, Edge, Safari)
2. No installation or server required - it runs entirely in the browser

### Loading Data

**Option 1: TXT File Format**
```
# Format: FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ
0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
1, 0, 0.9999, 0.001, 0.002, 0.001, 0.1, 0.05, 0.02
2, 0, 0.9998, 0.002, 0.004, 0.002, 0.2, 0.10, 0.04
...
```

**Option 2: NPY File Format**
- Export from Python using `numpy.save()` with shape `(N, 7)` or `(N, 8)`
- Columns: `[tx, ty, tz, qx, qy, qz, qw]` or `[timestamp, tx, ty, tz, qx, qy, qz, qw]`

Example Python export:
```python
import numpy as np
import pypose as pp

# Assuming you have poses as SE3 tensors
poses = ...  # Shape: (N, 7) - [tx, ty, tz, qx, qy, qz, qw]
np.save('trajectory.npy', poses.numpy())
```

### Workflow

1. **Load Data**: Click "Choose File" and select your trajectory file
2. **Explore**: Use mouse to rotate, pan, and zoom the 3D view
3. **Analyze**: Check total distance and pose information
4. **Select Range**: Set start/end frames to analyze specific segments
5. **Play**: Use playback controls to animate the trajectory
6. **Export** (optional): Take screenshots of interesting views

## Data Format Details

### TXT File Format
```
FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ
```
- `FRAME_ID`: Frame index (integer)
- `RIG_ID`: Camera rig identifier (integer)
- `QW, QX, QY, QZ`: Quaternion rotation (W is scalar part)
- `TX, TY, TZ`: Translation in meters

Lines starting with `#` are treated as comments and ignored.

### NPY File Format

**Shape (N, 7)**: Direct SE3 format
```
[tx, ty, tz, qx, qy, qz, qw]
```

**Shape (N, 8)**: Timestamped SE3 format (used by MAC-VO)
```
[timestamp, tx, ty, tz, qx, qy, qz, qw]
```

## Integration with MAC-VO

This visualization tool is designed to work seamlessly with MAC-VO output:

```python
# After running MAC-VO experiment
from Utility.Sandbox import Sandbox

box = Sandbox.load("path/to/experiment/output")

# Load the poses.npy file in the visualizer
# File is located at: box.path("poses.npy")
```

The tool automatically handles the `(N, 8)` format produced by MAC-VO's `IOdometry.receive_frames()` method.

## Performance

- **Optimized Rendering**: Uses Three.js WebGL for hardware acceleration
- **Large Datasets**: Efficiently handles 1000+ frames
- **Marker Sampling**: Automatically samples markers for very long trajectories (displays ~50 markers)
- **Smooth Playback**: Adjustable frame rate up to 100 FPS

## Browser Compatibility

- ✅ Chrome/Chromium (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

Requires a browser with:
- WebGL support
- ES6+ JavaScript
- File API support

## Troubleshooting

### File Won't Load
- Check file format matches expected structure
- Ensure no corrupt or invalid data in file
- Check browser console for specific error messages

### Performance Issues
- Reduce playback speed
- Use a browser with good WebGL support
- Close other tabs/applications

### Display Issues
- Try resetting the view
- Check WebGL is enabled in browser settings
- Update graphics drivers

## Technical Details

### Dependencies (via CDN)
- **Three.js** r128: 3D rendering engine
- **OrbitControls**: Camera control system

### Architecture
- **TrajectoryData**: Data model for pose storage and calculations
- **TrajectoryVisualizer**: Three.js scene management and rendering
- **App**: UI controller and file handling

### Coordinate System
- Right-handed coordinate system
- Y-up convention (standard Three.js)
- Quaternion format: (x, y, z, w)

## Future Enhancements

Potential features for future versions:
- [ ] Export trajectory to various formats
- [ ] Screenshot/video capture
- [ ] Multiple trajectory comparison
- [ ] Velocity/acceleration visualization
- [ ] Ground truth overlay
- [ ] Error metrics display
- [ ] Custom color schemes
- [ ] Path smoothing options

## License

This visualization tool is part of the MAC-VO project and follows the same license.

## Support

For issues or questions:
1. Check this README for solutions
2. Examine browser console for errors
3. Verify file format correctness
4. Open an issue in the MAC-VO repository
