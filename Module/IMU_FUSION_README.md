# IMU Fusion Implementation

This document describes the simple and clean IMU fusion scheme implemented for MAC-VO.

## Overview

The IMU fusion implementation provides two main components:

1. **SimpleIMUIntegrator**: A forward integration module that integrates IMU measurements (gyroscope and accelerometer) to estimate pose, velocity, and position changes.

2. **IMUPreintegrator**: A preintegration module for efficient multi-frame optimization, useful for factor graph-based approaches.

## Architecture

### SimpleIMUIntegrator

Located in `Module/IMUIntegration.py`, this class performs forward integration of IMU data:

```python
integrator = SimpleIMUIntegrator(gravity=9.81)
final_rot, final_vel, final_pos = integrator.integrate(
    init_rot, init_vel, init_pos, gyro, acc, dt
)
```

**Key Features:**
- Simple Euler integration for position and velocity
- Proper handling of SO(3) rotation integration
- Gravity compensation in world frame
- Numerically stable for typical IMU sampling rates (100-200 Hz)

**Integration Steps:**
1. Transform acceleration from body frame to world frame using current rotation
2. Remove gravity component from acceleration
3. Update velocity: `vel = vel + acc_world * dt`
4. Update position: `pos = pos + vel * dt + 0.5 * acc_world * dt^2`
5. Update rotation using gyroscope measurements via axis-angle representation

### IMUPreintegrator

Also in `Module/IMUIntegration.py`, this class preintegrates IMU measurements relative to an initial state:

```python
preintegrator = IMUPreintegrator(gravity=9.81)
delta_rot, delta_vel, delta_pos = preintegrator.preintegrate(gyro, acc, dt)
```

**Use Case:**
- Preintegration between keyframes for factor graph optimization
- Reduces computational cost in optimization by precomputing IMU constraints
- Can be extended to include covariance propagation for uncertainty quantification

## Motion Model Integration

### SimpleIMUMotion

Located in `Module/MotionModel.py`, this motion model uses IMU data to predict pose changes:

```python
motion_model = SimpleIMUMotion(config)
predicted_pose = motion_model.predict(stereo_inertial_frame, flow, depth)
```

**Configuration:**
```yaml
type: SimpleIMUMotion
args:
  gravity: 9.81  # Gravity constant in m/s^2
  device: cpu    # Device for computation
```

**How it Works:**
1. Receives a `StereoInertialFrame` containing both camera and IMU data
2. Extracts IMU measurements (gyro, acc) and timestamps
3. Integrates IMU data from previous pose to predict current pose
4. Can be updated with optimized poses via the `update()` method

## Usage Examples

### Basic Usage

```python
from Module.IMUIntegration import SimpleIMUIntegrator
from Module.MotionModel import SimpleIMUMotion
import pypose as pp
import torch

# Initialize integrator
integrator = SimpleIMUIntegrator(gravity=9.81)

# Prepare initial state
init_rot = pp.SO3(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # Identity rotation
init_vel = torch.zeros(3)
init_pos = torch.zeros(3)

# IMU measurements (N measurements)
gyro = torch.randn(100, 3) * 0.1  # Angular velocity (rad/s)
acc = torch.randn(100, 3) + torch.tensor([0.0, 0.0, 9.81])  # Acceleration (m/s^2)
dt = torch.ones(100) * 0.01  # 10ms between measurements

# Integrate
final_rot, final_vel, final_pos = integrator.integrate(
    init_rot, init_vel, init_pos, gyro, acc, dt
)
```

### Using with MAC-VO

1. **Configure the motion model** in your experiment YAML:

```yaml
Odometry:
  motion:
    !include Config/Module/MotionModel/SimpleIMUMotion.yaml
```

2. **Use a dataset with IMU data** (e.g., EuRoC):

```yaml
Data:
  type: EuRoC_Sequence
  args:
    root: /path/to/euroc/MH_01_easy
    gt_pose: true
```

3. **Run the system** - The motion model will automatically use IMU data when available.

## Design Principles

This implementation follows these principles for clean code:

1. **Separation of Concerns**: 
   - Integration logic is separate from motion model logic
   - Each class has a single, well-defined responsibility

2. **Type Safety**:
   - Proper type hints throughout the code
   - Uses PyPose's LieTensor types for geometrically correct operations

3. **Documentation**:
   - Clear docstrings explaining parameters and return values
   - Examples showing typical usage patterns

4. **Extensibility**:
   - Easy to extend with bias estimation
   - Can add covariance propagation without major refactoring
   - Preintegrator is ready for factor graph integration

5. **Numerical Stability**:
   - Checks for near-zero rotations
   - Proper handling of edge cases
   - Uses appropriate numerical thresholds

## Limitations and Future Improvements

### Current Limitations

1. **No Bias Estimation**: Does not estimate gyroscope or accelerometer biases
2. **Simple Integration**: Uses Euler integration (could use RK4 for better accuracy)
3. **No Covariance**: Does not propagate uncertainty through integration
4. **No Calibration**: Assumes pre-calibrated IMU measurements

### Potential Improvements

1. **Add Online Bias Estimation**:
   ```python
   class BiasEstimatingIMUIntegrator(SimpleIMUIntegrator):
       def __init__(self, gravity: float = 9.81):
           super().__init__(gravity)
           self.gyro_bias = torch.zeros(3)
           self.acc_bias = torch.zeros(3)
   ```

2. **Add Covariance Propagation**:
   ```python
   def integrate_with_covariance(self, ...):
       # Propagate covariance using Jacobians
       # Useful for factor graph optimization
       pass
   ```

3. **Use Higher-Order Integration**:
   - Implement RK4 or other higher-order methods
   - Trade-off between accuracy and computational cost

4. **Add IMU-Visual Fusion**:
   - Tightly couple IMU with visual features
   - Use IMU for feature prediction and tracking

## Testing

To verify the IMU integration works correctly:

1. **Unit Test** (TODO):
   ```python
   def test_static_integration():
       # IMU at rest should maintain initial pose
       pass
   
   def test_pure_rotation():
       # Gyro-only should produce pure rotation
       pass
   ```

2. **Integration Test**:
   - Run on EuRoC dataset
   - Compare with ground truth trajectory
   - Verify pose estimates are reasonable

## References

- PyPose Documentation: https://pypose.org/
- IMU Preintegration: Forster et al., "IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation"
- Lie Group Integration: Sola et al., "A micro Lie theory for state estimation in robotics"

## Contributing

When extending this module:

1. Maintain clean code structure
2. Add comprehensive docstrings
3. Include type hints
4. Write unit tests for new functionality
5. Update this README with new features
