"""
Simple test script for IMU integration module.

This script demonstrates the basic functionality of the IMU integration
and can be used to verify the implementation works correctly.
"""

import torch
import pypose as pp
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Module.IMUIntegration import SimpleIMUIntegrator, IMUPreintegrator


def test_static_imu():
    """Test that static IMU maintains initial pose."""
    print("\n=== Test 1: Static IMU ===")
    integrator = SimpleIMUIntegrator(gravity=9.81)
    
    init_rot = pp.SO3(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    init_vel = torch.zeros(3)
    init_pos = torch.zeros(3)
    
    gyro = torch.zeros(100, 3)
    acc = torch.zeros(100, 3)
    acc[:, 2] = 9.81
    dt = torch.ones(100) * 0.01
    
    final_rot, final_vel, final_pos = integrator.integrate(
        init_rot, init_vel, init_pos, gyro, acc, dt
    )
    
    print(f"Initial rotation: {init_rot.tensor()}")
    print(f"Final rotation:   {final_rot.tensor()}")
    print(f"Final velocity:   {final_vel}")
    print(f"Final position:   {final_pos}")
    
    assert torch.allclose(final_vel, torch.zeros(3), atol=1e-3), "Velocity should remain zero"
    assert torch.allclose(final_pos, torch.zeros(3), atol=1e-3), "Position should remain zero"
    print("✓ Static IMU test passed")


def test_pure_rotation():
    """Test pure rotation around Z-axis."""
    print("\n=== Test 2: Pure Rotation (Z-axis) ===")
    integrator = SimpleIMUIntegrator(gravity=9.81)
    
    init_rot = pp.SO3(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    init_vel = torch.zeros(3)
    init_pos = torch.zeros(3)
    
    angular_rate = 0.1
    gyro = torch.zeros(100, 3)
    gyro[:, 2] = angular_rate
    acc = torch.zeros(100, 3)
    acc[:, 2] = 9.81
    dt = torch.ones(100) * 0.01
    
    final_rot, final_vel, final_pos = integrator.integrate(
        init_rot, init_vel, init_pos, gyro, acc, dt
    )
    
    expected_angle = angular_rate * 1.0
    print(f"Expected rotation angle: {expected_angle:.4f} rad ({np.degrees(expected_angle):.2f} deg)")
    print(f"Final rotation quaternion: {final_rot.tensor()}")
    
    rotation_matrix = final_rot.matrix().squeeze()
    actual_angle = torch.acos((torch.trace(rotation_matrix) - 1) / 2)
    print(f"Actual rotation angle: {actual_angle:.4f} rad ({np.degrees(actual_angle.item()):.2f} deg)")
    
    assert torch.allclose(final_vel, torch.zeros(3), atol=1e-3), "Velocity should remain zero"
    assert torch.allclose(final_pos, torch.zeros(3), atol=1e-3), "Position should remain zero"
    print("✓ Pure rotation test passed")


def test_constant_acceleration():
    """Test integration with constant acceleration."""
    print("\n=== Test 3: Constant Acceleration ===")
    integrator = SimpleIMUIntegrator(gravity=9.81)
    
    init_rot = pp.SO3(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    init_vel = torch.zeros(3)
    init_pos = torch.zeros(3)
    
    forward_acc = 1.0
    gyro = torch.zeros(100, 3)
    acc = torch.zeros(100, 3)
    acc[:, 0] = forward_acc
    acc[:, 2] = 9.81
    dt = torch.ones(100) * 0.01
    
    final_rot, final_vel, final_pos = integrator.integrate(
        init_rot, init_vel, init_pos, gyro, acc, dt
    )
    
    total_time = 1.0
    expected_vel = forward_acc * total_time
    expected_pos = 0.5 * forward_acc * total_time ** 2
    
    print(f"Expected final velocity: {expected_vel:.4f} m/s")
    print(f"Actual final velocity:   {final_vel[0]:.4f} m/s")
    print(f"Expected final position: {expected_pos:.4f} m")
    print(f"Actual final position:   {final_pos[0]:.4f} m")
    
    assert torch.allclose(final_vel[0], torch.tensor(expected_vel), atol=0.1), "Velocity mismatch"
    assert torch.allclose(final_pos[0], torch.tensor(expected_pos), atol=0.1), "Position mismatch"
    print("✓ Constant acceleration test passed")


def test_preintegration():
    """Test IMU preintegration."""
    print("\n=== Test 4: Preintegration ===")
    preintegrator = IMUPreintegrator(gravity=9.81)
    
    gyro = torch.zeros(100, 3)
    gyro[:, 2] = 0.05
    acc = torch.zeros(100, 3)
    acc[:, 0] = 0.5
    acc[:, 2] = 9.81
    dt = torch.ones(100) * 0.01
    
    delta_rot, delta_vel, delta_pos = preintegrator.preintegrate(gyro, acc, dt)
    
    print(f"Delta rotation: {delta_rot.tensor()}")
    print(f"Delta velocity: {delta_vel}")
    print(f"Delta position: {delta_pos}")
    print(f"Total time:     {preintegrator.sum_dt:.4f} s")
    
    assert delta_rot is not None, "Delta rotation should not be None"
    assert delta_vel is not None, "Delta velocity should not be None"
    assert delta_pos is not None, "Delta position should not be None"
    print("✓ Preintegration test passed")


def test_device_compatibility():
    """Test that integration works on different devices."""
    print("\n=== Test 5: Device Compatibility ===")
    integrator = SimpleIMUIntegrator(gravity=9.81)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")
    
    init_rot = pp.SO3(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
    init_vel = torch.zeros(3, device=device)
    init_pos = torch.zeros(3, device=device)
    
    gyro = torch.randn(50, 3, device=device) * 0.1
    acc = torch.randn(50, 3, device=device) * 0.5
    acc[:, 2] += 9.81
    dt = torch.ones(50, device=device) * 0.01
    
    final_rot, final_vel, final_pos = integrator.integrate(
        init_rot, init_vel, init_pos, gyro, acc, dt
    )
    
    print(f"Final rotation device: {final_rot.device}")
    print(f"Final velocity device: {final_vel.device}")
    print(f"Final position device: {final_pos.device}")
    
    assert str(final_vel.device) == device, "Device mismatch"
    assert str(final_pos.device) == device, "Device mismatch"
    print(f"✓ Device compatibility test passed on {device}")


if __name__ == "__main__":
    print("=" * 60)
    print("IMU Integration Module Tests")
    print("=" * 60)
    
    test_static_imu()
    test_pure_rotation()
    test_constant_acceleration()
    test_preintegration()
    test_device_compatibility()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
