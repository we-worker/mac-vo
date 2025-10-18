"""
Demonstration script for IMU-Visual Fusion in MAC-VO.

This script shows how to use the IMUVisualFusion motion model
with stereo-inertial data to achieve robust pose estimation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pypose as pp
import numpy as np
from types import SimpleNamespace

from Module.MotionModel import IMUVisualFusion, SimpleIMUMotion
from Module.IMUIntegration import SimpleIMUIntegrator
from DataLoader.Interface import StereoInertialFrame, StereoData, IMUData


def create_synthetic_stereo_imu_data(frame_idx: int, pose_gt: pp.LieTensor) -> StereoInertialFrame:
    """
    Create synthetic stereo-inertial frame for testing.
    
    Args:
        frame_idx: Frame index
        pose_gt: Ground truth pose
        
    Returns:
        StereoInertialFrame with synthetic data
    """
    H, W = 480, 640
    
    # Create synthetic stereo images
    imageL = torch.randn(1, 3, H, W) * 0.1 + 0.5
    imageR = imageL + torch.randn(1, 3, H, W) * 0.05
    
    # Camera intrinsics
    K = torch.tensor([[500.0, 0.0, 320.0],
                      [0.0, 500.0, 240.0],
                      [0.0, 0.0, 1.0]]).unsqueeze(0)
    
    baseline = torch.tensor([0.1])
    
    # Create stereo data
    stereo = StereoData(
        T_BS=pp.identity_SE3(1),
        K=K,
        baseline=baseline,
        time_ns=[frame_idx * 33333333],  # ~30 Hz
        height=H,
        width=W,
        imageL=imageL,
        imageR=imageR
    )
    
    # Create synthetic IMU data (100 Hz, so ~3 samples between frames at 30 Hz)
    num_imu_samples = 3
    gyro = torch.randn(1, num_imu_samples, 3) * 0.05  # Small angular velocity
    acc = torch.randn(1, num_imu_samples, 3) * 0.2   # Small acceleration
    acc[:, :, 2] += 9.81  # Add gravity
    
    imu_time_ns = torch.linspace(
        frame_idx * 33333333,
        (frame_idx + 1) * 33333333,
        num_imu_samples + 1,
        dtype=torch.long
    )[:-1].unsqueeze(0).unsqueeze(-1)
    
    imu = IMUData(
        T_BS=pp.identity_SE3(1),
        time_ns=imu_time_ns,
        gravity=[9.81],
        acc=acc,
        gyro=gyro
    )
    
    # Create frame
    frame = StereoInertialFrame(
        idx=[frame_idx],
        time_ns=[frame_idx * 33333333],
        gt_pose=pose_gt.unsqueeze(0) if pose_gt is not None else None,
        stereo=stereo,
        imu=imu
    )
    
    return frame


def demo_imu_only_motion():
    """Demonstrate IMU-only motion prediction."""
    print("\n" + "="*70)
    print("Demo 1: IMU-Only Motion Model")
    print("="*70)
    
    config = SimpleNamespace(
        gravity=9.81,
        device="cpu"
    )
    
    motion_model = SimpleIMUMotion(config)
    
    current_pose = pp.identity_SE3()
    trajectory = [current_pose.tensor().numpy()]
    
    print("\nRunning IMU-only odometry for 10 frames...")
    for i in range(10):
        frame = create_synthetic_stereo_imu_data(i, current_pose)
        predicted_pose = motion_model.predict(frame, flow=None, depth=None)
        motion_model.update(predicted_pose)
        
        trajectory.append(predicted_pose.tensor().numpy())
        
        if i % 3 == 0:
            pos = predicted_pose.translation()
            print(f"  Frame {i:2d}: Position = [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
    
    print(f"\n✓ IMU-only trajectory generated with {len(trajectory)} poses")


def demo_imu_visual_fusion():
    """Demonstrate IMU-Visual fusion motion prediction."""
    print("\n" + "="*70)
    print("Demo 2: IMU-Visual Fusion Motion Model")
    print("="*70)
    
    config = SimpleNamespace(
        gravity=9.81,
        device="cpu",
        imu_weight=0.3,
        visual_weight=0.7,
        forward_scale=0.5,
        use_visual=True
    )
    
    motion_model = IMUVisualFusion(config)
    
    current_pose = pp.identity_SE3()
    trajectory = [current_pose.tensor().numpy()]
    
    print("\nRunning IMU-Visual fusion odometry for 10 frames...")
    for i in range(10):
        frame = create_synthetic_stereo_imu_data(i, current_pose)
        
        # Create synthetic flow and depth
        flow = torch.randn(1, 2, 480, 640) * 2.0
        depth = torch.rand(1, 1, 480, 640) * 5.0 + 1.0
        
        predicted_pose = motion_model.predict(frame, flow=flow, depth=depth)
        motion_model.update(predicted_pose)
        
        trajectory.append(predicted_pose.tensor().numpy())
        
        if i % 3 == 0:
            pos = predicted_pose.translation()
            print(f"  Frame {i:2d}: Position = [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
    
    print(f"\n✓ IMU-Visual fusion trajectory generated with {len(trajectory)} poses")


def demo_weight_comparison():
    """Compare different fusion weights."""
    print("\n" + "="*70)
    print("Demo 3: Comparing Different Fusion Weights")
    print("="*70)
    
    weight_configs = [
        ("IMU-Heavy", 0.8, 0.2),
        ("Balanced", 0.5, 0.5),
        ("Visual-Heavy", 0.2, 0.8),
    ]
    
    for name, imu_w, vis_w in weight_configs:
        print(f"\n{name} (IMU: {imu_w}, Visual: {vis_w}):")
        
        config = SimpleNamespace(
            gravity=9.81,
            device="cpu",
            imu_weight=imu_w,
            visual_weight=vis_w,
            forward_scale=0.5,
            use_visual=True
        )
        
        motion_model = IMUVisualFusion(config)
        current_pose = pp.identity_SE3()
        
        for i in range(5):
            frame = create_synthetic_stereo_imu_data(i, current_pose)
            flow = torch.randn(1, 2, 480, 640) * 2.0
            depth = torch.rand(1, 1, 480, 640) * 5.0 + 1.0
            
            predicted_pose = motion_model.predict(frame, flow=flow, depth=depth)
            motion_model.update(predicted_pose)
        
        final_pos = predicted_pose.translation()
        print(f"  Final position: [{final_pos[0]:7.4f}, {final_pos[1]:7.4f}, {final_pos[2]:7.4f}]")


def demo_fallback_behavior():
    """Demonstrate fallback to IMU-only when visual fails."""
    print("\n" + "="*70)
    print("Demo 4: Graceful Fallback to IMU-Only")
    print("="*70)
    
    config = SimpleNamespace(
        gravity=9.81,
        device="cpu",
        imu_weight=0.3,
        visual_weight=0.7,
        forward_scale=0.5,
        use_visual=True
    )
    
    motion_model = IMUVisualFusion(config)
    current_pose = pp.identity_SE3()
    
    print("\nSimulating visual failure scenarios...")
    
    for i in range(5):
        frame = create_synthetic_stereo_imu_data(i, current_pose)
        
        # Simulate visual failure on odd frames
        if i % 2 == 1:
            print(f"  Frame {i}: Visual data unavailable -> Using IMU-only")
            flow = None
            depth = None
        else:
            print(f"  Frame {i}: Both IMU and visual available -> Using fusion")
            flow = torch.randn(1, 2, 480, 640) * 2.0
            depth = torch.rand(1, 1, 480, 640) * 5.0 + 1.0
        
        predicted_pose = motion_model.predict(frame, flow=flow, depth=depth)
        motion_model.update(predicted_pose)
    
    final_pos = predicted_pose.translation()
    print(f"\n✓ System handled visual failures gracefully")
    print(f"  Final position: [{final_pos[0]:7.4f}, {final_pos[1]:7.4f}, {final_pos[2]:7.4f}]")


if __name__ == "__main__":
    print("="*70)
    print("IMU-Visual Fusion Demonstration")
    print("="*70)
    print("\nThis demo shows the capabilities of IMU-visual fusion in MAC-VO:")
    print("1. IMU-only motion prediction")
    print("2. IMU-Visual fusion with balanced weights")
    print("3. Impact of different fusion weights")
    print("4. Graceful fallback when visual fails")
    
    try:
        demo_imu_only_motion()
        demo_imu_visual_fusion()
        demo_weight_comparison()
        demo_fallback_behavior()
        
        print("\n" + "="*70)
        print("All demonstrations completed successfully! ✓")
        print("="*70)
        print("\nNext steps:")
        print("1. Run on real data: Use EuRoC or other stereo-inertial datasets")
        print("2. Tune fusion weights: Adjust based on sensor characteristics")
        print("3. Evaluate performance: Compare with ground truth trajectories")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
