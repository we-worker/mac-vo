#!/usr/bin/env python3
"""
Export trajectory data from MAC-VO sandbox to formats compatible with the 3D visualizer.

Usage:
    python export_trajectory.py --sandbox /path/to/sandbox/folder --output trajectory.txt
    python export_trajectory.py --sandbox /path/to/sandbox/folder --output trajectory.npy --format npy
"""

import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from Utility.Sandbox import Sandbox


def export_to_txt(poses_data: np.ndarray, output_path: Path):
    """
    Export poses to TXT format: FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ
    
    Args:
        poses_data: numpy array of shape (N, 8) with [timestamp, tx, ty, tz, qx, qy, qz, qw]
        output_path: path to output txt file
    """
    with open(output_path, 'w') as f:
        f.write("# Camera Trajectory Data\n")
        f.write("# Format: FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ\n")
        f.write("# Exported from MAC-VO\n\n")
        
        for frame_id in range(poses_data.shape[0]):
            row = poses_data[frame_id]
            
            if poses_data.shape[1] == 8:
                _, tx, ty, tz, qx, qy, qz, qw = row
            elif poses_data.shape[1] == 7:
                tx, ty, tz, qx, qy, qz, qw = row
            else:
                raise ValueError(f"Unexpected pose data shape: {poses_data.shape}")
            
            rig_id = 0
            
            f.write(f"{frame_id}, {rig_id}, {qw:.6f}, {qx:.6f}, {qy:.6f}, {qz:.6f}, "
                   f"{tx:.6f}, {ty:.6f}, {tz:.6f}\n")
    
    print(f"✓ Exported {poses_data.shape[0]} poses to {output_path}")


def export_to_npy(poses_data: np.ndarray, output_path: Path):
    """
    Export poses to NPY format (can be directly loaded by visualizer)
    
    Args:
        poses_data: numpy array of shape (N, 7) or (N, 8)
        output_path: path to output npy file
    """
    np.save(output_path, poses_data)
    print(f"✓ Exported {poses_data.shape[0]} poses to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export MAC-VO trajectory data for 3D visualization"
    )
    parser.add_argument(
        '--sandbox',
        type=str,
        required=True,
        help='Path to MAC-VO sandbox folder containing poses.npy'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file path (with .txt or .npy extension)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['txt', 'npy', 'auto'],
        default='auto',
        help='Output format (auto-detect from extension by default)'
    )
    parser.add_argument(
        '--use-ref',
        action='store_true',
        help='Use reference poses (ref_poses.npy) instead of estimated poses'
    )
    
    args = parser.parse_args()
    
    sandbox_path = Path(args.sandbox)
    if not sandbox_path.exists():
        print(f"Error: Sandbox path does not exist: {sandbox_path}")
        sys.exit(1)
    
    box = Sandbox.load(sandbox_path)
    
    pose_file = "ref_poses.npy" if args.use_ref else "poses.npy"
    poses_path = box.path(pose_file)
    
    if not poses_path.exists():
        print(f"Error: {pose_file} not found in sandbox: {poses_path}")
        sys.exit(1)
    
    print(f"Loading poses from: {poses_path}")
    poses_data = np.load(poses_path)
    print(f"  Shape: {poses_data.shape}")
    print(f"  Frames: {poses_data.shape[0]}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_format = args.format
    if output_format == 'auto':
        output_format = output_path.suffix.lstrip('.')
    
    if output_format == 'txt':
        export_to_txt(poses_data, output_path)
    elif output_format == 'npy':
        export_to_npy(poses_data, output_path)
    else:
        print(f"Error: Unsupported format '{output_format}'. Use 'txt' or 'npy'")
        sys.exit(1)
    
    print("\nTo visualize:")
    print(f"  1. Open Visualization/camera_trajectory_3d.html in a web browser")
    print(f"  2. Click 'Choose File' and select: {output_path.absolute()}")


if __name__ == '__main__':
    main()
