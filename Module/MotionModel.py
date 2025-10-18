import torch
import pypose as pp
import pypose.module as pm

from abc import ABC, abstractmethod
from pathlib import Path

from typing import Generic, cast
from types import SimpleNamespace
from Utility.Extensions import ConfigTestableSubclass,TensorQueue
from Utility.PrettyPrint import Logger
from DataLoader import StereoFrame, StereoInertialFrame, T_Data
from Utility.Timer import Timer
from .IMUIntegration import SimpleIMUIntegrator


class IMotionModel(ABC, Generic[T_Data], ConfigTestableSubclass):
    """
    A motion model class receives informations (e.g. frames, estimated flow and depth) and produce an
    initial guess to the pose of incoming frame **under global coordinate**.
    """
    def __init__(self, config: SimpleNamespace):
        self.config : SimpleNamespace = config
    
    @abstractmethod
    def predict(self, frame: T_Data, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        """
        Estimate the pose of next frame given current frame, estimated depth and flow.
        
        NOTE: returned pose should be under global coordinate!
        
        Returns
        *   pose  - 7, shaped pypose.LieTensor (SE3 ltype) under world coordinate
                  predicted pose of next frame.
        """
        ...
    
    @abstractmethod
    def update(self, pose: pp.LieTensor) -> None:
        """
        Receive a feedback (optimized pose) and may (or may not) use this method to refine next prediction.
        """
        ...


class GTMotionwithNoise(IMotionModel[StereoFrame]):
    """
    Apply GT motion with noise (can be disabled by setting `noise_std` to 0.0 in config) on previous optimized pose to predict next pose.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: pp.LieTensor | None = None
        self.prev_gt_pose: pp.LieTensor | None = None
    
    def init_context(self) -> None: return None
    
    def _stableNoiseModel(self) -> pp.LieTensor:
        if self.config.noise_std == 0.0:
            return pp.identity_SE3()
        noise: pp.LieTensor = pp.randn_SE3(sigma=self.noise_std)    #type:ignore
        return noise

    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        assert frame.gt_pose is not None
        frame_gtpose = cast(pp.LieTensor, frame.gt_pose.squeeze(0))

        if self.prev_pose is None or self.prev_gt_pose is None:
            self.prev_pose    = pp.identity_SE3()
            self.prev_gt_pose = frame_gtpose
            return pp.identity_SE3()

        gtMotion = self.prev_gt_pose.Inv() @ frame_gtpose
        gtMotion_w_noise = gtMotion @ self._stableNoiseModel()
        predict = self.prev_pose @ gtMotion_w_noise
        
        self.prev_pose = predict
        self.prev_gt_pose = frame_gtpose
        
        return predict

    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "noise_std": lambda noise: isinstance(noise, (int, float)) and noise >= 0.0
        })


class TartanMotionNet(IMotionModel[StereoFrame]):
    """
    Apply motion estimated by MotionNet adapted from TartanVO on previously optimized pose to predict next pose.
    """
    def __init__(self, config: SimpleNamespace):
        from .Network.TartanVOStereo import TartanStereoVOMotion
        
        super().__init__(config)
        self.model = TartanStereoVOMotion(self.config.weight, True, self.config.device)
        self.prev_pose = None

    @Timer.cpu_timeit("MotionModel")
    @Timer.gpu_timeit("MotionModel")
    @torch.inference_mode()
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3(device=self.config.device)
            return pp.identity_SE3(device=self.config.device)

        assert flow is not None and depth is not None, "Motion model requires flow and depth to predict motion"
        motion_se3: torch.Tensor = self.model.inference(frame, flow, depth)
        new_pose = self.prev_pose @ pp.se3(motion_se3).Exp()
        self.prev_pose = new_pose
        return new_pose
    
    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose.to(self.prev_pose.device)    #type: ignore

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "weight": lambda f: isinstance(f, str),
            "device": lambda dev: isinstance(dev, str) and (("cuda" in dev) or (dev == "cpu"))
        })


class StaticMotionModel(IMotionModel[StereoFrame]):
    """
    Assumes the camera is static and simply record and returns the pose of previous frame.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: pp.LieTensor | None = None
    
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3()
            return pp.identity_SE3()
        return self.prev_pose

    def update(self, pose: pp.LieTensor) -> None:
        assert self.prev_pose is not None
        self.prev_pose = pose.to(self.prev_pose.device) # type: ignore
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None: return


class ReadPoseFile(IMotionModel[StereoFrame]):
    """
    Use an external file of Nx7 SE3 poses as motion model output poses.
    
    NOTE: Specifically, the module will *not* output these poses directly but calculate the motion
    and apply motion on modified poses (potentially by optimizer) iteratively.
    """
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: None | pp.LieTensor = None
        self.prev_gt_pose: None | pp.LieTensor = None
        self.poses: pp.LieTensor = self.load_poses()
    
    def load_poses(self) -> pp.LieTensor:
        pose_file = Path(self.config.pose_file)
        if not pose_file.exists():
            Logger.write("error", f"Cannot read pose file at {pose_file} - File Not Exist!")
            raise FileNotFoundError(f"Cannot read pose file at {pose_file} - File Not Exist!")
        
        poses: pp.LieTensor
        match pose_file.suffix:
            case ".npy":
                import numpy as np
                poses_data: torch.Tensor = torch.from_numpy(np.load(str(pose_file)))
            case ".pt" | ".pth":
                poses_data: torch.Tensor = torch.load(str(pose_file), weights_only=False)
            case ".txt":
                import numpy as np
                poses_data: torch.Tensor = torch.from_numpy(np.loadtxt(str(pose_file)))
            case suffix:
                raise NameError(f"Cannot handle a file with suffix '{suffix}'. Consider change it to .npy/.pt/.pth/.txt or write a custom loader.")
        assert poses_data.ndim == 2 and poses_data.shape[1] == 7
        poses = pp.SE3(poses_data)
        return poses
        
    def predict(self, frame: StereoFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None or self.prev_gt_pose is None:
            self.prev_pose = pp.identity_SE3()
            self.prev_gt_pose = pp.SE3(self.poses[frame.frame_idx])
            return pp.identity_SE3()

        motion = self.prev_gt_pose.Inv() @ pp.SE3(self.poses[frame.frame_idx])
        predict = self.prev_pose @ motion
        
        self.prev_pose = predict
        self.prev_gt_pose = pp.SE3(self.poses[frame.frame_idx])
        return predict

    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "pose_file": lambda s: isinstance(s, str)
        })


class SimpleIMUMotion(IMotionModel[StereoInertialFrame]):
    """
    A simple IMU-based motion model that integrates IMU measurements
    to predict pose changes between frames.
    """
    
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: pp.LieTensor | None = None
        self.prev_velocity: torch.Tensor | None = None
        gravity = getattr(self.config, "gravity", 9.81)
        device_str = getattr(self.config, "device", "cpu")
        self.integrator = SimpleIMUIntegrator(gravity=gravity)
        self.device = torch.device(device_str)
        self.device_str = device_str
        self.dtype = torch.float32
    
    def predict(self, frame: StereoInertialFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3(device=self.device_str, dtype=self.dtype)
            self.prev_velocity = torch.zeros(3, device=self.device, dtype=self.dtype)
            return pp.identity_SE3(device=self.device_str, dtype=self.dtype)
        
        imu_data = frame.imu
        gyro = imu_data.gyro
        acc = imu_data.acc
        if gyro.numel() == 0 or acc.numel() == 0:
            Logger.write("warn", "No IMU measurements available, using previous pose")
            return self.prev_pose
        
        gyro = gyro.squeeze(0).to(device=self.device, dtype=self.dtype)
        acc = acc.squeeze(0).to(device=self.device, dtype=self.dtype)
        if gyro.ndim == 0 or acc.ndim == 0:
            Logger.write("warn", "IMU measurements malformed, using previous pose")
            return self.prev_pose
        
        time_delta = imu_data.time_delta.squeeze(0).float()
        if time_delta.ndim == 0:
            time_delta = time_delta.unsqueeze(0)
        if time_delta.ndim == 2:
            time_delta = time_delta.squeeze(-1)
        time_delta = (time_delta / 1e9).to(device=self.device, dtype=self.dtype)
        if time_delta.numel() == 0:
            Logger.write("warn", "IMU time deltas unavailable, using previous pose")
            return self.prev_pose
        
        steps = min(time_delta.numel(), gyro.shape[0])
        if steps == 0:
            Logger.write("warn", "Insufficient IMU samples, using previous pose")
            return self.prev_pose
        gyro = gyro[:steps]
        acc = acc[:steps]
        time_delta = time_delta[:steps]
        
        prev_rot = self.prev_pose.to(device=self.device, dtype=self.dtype).rotation()
        prev_pos = self.prev_pose.translation().to(device=self.device, dtype=self.dtype).reshape(-1)
        init_vel = self.prev_velocity if self.prev_velocity is not None else torch.zeros(3, device=self.device, dtype=self.dtype)
        
        final_rot, final_vel, final_pos = self.integrator.integrate(
            prev_rot, init_vel, prev_pos, gyro, acc, time_delta
        )
        
        self.prev_velocity = final_vel.clone()
        pose_tensor = torch.cat([final_pos, final_rot.tensor()], dim=0)
        new_pose = pp.SE3(pose_tensor)
        self.prev_pose = new_pose
        
        return new_pose
    
    def update(self, pose: pp.LieTensor) -> None:
        self.prev_pose = pose.to(self.device, dtype=self.dtype)
        if self.prev_velocity is not None:
            self.prev_velocity = self.prev_velocity.to(device=self.device, dtype=self.dtype)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        if config is None:
            return
        gravity = getattr(config, "gravity", 9.81)
        device = getattr(config, "device", "cpu")
        
        if not isinstance(gravity, (int, float)) or gravity <= 0:
            raise ValueError("gravity must be a positive number")
        if not isinstance(device, str):
            raise ValueError("device must be a string")


class IMUVisualFusion(IMotionModel[StereoInertialFrame]):
    """
    IMU-Visual fusion motion model that combines IMU integration with visual odometry.
    
    This model uses IMU for high-rate prediction and visual information for drift correction.
    The fusion strategy:
    1. IMU provides initial pose prediction
    2. Visual flow/depth provides motion refinement
    3. Weighted fusion based on confidence/covariance
    """
    
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.prev_pose: pp.LieTensor | None = None
        self.prev_velocity: torch.Tensor | None = None
        
        gravity = getattr(self.config, "gravity", 9.81)
        device_str = getattr(self.config, "device", "cpu")
        imu_weight = getattr(self.config, "imu_weight", 0.3)
        visual_weight = getattr(self.config, "visual_weight", 0.7)
        use_visual = getattr(self.config, "use_visual", True)
        forward_scale = getattr(self.config, "forward_scale", 0.5)
        
        total_weight = float(imu_weight + visual_weight)
        if total_weight <= 1e-6:
            Logger.write("warn", "IMUVisualFusion received near-zero weights; falling back to equal weighting.")
            imu_weight = visual_weight = 0.5
            total_weight = 1.0
        
        self.imu_weight = float(imu_weight) / total_weight
        self.visual_weight = float(visual_weight) / total_weight
        self.use_visual = bool(use_visual)
        self.forward_scale = float(forward_scale)
        
        self.integrator = SimpleIMUIntegrator(gravity=gravity)
        self.device = torch.device(device_str)
        self.device_str = device_str
        self.dtype = torch.float32
        
        self.prev_frame_time: int | None = None
        
        Logger.write("info", f"IMU-Visual Fusion initialized: IMU weight={self.imu_weight:.2f}, Visual weight={self.visual_weight:.2f}, forward scale={self.forward_scale:.2f}")
    
    def predict(self, frame: StereoInertialFrame, flow: torch.Tensor | None, depth: torch.Tensor | None) -> pp.LieTensor:
        if self.prev_pose is None:
            self.prev_pose = pp.identity_SE3(device=self.device_str, dtype=self.dtype)
            self.prev_velocity = torch.zeros(3, device=self.device, dtype=self.dtype)
            self.prev_frame_time = frame.stereo.frame_ns
            return pp.identity_SE3(device=self.device_str, dtype=self.dtype)
        
        imu_pose = self._predict_from_imu(frame)
        
        if self.use_visual and flow is not None and depth is not None:
            visual_pose = self._predict_from_visual(frame, flow, depth)
            
            if visual_pose is not None:
                fused_pose = self._fuse_poses(imu_pose, visual_pose)
            else:
                fused_pose = imu_pose
        else:
            fused_pose = imu_pose
        
        self.prev_pose = fused_pose
        self.prev_frame_time = frame.stereo.frame_ns
        
        return fused_pose
    
    def _predict_from_imu(self, frame: StereoInertialFrame) -> pp.LieTensor:
        """Use IMU measurements to predict pose."""
        imu_data = frame.imu
        gyro = imu_data.gyro
        acc = imu_data.acc
        
        if gyro.numel() == 0 or acc.numel() == 0:
            return self.prev_pose
        
        gyro = gyro.squeeze(0).to(device=self.device, dtype=self.dtype)
        acc = acc.squeeze(0).to(device=self.device, dtype=self.dtype)
        
        if gyro.ndim == 0 or acc.ndim == 0:
            return self.prev_pose
        
        time_delta = imu_data.time_delta.squeeze(0).float()
        if time_delta.ndim == 0:
            time_delta = time_delta.unsqueeze(0)
        if time_delta.ndim == 2:
            time_delta = time_delta.squeeze(-1)
        time_delta = (time_delta / 1e9).to(device=self.device, dtype=self.dtype)
        
        if time_delta.numel() == 0:
            return self.prev_pose
        
        steps = min(time_delta.numel(), gyro.shape[0])
        if steps == 0:
            return self.prev_pose
        
        gyro = gyro[:steps]
        acc = acc[:steps]
        time_delta = time_delta[:steps]
        
        prev_rot = self.prev_pose.to(device=self.device, dtype=self.dtype).rotation()
        prev_pos = self.prev_pose.translation().to(device=self.device, dtype=self.dtype).reshape(-1)
        init_vel = self.prev_velocity if self.prev_velocity is not None else torch.zeros(3, device=self.device, dtype=self.dtype)
        
        final_rot, final_vel, final_pos = self.integrator.integrate(
            prev_rot, init_vel, prev_pos, gyro, acc, time_delta
        )
        
        self.prev_velocity = final_vel.clone()
        pose_tensor = torch.cat([final_pos, final_rot.tensor()], dim=0)
        
        return pp.SE3(pose_tensor)
    
    def _predict_from_visual(self, frame: StereoInertialFrame, flow: torch.Tensor, depth: torch.Tensor) -> pp.LieTensor | None:
        """
        Estimate motion from visual flow and depth.
        This is a simplified visual odometry estimation.
        """
        try:
            if flow.numel() < 100 or depth.numel() < 100:
                return None
            
            flow = flow.to(device=self.device, dtype=self.dtype)
            depth = depth.to(device=self.device, dtype=self.dtype)
            K = frame.stereo.frame_K.to(device=self.device, dtype=self.dtype)
            fx = torch.clamp(K[0, 0], min=1e-6)
            fy = torch.clamp(K[1, 1], min=1e-6)
            
            flow_b = flow[0]
            depth_b = depth[0, 0]
            flow_flat = flow_b.view(2, -1)
            depth_flat = depth_b.view(-1)
            
            flow_magnitude = torch.norm(flow_flat, dim=0)
            valid_mask = (flow_magnitude > 0.1) & (flow_magnitude < 50.0)
            
            if valid_mask.sum() < 10:
                return None
            
            valid_flow = flow_flat[:, valid_mask]
            valid_depth = depth_flat[valid_mask]
            valid_depth = valid_depth[valid_depth > 0]
            
            if valid_depth.numel() == 0:
                depth_positive = depth_flat[depth_flat > 0]
                mean_depth = depth_positive.mean() if depth_positive.numel() > 0 else torch.tensor(1.0, device=self.device, dtype=self.dtype)
            else:
                mean_depth = valid_depth.mean()
            mean_depth = torch.clamp(mean_depth, min=0.1, max=200.0)
            
            avg_flow_vec = valid_flow.mean(dim=1)
            flow_norm = torch.norm(avg_flow_vec)
            if flow_norm < 1e-6:
                return self.prev_pose
            
            time_diff = (frame.stereo.frame_ns - self.prev_frame_time) / 1e9 if self.prev_frame_time else 0.01
            time_diff = float(torch.clamp(torch.tensor(time_diff, device=self.device, dtype=self.dtype), min=0.001, max=1.0).item())
            
            translation = torch.zeros(3, device=self.device, dtype=self.dtype)
            translation[0] = avg_flow_vec[0] * mean_depth / fx
            translation[1] = avg_flow_vec[1] * mean_depth / fy
            translation_xy_norm = torch.norm(translation[:2])
            translation[2] = -self.forward_scale * translation_xy_norm
            translation *= time_diff
            
            yaw_rate = (avg_flow_vec[0] / fx - avg_flow_vec[1] / fy) * 0.5
            rotation_angle = torch.clamp(yaw_rate * time_diff, min=-0.5, max=0.5)
            
            if torch.abs(rotation_angle) > 1e-8:
                axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=self.dtype)
                half_angle = rotation_angle * 0.5
                cos_half = torch.cos(half_angle)
                sin_half = torch.sin(half_angle)
                quat = torch.empty(4, device=self.device, dtype=self.dtype)
                quat[0] = cos_half
                quat[1:] = sin_half * axis
                delta_rot = pp.SO3(quat)
            else:
                delta_rot = pp.SO3(torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype))
            
            prev_rot = self.prev_pose.rotation().to(device=self.device, dtype=self.dtype)
            prev_pos = self.prev_pose.translation().to(device=self.device, dtype=self.dtype).reshape(-1)
            
            new_rot = prev_rot @ delta_rot
            rotation_mat = prev_rot.matrix().squeeze()
            new_pos = prev_pos + rotation_mat @ translation
            
            pose_tensor = torch.cat([new_pos, new_rot.tensor()], dim=0)
            return pp.SE3(pose_tensor)
            
        except Exception as e:
            Logger.write("warn", f"Visual prediction failed: {e}")
            return None
    
    def _fuse_poses(self, imu_pose: pp.LieTensor, visual_pose: pp.LieTensor) -> pp.LieTensor:
        """
        Fuse IMU and visual pose predictions using weighted average.
        """
        imu_pos = imu_pose.translation().reshape(-1)
        imu_rot = imu_pose.rotation()
        
        visual_pos = visual_pose.translation().reshape(-1)
        visual_rot = visual_pose.rotation()
        
        fused_pos = self.imu_weight * imu_pos + self.visual_weight * visual_pos
        
        imu_quat = imu_rot.tensor()
        visual_quat = visual_rot.tensor()
        
        if torch.dot(imu_quat, visual_quat) < 0:
            visual_quat = -visual_quat
        
        fused_quat = self.imu_weight * imu_quat + self.visual_weight * visual_quat
        quat_norm = torch.norm(fused_quat)
        if quat_norm < 1e-8:
            fused_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        else:
            fused_quat = fused_quat / quat_norm
        fused_rot = pp.SO3(fused_quat)
        
        pose_tensor = torch.cat([fused_pos, fused_rot.tensor()], dim=0)
        return pp.SE3(pose_tensor)
    
    def update(self, pose: pp.LieTensor) -> None:
        """Update with optimized pose."""
        self.prev_pose = pose.to(self.device, dtype=self.dtype)
        if self.prev_velocity is not None:
            self.prev_velocity = self.prev_velocity.to(device=self.device, dtype=self.dtype)
    
    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        if config is None:
            return
        
        gravity = getattr(config, "gravity", 9.81)
        device = getattr(config, "device", "cpu")
        imu_weight = getattr(config, "imu_weight", 0.3)
        visual_weight = getattr(config, "visual_weight", 0.7)
        forward_scale = getattr(config, "forward_scale", 0.5)
        use_visual = getattr(config, "use_visual", True)
        
        if not isinstance(gravity, (int, float)) or gravity <= 0:
            raise ValueError("gravity must be a positive number")
        if not isinstance(device, str):
            raise ValueError("device must be a string")
        if not isinstance(imu_weight, (int, float)) or imu_weight < 0 or imu_weight > 1:
            raise ValueError("imu_weight must be between 0 and 1")
        if not isinstance(visual_weight, (int, float)) or visual_weight < 0 or visual_weight > 1:
            raise ValueError("visual_weight must be between 0 and 1")
        if not isinstance(forward_scale, (int, float)) or forward_scale < 0:
            raise ValueError("forward_scale must be non-negative")
        if not isinstance(use_visual, bool):
            raise ValueError("use_visual must be a boolean")
        
        total_weight = imu_weight + visual_weight
        if abs(total_weight - 1.0) > 0.01:
            Logger.write("warn", f"IMU and visual weights sum to {total_weight:.3f}, not 1.0. Weights will be normalized.")
