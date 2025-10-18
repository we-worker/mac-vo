import torch
import pypose as pp
from typing import Optional


class SimpleIMUIntegrator:
    """
    A simple IMU integration scheme for fusing IMU measurements.
    
    This class performs forward integration of IMU measurements to estimate
    pose, velocity, and position changes between frames.
    """
    
    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity
        self.gravity_vec = torch.tensor([0.0, 0.0, gravity], dtype=torch.float32)
        
    def integrate(
        self,
        init_rot: pp.LieTensor,
        init_vel: torch.Tensor,
        init_pos: torch.Tensor,
        gyro: torch.Tensor,
        acc: torch.Tensor,
        dt: torch.Tensor
    ) -> tuple[pp.LieTensor, torch.Tensor, torch.Tensor]:
        """
        Integrate IMU measurements to estimate pose, velocity, and position.
        
        Args:
            init_rot: Initial rotation as SO3 LieTensor, shape (4,)
            init_vel: Initial velocity, shape (3,)
            init_pos: Initial position, shape (3,)
            gyro: Angular velocity measurements, shape (N, 3)
            acc: Acceleration measurements with gravity, shape (N, 3)
            dt: Time deltas between measurements, shape (N,) or (N, 1)
            
        Returns:
            final_rot: Final rotation as SO3 LieTensor
            final_vel: Final velocity tensor
            final_pos: Final position tensor
        """
        device = gyro.device
        dtype = gyro.dtype
        self.gravity_vec = self.gravity_vec.to(device=device, dtype=dtype)
        
        dt = dt.to(device=device, dtype=dtype).reshape(-1)
        if dt.numel() == 0:
            return init_rot.to(device=device, dtype=dtype), init_vel.to(device=device, dtype=dtype).reshape(-1), init_pos.to(device=device, dtype=dtype).reshape(-1)
        
        rot = init_rot.to(device=device, dtype=dtype)
        vel = init_vel.to(device=device, dtype=dtype).reshape(-1)
        pos = init_pos.to(device=device, dtype=dtype).reshape(-1)
        
        gyro = gyro.to(device=device, dtype=dtype)
        acc = acc.to(device=device, dtype=dtype)
        if gyro.dim() == 1:
            gyro = gyro.unsqueeze(0)
            acc = acc.unsqueeze(0)
        steps = min(dt.numel(), gyro.shape[0])
        gyro = gyro[:steps]
        acc = acc[:steps]
        
        for i in range(steps):
            delta_t = dt[i]
            rotation_mat = rot.matrix().squeeze()
            acc_world = rotation_mat @ acc[i] - self.gravity_vec
            
            pos = pos + vel * delta_t + 0.5 * acc_world * delta_t * delta_t
            vel = vel + acc_world * delta_t
            
            omega = gyro[i] * delta_t
            omega_norm = torch.norm(omega)
            if omega_norm > 1e-8:
                axis = omega / omega_norm
                half_angle = omega_norm * 0.5
                cos_half = torch.cos(half_angle)
                sin_half = torch.sin(half_angle)
                quat = torch.empty(4, device=device, dtype=dtype)
                quat[0] = cos_half
                quat[1:] = sin_half * axis
            else:
                quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
            delta_rot = pp.SO3(quat)
            rot = rot @ delta_rot
        
        return rot, vel, pos


class IMUPreintegrator:
    """
    A simple IMU preintegration module for efficient multi-frame optimization.
    
    Preintegrates IMU measurements between keyframes for use in factor graph optimization.
    """
    
    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity
        self.integrator = SimpleIMUIntegrator(gravity)
        self.reset()
    
    def reset(self):
        """Reset preintegration state."""
        self.delta_rot: Optional[pp.LieTensor] = None
        self.delta_vel: Optional[torch.Tensor] = None
        self.delta_pos: Optional[torch.Tensor] = None
        self.sum_dt: float = 0.0
    
    def preintegrate(
        self,
        gyro: torch.Tensor,
        acc: torch.Tensor,
        dt: torch.Tensor
    ) -> tuple[pp.LieTensor, torch.Tensor, torch.Tensor]:
        """
        Preintegrate IMU measurements relative to initial state.
        
        Args:
            gyro: Angular velocity measurements, shape (N, 3)
            acc: Acceleration measurements, shape (N, 3)
            dt: Time deltas, shape (N,) or (N, 1)
            
        Returns:
            delta_rot: Preintegrated rotation change
            delta_vel: Preintegrated velocity change
            delta_pos: Preintegrated position change
        """
        device = gyro.device
        
        init_rot = pp.SO3(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device))
        init_vel = torch.zeros(3, device=device)
        init_pos = torch.zeros(3, device=device)
        
        self.delta_rot, self.delta_vel, self.delta_pos = self.integrator.integrate(
            init_rot, init_vel, init_pos, gyro, acc, dt
        )
        
        self.sum_dt = dt.sum().item()
        
        return self.delta_rot, self.delta_vel, self.delta_pos
    
    def get_preintegrated(self) -> tuple[Optional[pp.LieTensor], Optional[torch.Tensor], Optional[torch.Tensor], float]:
        """
        Get the current preintegrated measurements.
        
        Returns:
            delta_rot: Preintegrated rotation
            delta_vel: Preintegrated velocity
            delta_pos: Preintegrated position
            sum_dt: Total integration time
        """
        return self.delta_rot, self.delta_vel, self.delta_pos, self.sum_dt
