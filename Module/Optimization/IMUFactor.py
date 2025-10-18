"""
IMU Preintegration Factor for Graph Optimization.

This module implements IMU preintegration as a factor in the pose graph optimization,
following the approach from Forster et al. "IMU Preintegration on Manifold for 
Efficient Visual-Inertial Maximum-a-Posteriori Estimation" (RSS 2015).
"""

import torch
import pypose as pp
import typing as T
from dataclasses import dataclass

from ..IMUIntegration import SimpleIMUIntegrator
from .PyposeOptimizers import AnalyticModule, FactorGraph


@dataclass
class IMUMeasurement:
    """IMU measurements between two keyframes."""
    gyro: torch.Tensor
    acc: torch.Tensor
    dt: torch.Tensor
    gravity: float = 9.81


@dataclass
class IMUPreintegrated:
    """Preintegrated IMU measurements."""
    delta_R: pp.LieTensor
    delta_v: torch.Tensor
    delta_p: torch.Tensor
    sum_dt: float
    gravity: float
    
    def to(self, device: torch.device, dtype: torch.dtype) -> 'IMUPreintegrated':
        """Move preintegrated data to device and dtype."""
        return IMUPreintegrated(
            delta_R=self.delta_R.to(device=device, dtype=dtype),
            delta_v=self.delta_v.to(device=device, dtype=dtype),
            delta_p=self.delta_p.to(device=device, dtype=dtype),
            sum_dt=self.sum_dt,
            gravity=self.gravity
        )


class IMUPreintegratorFactor:
    """
    Preintegrates IMU measurements for use in factor graph optimization.
    
    This class computes relative pose changes between keyframes using IMU data,
    which can then be used as a constraint in pose graph optimization.
    """
    
    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity
        self.integrator = SimpleIMUIntegrator(gravity=gravity)
    
    def preintegrate(self, imu_meas: IMUMeasurement) -> IMUPreintegrated:
        """
        Preintegrate IMU measurements relative to initial state.
        
        Args:
            imu_meas: IMU measurements (gyro, acc, dt)
            
        Returns:
            Preintegrated measurements (delta_R, delta_v, delta_p)
        """
        device = imu_meas.gyro.device
        dtype = imu_meas.gyro.dtype
        
        init_rot = pp.SO3(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype))
        init_vel = torch.zeros(3, device=device, dtype=dtype)
        init_pos = torch.zeros(3, device=device, dtype=dtype)
        
        delta_rot, delta_vel, delta_pos = self.integrator.integrate(
            init_rot, init_vel, init_pos,
            imu_meas.gyro, imu_meas.acc, imu_meas.dt
        )
        
        sum_dt = imu_meas.dt.sum().item()
        
        return IMUPreintegrated(
            delta_R=delta_rot,
            delta_v=delta_vel,
            delta_p=delta_pos,
            sum_dt=sum_dt,
            gravity=self.gravity
        )


@dataclass
class IMUGraphInput:
    """Input data for IMU factor graph optimization."""
    pose_i: pp.LieTensor
    pose_j: pp.LieTensor
    vel_i: torch.Tensor
    vel_j: torch.Tensor
    preint: IMUPreintegrated
    device: str = "cpu"


class IMUFactor_Graph(FactorGraph):
    """
    IMU preintegration factor for pose graph optimization.
    
    This factor enforces IMU constraints between two poses (i and j) based on
    preintegrated IMU measurements.
    
    Residual: r = [r_rotation, r_velocity, r_position]
    where:
        r_rotation = Log(delta_R^T * R_i^T * R_j)
        r_velocity = R_i^T * (v_j - v_i - g * dt) - delta_v
        r_position = R_i^T * (p_j - p_i - v_i * dt - 0.5 * g * dt^2) - delta_p
    """
    
    def __init__(self, graph_data: IMUGraphInput):
        super().__init__()
        self.device = graph_data.device
        
        self.pose_i = pp.Parameter(pp.SE3(graph_data.pose_i))
        self.pose_j = pp.Parameter(pp.SE3(graph_data.pose_j))
        
        self.register_buffer("vel_i", graph_data.vel_i)
        self.register_buffer("vel_j", graph_data.vel_j)
        
        preint = graph_data.preint
        self.register_buffer("delta_R", preint.delta_R.tensor())
        self.register_buffer("delta_v", preint.delta_v)
        self.register_buffer("delta_p", preint.delta_p)
        self.register_buffer("dt", torch.tensor([preint.sum_dt], dtype=torch.float32))
        self.register_buffer("gravity_vec", torch.tensor([0.0, 0.0, preint.gravity], dtype=torch.float32))
    
    def forward(self) -> torch.Tensor:
        """
        Compute IMU preintegration residuals.
        
        Returns:
            Residual tensor of shape (9,) = [3 rotation + 3 velocity + 3 position]
        """
        R_i = self.pose_i.rotation()
        R_j = self.pose_j.rotation()
        p_i = self.pose_i.translation()
        p_j = self.pose_j.translation()
        
        delta_R_preint = pp.SO3(self.delta_R)
        
        r_R = (delta_R_preint.Inv() @ (R_i.Inv() @ R_j)).Log()
        
        dt = self.dt.item()
        R_i_T = R_i.matrix().transpose(-2, -1).squeeze()
        r_v = R_i_T @ (self.vel_j - self.vel_i - self.gravity_vec * dt) - self.delta_v
        
        r_p = R_i_T @ (p_j - p_i - self.vel_i * dt - 0.5 * self.gravity_vec * dt * dt) - self.delta_p
        
        residual = torch.cat([r_R, r_v, r_p], dim=0)
        return residual
    
    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        """
        Return covariance matrix for the IMU factor.
        
        For simplicity, we use a diagonal covariance matrix.
        In practice, this should be computed from IMU noise characteristics.
        """
        cov = torch.eye(9, device=self.device, dtype=torch.float32)
        
        cov[:3, :3] *= 0.01
        cov[3:6, 3:6] *= 0.1
        cov[6:9, 6:9] *= 0.1
        
        return cov
    
    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self) -> tuple[pp.LieTensor, pp.LieTensor, torch.Tensor, torch.Tensor]:
        """Return optimized poses and velocities."""
        return (
            self.pose_i.clone(),
            self.pose_j.clone(),
            self.vel_i.clone(),
            self.vel_j.clone()
        )


class Analytic_IMUFactor_Graph(IMUFactor_Graph, AnalyticModule):
    """
    Analytic Jacobian version of IMU factor for faster optimization.
    
    Computes analytical Jacobians with respect to pose_i, pose_j, vel_i, vel_j.
    """
    
    def __init__(self, graph_data: IMUGraphInput):
        super().__init__(graph_data)
    
    @torch.no_grad()
    def build_jacobian(self) -> torch.Tensor:
        """
        Build analytical Jacobian of IMU residual.
        
        Returns:
            Jacobian matrix of shape (9, 28) for [pose_i(7), pose_j(7), vel_i(3), vel_j(3)]
            
        Note: For simplicity, we compute partial Jacobians. 
        Full implementation should include all cross-terms.
        """
        R_i = self.pose_i.rotation()
        R_j = self.pose_j.rotation()
        p_i = self.pose_i.translation()
        p_j = self.pose_j.translation()
        
        dt = self.dt.item()
        R_i_mat = R_i.matrix().squeeze()
        R_i_T = R_i_mat.transpose(-2, -1)
        
        J = torch.zeros(9, 28, device=self.device, dtype=torch.float32)
        
        delta_R_preint = pp.SO3(self.delta_R)
        dR = delta_R_preint.Inv() @ (R_i.Inv() @ R_j)
        
        I3 = torch.eye(3, device=self.device, dtype=torch.float32)
        
        J[0:3, 3:6] = -dR.Adj().squeeze()
        J[0:3, 10:13] = dR.Inv().matrix().squeeze()
        
        dp_ij = p_j - p_i - self.vel_i * dt - 0.5 * self.gravity_vec * dt * dt
        J[6:9, 3:6] = R_i_T @ pp.vec2skew(dp_ij).squeeze()
        J[6:9, 0:3] = -R_i_T
        J[6:9, 7:10] = R_i_T
        J[6:9, 21:24] = -R_i_T * dt
        
        dv_ij = self.vel_j - self.vel_i - self.gravity_vec * dt
        J[3:6, 3:6] = R_i_T @ pp.vec2skew(dv_ij).squeeze()
        J[3:6, 21:24] = -R_i_T
        J[3:6, 24:27] = R_i_T
        
        return J


class IMU_VisualFactor_Graph(FactorGraph):
    """
    Combined IMU and visual factor graph for joint optimization.
    
    This graph combines:
    1. Visual reprojection factors (from MAC-VO)
    2. IMU preintegration factors
    
    Optimizes: [pose_0, ..., pose_N, vel_0, ..., vel_N]
    """
    
    def __init__(self, visual_residual_fn, imu_factors: list[IMUFactor_Graph], 
                 visual_weight: float = 1.0, imu_weight: float = 1.0):
        super().__init__()
        self.visual_residual_fn = visual_residual_fn
        self.imu_factors = torch.nn.ModuleList(imu_factors)
        self.visual_weight = visual_weight
        self.imu_weight = imu_weight
    
    def forward(self) -> torch.Tensor:
        """
        Compute combined residuals from visual and IMU factors.
        
        Returns:
            Combined residual vector
        """
        visual_res = self.visual_residual_fn() * self.visual_weight
        
        imu_residuals = []
        for imu_factor in self.imu_factors:
            imu_residuals.append(imu_factor() * self.imu_weight)
        
        if len(imu_residuals) > 0:
            imu_res = torch.cat(imu_residuals, dim=0)
            return torch.cat([visual_res, imu_res], dim=0)
        else:
            return visual_res
    
    @torch.no_grad()
    @torch.inference_mode()
    def covariance_array(self) -> torch.Tensor:
        """Return block-diagonal covariance for visual and IMU factors."""
        visual_cov = self.visual_residual_fn.covariance_array()
        
        imu_covs = [factor.covariance_array() for factor in self.imu_factors]
        
        if len(imu_covs) > 0:
            imu_cov = torch.block_diag(*imu_covs)
            return torch.block_diag(visual_cov, imu_cov)
        else:
            return visual_cov
    
    @torch.no_grad()
    @torch.inference_mode()
    def write_back(self):
        """Write back optimized states."""
        visual_result = self.visual_residual_fn.write_back()
        
        imu_results = [factor.write_back() for factor in self.imu_factors]
        
        return {
            "visual": visual_result,
            "imu": imu_results
        }
