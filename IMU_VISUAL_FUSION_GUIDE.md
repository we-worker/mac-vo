# IMU-Visual Fusion Implementation Guide

## 概述 (Overview)

本实现为MAC-VO视觉里程计系统添加了IMU数据融合功能，实现了IMU与双目视觉的协同定位。

This implementation adds IMU data fusion to the MAC-VO visual odometry system, enabling collaborative localization between IMU and stereo vision.

## 核心特性 (Key Features)

### 1. 简洁的代码架构 (Clean Code Architecture)
- **模块化设计**: IMU集成、运动模型、融合策略完全分离
- **类型安全**: 完整的类型注解和PyPose Lie群操作
- **易于扩展**: 可轻松添加偏差估计、协方差传播等高级特性

### 2. 三种运动模型 (Three Motion Models)

#### SimpleIMUMotion - 纯IMU模型
- 仅使用IMU数据进行位姿预测
- 适用场景: 短时序列、高质量IMU、无视觉数据
- 配置文件: `Config/Module/MotionModel/SimpleIMUMotion.yaml`

#### IMUVisualFusion - IMU-视觉融合模型 (推荐)
- **融合策略**: 加权平均IMU预测和视觉预测
- **自适应降级**: 视觉失效时自动切换到纯IMU模式
- **可配置权重**: 根据传感器质量调整融合比例
- 配置文件: `Config/Module/MotionModel/IMUVisualFusion.yaml`

#### 其他视觉模型
- GTMotionwithNoise、TartanMotionNet、StaticMotionModel等
- 可与IMU融合模型组合使用

## 快速开始 (Quick Start)

### 步骤1: 使用IMU数据集

```bash
# 使用EuRoC数据集 (包含IMU数据)
python MACVO.py \
  --odom Config/Experiment/MACVO/MACVO_IMUVisualFusion_Example.yaml \
  --data Config/Sequence/EuRoC_MH01.yaml \
  --useRR  # 可选: 启用Rerun可视化
```

### 步骤2: 配置融合参数

编辑配置文件或在YAML中指定:

```yaml
Odometry:
  motion:
    type: IMUVisualFusion
    args:
      gravity: 9.81         # 重力常数 (m/s²)
      device: cpu           # 计算设备
      imu_weight: 0.3       # IMU权重 (0-1)
      visual_weight: 0.7    # 视觉权重 (0-1)
      forward_scale: 0.5    # 前向运动缩放
      use_visual: true      # 启用视觉分量
```

### 步骤3: 根据场景调整权重

| 场景 | IMU权重 | 视觉权重 | 原因 |
|------|---------|----------|------|
| 高质量IMU + 弱光 | 0.6 | 0.4 | IMU更可靠 |
| 消费级IMU + 丰富纹理 | 0.2 | 0.8 | 视觉更准确 |
| 平衡场景 (默认) | 0.3 | 0.7 | 综合性能最优 |
| 高速运动 | 0.5 | 0.5 | 利用IMU高频率 |

## 技术细节 (Technical Details)

### IMU集成算法

**SimpleIMUIntegrator** 实现了前向积分:

1. **旋转更新**: 使用轴角表示和SO(3)群操作
2. **速度更新**: `v = v + (R * a - g) * dt`
3. **位置更新**: `p = p + v * dt + 0.5 * (R * a - g) * dt²`
4. **重力补偿**: 在世界坐标系中移除重力分量

### 融合策略

**IMUVisualFusion** 实现互补融合:

#### IMU分支
- 输入: 陀螺仪 (角速度) + 加速度计
- 频率: 100-200 Hz
- 优点: 高频、短期精确
- 缺点: 长期漂移

#### 视觉分支
- 输入: 光流 + 深度图
- 频率: ~30 Hz (相机帧率)
- 优点: 无漂移
- 缺点: 计算量大、对纹理敏感

#### 加权融合
```python
# 位置融合
fused_pos = imu_weight * imu_pos + visual_weight * visual_pos

# 旋转融合 (四元数插值)
fused_quat = normalized(imu_weight * imu_quat + visual_weight * visual_quat)
```

## 代码结构 (Code Structure)

```
Module/
├── IMUIntegration.py              # IMU积分核心
│   ├── SimpleIMUIntegrator        # 前向积分器
│   └── IMUPreintegrator           # 预积分器 (用于优化)
│
├── MotionModel.py                 # 运动模型
│   ├── SimpleIMUMotion            # 纯IMU运动模型
│   └── IMUVisualFusion            # IMU-视觉融合模型
│
└── IMU_FUSION_README.md           # 详细技术文档

Config/Module/MotionModel/
├── SimpleIMUMotion.yaml           # 纯IMU配置
└── IMUVisualFusion.yaml           # 融合配置

Scripts/Test/
├── test_imu_integration.py        # 单元测试
└── demo_imu_visual_fusion.py      # 融合演示
```

## 演示和测试 (Demos & Tests)

### 运行单元测试

```bash
python Scripts/Test/test_imu_integration.py
```

测试内容:
- ✓ 静态IMU测试 (无运动时姿态保持)
- ✓ 纯旋转测试 (仅陀螺仪数据)
- ✓ 恒定加速度测试 (加速度积分)
- ✓ 预积分测试
- ✓ 设备兼容性测试 (CPU/GPU)

### 运行融合演示

```bash
python Scripts/Test/demo_imu_visual_fusion.py
```

演示内容:
1. 纯IMU运动预测
2. IMU-视觉融合预测
3. 不同权重配置比较
4. 视觉失效时的降级行为

## 性能优化建议 (Performance Tips)

### 1. 硬件加速
```yaml
args:
  device: cuda  # 使用GPU加速
```

### 2. 调整IMU采样率
- 100-200 Hz: 标准配置，平衡精度和性能
- 50-100 Hz: 降低计算量，略微降低精度
- 200+ Hz: 提高精度，增加计算量

### 3. 动态权重调整
根据场景动态调整融合权重:
```python
# 伪代码示例
if low_texture_detected:
    imu_weight = 0.6  # 增加IMU权重
else:
    imu_weight = 0.3  # 标准配置
```

## 未来改进方向 (Future Improvements)

### 短期改进
1. **在线偏差估计**: 估计陀螺仪和加速度计偏差
2. **协方差传播**: 跟踪不确定性并用于优化
3. **高阶积分**: 使用Runge-Kutta 4阶积分提高精度

### 中期改进
1. **紧耦合融合**: 在因子图中联合优化IMU和视觉因子
2. **IMU预积分因子**: 在后端优化中使用预积分约束
3. **自适应权重**: 基于观测质量动态调整融合权重

### 长期改进
1. **学习式融合**: 使用神经网络学习最优融合策略
2. **多传感器融合**: 添加GPS、磁力计等其他传感器
3. **实时校准**: 在线估计IMU-相机外参

## 数据集支持 (Dataset Support)

已测试的数据集:
- ✓ **EuRoC MAV Dataset**: 微型飞行器数据，室内环境
- ✓ **TartanAir**: 仿真数据集，包含IMU
- ✓ **VBR Dataset**: 车载数据集

配置示例:
```bash
# EuRoC
--data Config/Sequence/EuRoC_MH01.yaml

# TartanAir2
--data Config/Sequence/TartanAir2_abandonfac_001.yaml
```

## 常见问题 (FAQ)

### Q1: IMU数据格式要求?
**A**: 需要StereoInertialFrame格式，包含:
- `gyro`: 角速度 (rad/s), shape (B, N, 3)
- `acc`: 加速度 (m/s²), shape (B, N, 3)
- `time_ns`: 时间戳 (纳秒)

### Q2: 融合权重如何选择?
**A**: 
- 从默认值开始 (0.3/0.7)
- 观察轨迹漂移情况
- IMU漂移严重 → 降低imu_weight
- 视觉不稳定 → 增加imu_weight

### Q3: 可以只用IMU吗?
**A**: 可以，使用SimpleIMUMotion模型，但会有累积漂移。

### Q4: 如何处理不同的重力值?
**A**: 根据地理位置调整`gravity`参数:
- 赤道: 9.78 m/s²
- 标准: 9.81 m/s²
- 极地: 9.83 m/s²

### Q5: 支持哪些IMU频率?
**A**: 理论支持任意频率，推荐100-200 Hz。频率过低会降低精度。

## 引用 (Citation)

如果使用本IMU融合实现，请引用:

```bibtex
@software{macvo_imu_fusion,
  title={IMU-Visual Fusion for MAC-VO},
  author={MAC-VO Team},
  year={2024},
  note={Clean and simple IMU integration for visual odometry}
}
```

## 参考资料 (References)

1. **PyPose**: Lie group operations for robotics
   - https://pypose.org/

2. **IMU Preintegration**: Forster et al., RSS 2015
   - "IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation"

3. **Lie Groups for Robotics**: Sola et al., 2018
   - "A micro Lie theory for state estimation in robotics"

4. **EuRoC Dataset**: Burri et al., IJRR 2016
   - Stereo-inertial benchmark dataset

## 联系方式 (Contact)

如有问题或建议，请:
1. 查看详细文档: `Module/IMU_FUSION_README.md`
2. 运行测试脚本验证安装
3. 在项目仓库提交Issue

---

**简洁、清晰、可扩展** - Clean, Clear, Extensible
