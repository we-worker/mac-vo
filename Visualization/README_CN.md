# 相机轨迹3D可视化

一个基于Web的交互式应用程序，用于在3D空间中可视化和分析相机位姿轨迹。

## 功能特性

### 📁 数据加载
- **文件上传**：支持 `.txt` 和 `.npy` 文件格式
- **TXT格式**：类CSV格式，列为：`FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ`
- **NPY格式**：NumPy二进制格式，形状为 `(N, 7)` 或 `(N, 8)`，包含SE3位姿

### 🎨 3D可视化
- **交互场景**：使用Three.js构建，实现流畅的3D渲染
- **相机路径**：轨迹显示为连接线
- **相机标记**：关键帧处的简化相机几何体
- **当前相机**：高亮显示的3D相机模型，显示当前帧的位置和方向
- **网格与坐标轴**：用于空间定位的参考网格和坐标轴

### 🎮 控制方式
- **鼠标控制**：
  - 左键拖拽：旋转视图
  - 右键拖拽：平移视图
  - 滚轮：缩放
- **相机可见性**：切换相机标记的显示/隐藏
- **坐标轴可见性**：切换坐标轴的显示/隐藏
- **重置视图**：自动将轨迹适配到视口中

### 📊 距离计算
- **总距离**：计算整个轨迹的累计距离
- **范围距离**：计算选定帧范围的距离
- **实时更新**：帧范围改变时距离实时更新

### ▶️ 播放功能
- **播放/暂停**：动画播放轨迹帧
- **速度控制**：可调节播放速度（1-100 FPS）
- **帧滑块**：手动浏览帧
- **进度条**：播放进度的可视化指示
- **重置**：跳回起始帧

### 📍 位姿信息
实时显示当前帧的位姿数据：
- 帧ID
- 位置（X, Y, Z）
- 四元数（W, X, Y, Z）

### 🎯 帧范围选择
- 设置自定义起始和结束帧
- 计算特定轨迹段的距离
- 在选定范围内播放

## 使用方法

### 打开应用程序

1. 在现代Web浏览器中打开 `camera_trajectory_3d.html`（推荐Chrome、Firefox、Edge、Safari）
2. 无需安装或服务器 - 完全在浏览器中运行

### 加载数据

**方式1：TXT文件格式**
```
# 格式：FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ
0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
1, 0, 0.9999, 0.001, 0.002, 0.001, 0.1, 0.05, 0.02
2, 0, 0.9998, 0.002, 0.004, 0.002, 0.2, 0.10, 0.04
...
```

**方式2：NPY文件格式**
- 使用 `numpy.save()` 导出，形状为 `(N, 7)` 或 `(N, 8)`
- 列：`[tx, ty, tz, qx, qy, qz, qw]` 或 `[timestamp, tx, ty, tz, qx, qy, qz, qw]`

Python导出示例：
```python
import numpy as np
import pypose as pp

# 假设你有SE3张量格式的位姿
poses = ...  # 形状：(N, 7) - [tx, ty, tz, qx, qy, qz, qw]
np.save('trajectory.npy', poses.numpy())
```

### 工作流程

1. **加载数据**：点击"Choose File"选择轨迹文件
2. **探索**：使用鼠标旋转、平移和缩放3D视图
3. **分析**：查看总距离和位姿信息
4. **选择范围**：设置起始/结束帧以分析特定段
5. **播放**：使用播放控制来动画展示轨迹
6. **导出**（可选）：截取有趣视图的屏幕截图

## 数据格式详情

### TXT文件格式
```
FRAME_ID, RIG_ID, QW, QX, QY, QZ, TX, TY, TZ
```
- `FRAME_ID`：帧索引（整数）
- `RIG_ID`：相机设备标识符（整数）
- `QW, QX, QY, QZ`：四元数旋转（W是标量部分）
- `TX, TY, TZ`：平移，单位米

以 `#` 开头的行被视为注释并忽略。

### NPY文件格式

**形状 (N, 7)**：直接SE3格式
```
[tx, ty, tz, qx, qy, qz, qw]
```

**形状 (N, 8)**：带时间戳的SE3格式（MAC-VO使用）
```
[timestamp, tx, ty, tz, qx, qy, qz, qw]
```

## 与MAC-VO集成

此可视化工具设计为与MAC-VO输出无缝协作：

```python
# 运行MAC-VO实验后
from Utility.Sandbox import Sandbox

box = Sandbox.load("path/to/experiment/output")

# 在可视化工具中加载poses.npy文件
# 文件位于：box.path("poses.npy")
```

该工具自动处理MAC-VO的 `IOdometry.receive_frames()` 方法产生的 `(N, 8)` 格式。

## 快速开始

### 1. 尝试示例数据

最简单的入门方式是使用提供的示例轨迹：

1. 在Web浏览器中打开 `camera_trajectory_3d.html`
2. 点击"Choose File"并选择 `sample_trajectory.txt`
3. 探索3D可视化！

### 2. 从MAC-VO导出数据

运行MAC-VO实验后，可以导出轨迹：

```bash
# 导出为TXT格式（人类可读）
python Visualization/export_trajectory.py \
    --sandbox /path/to/your/experiment/sandbox \
    --output my_trajectory.txt

# 或导出为NPY格式（二进制，文件更小）
python Visualization/export_trajectory.py \
    --sandbox /path/to/your/experiment/sandbox \
    --output my_trajectory.npy
```

## 性能

- **优化渲染**：使用Three.js WebGL进行硬件加速
- **大数据集**：高效处理1000+帧
- **标记采样**：对于非常长的轨迹自动采样标记（显示约50个标记）
- **流畅播放**：可调节帧率高达100 FPS

## 浏览器兼容性

- ✅ Chrome/Chromium（推荐）
- ✅ Firefox
- ✅ Safari
- ✅ Edge

需要浏览器支持：
- WebGL
- ES6+ JavaScript
- File API

## 故障排除

### 文件无法加载
- 检查文件格式是否符合预期结构
- 确保文件中没有损坏或无效的数据
- 查看浏览器控制台中的具体错误消息

### 性能问题
- 降低播放速度
- 使用具有良好WebGL支持的浏览器
- 关闭其他标签页/应用程序

### 显示问题
- 尝试重置视图
- 检查浏览器设置中是否启用了WebGL
- 更新显卡驱动程序

## 技术细节

### 依赖项（通过CDN）
- **Three.js** r128：3D渲染引擎
- **OrbitControls**：相机控制系统

### 架构
- **TrajectoryData**：位姿存储和计算的数据模型
- **TrajectoryVisualizer**：Three.js场景管理和渲染
- **App**：UI控制器和文件处理

### 坐标系统
- 右手坐标系
- Y轴向上约定（标准Three.js）
- 四元数格式：(x, y, z, w)

## 更多文档

- 详细文档：[`README.md`](README.md)（英文）
- 快速指南：[`USAGE.md`](USAGE.md)（英文）
- 实现总结：[`.summary.md`](.summary.md)（英文）

## 许可证

此可视化工具是MAC-VO项目的一部分，遵循相同的许可证。

## 支持

遇到问题或疑问：
1. 查看本README寻找解决方案
2. 检查浏览器控制台中的错误
3. 验证文件格式正确性
4. 在MAC-VO仓库中提出issue
