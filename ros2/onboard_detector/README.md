# Onboard Detector 功能包详细解读

## 目录

- [一、功能包概述](#一功能包概述)
- [二、系统架构](#二系统架构)
- [三、核心算法详解](#三核心算法详解)
- [四、ROS2 接口详解](#四ros2-接口详解)
- [五、配置文件参数说明](#五配置文件参数说明)
- [六、启动与使用](#六启动与使用)
- [七、技术亮点](#七技术亮点)

---

## 一、功能包概述

### 1.1 核心功能与设计目标

`onboard_detector` 是 NavRL 项目的**机载感知模块**，负责实时检测和跟踪周围环境中的动态障碍物。其核心功能包括：

| 功能模块 | 说明 |
|---------|------|
| **深度图像处理** | 处理深度相机数据，提取 3D 点云信息 |
| **障碍物检测** | 基于 UV-Detector 和 DBSCAN 的双检测器架构 |
| **多目标跟踪** | 使用卡尔曼滤波器实现障碍物连续跟踪 |
| **动态分类** | 区分静态和动态障碍物 |
| **视觉检测集成** | 集成 YOLO 目标检测，支持语义信息 |

### 1.2 系统定位

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NavRL 感知系统架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Onboard Detector                                  │  │
│   │                      机载感知模块                                     │  │
│   │                                                                     │  │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │  │
│   │  │  传感器输入   │  │   检测算法    │  │      跟踪与分类          │  │  │
│   │  │              │  │              │  │                         │  │  │
│   │  │ • 深度图像    │  │ • UV-Detector│  │ • Kalman Filter         │  │  │
│   │  │ • 位姿/里程计 │  │ • DBSCAN     │  │ • Data Association      │  │  │
│   │  │ • RGB 图像    │  │ • YOLO       │  │ • Dynamic Classification│  │  │
│   │  │              │  │              │  │                         │  │  │
│   │  └──────┬───────┘  └──────┬───────┘  └───────────┬─────────────┘  │  │
│   │         │                 │                      │                │  │
│   │         └─────────────────┴──────────────────────┘                │  │
│   │                           │                                       │  │
│   │                           ▼                                       │  │
│   │                  ┌─────────────────┐                              │  │
│   │                  │  动态障碍物输出   │                              │  │
│   │                  │  (位置/速度/尺寸) │                              │  │
│   │                  └─────────────────┘                              │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │              Navigation Runner (导航执行模块)                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、系统架构

### 2.1 模块组成

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Onboard Detector 架构详解                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Dynamic Detector Node (C++)                             │   │
│  │                    动态检测主节点                                     │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  1. 数据同步与预处理 (Message Filters)                        │   │   │
│  │  │     • 深度图像 + 位姿 时间同步                                │   │   │
│  │  │     • 深度图像 + 里程计 时间同步                              │   │   │
│  │  │     • RGB 图像独立订阅                                       │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  2. 双检测器架构 (Dual Detector)                              │   │   │
│  │  │                                                             │   │   │
│  │  │   UV-Detector              DBSCAN                           │   │   │
│  │  │   (快速检测)               (精确聚类)                        │   │   │
│  │  │        │                       │                            │   │   │
│  │  │        ▼                       ▼                            │   │   │
│  │  │   ┌─────────┐            ┌─────────┐                        │   │   │
│  │  │   │深度图像  │            │ 点云数据 │                        │   │   │
│  │  │   │→ 2D检测 │            │ → 3D聚类 │                        │   │   │
│  │  │   │→ 快速   │            │ → 精确   │                        │   │   │
│  │  │   └────┬────┘            └────┬────┘                        │   │   │
│  │  │        │                       │                            │   │   │
│  │  │        └───────────┬───────────┘                            │   │   │
│  │  │                    ▼                                       │   │   │
│  │  │            障碍物候选框合并                                  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  3. 多目标跟踪 (Multi-Object Tracking)                        │   │   │
│  │  │     • 卡尔曼滤波器预测                                        │   │   │
│  │  │     • 数据关联 (相似度匹配)                                   │   │   │
│  │  │     • 轨迹管理 (创建/更新/删除)                               │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                              ▼                                      │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  4. 动态分类 (Dynamic Classification)                         │   │   │
│  │  │     • 速度阈值判断                                            │   │   │
│  │  │     • 投票机制提高鲁棒性                                       │   │   │
│  │  │     • 一致性检查                                              │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              YOLO Detector Node (Python)                             │   │
│  │                    视觉检测节点                                       │   │
│  │                                                                     │   │
│  │  • 基于 ShuffleNetV2 的轻量级 YOLO 网络                             │   │
│  │  • 支持 COCO 数据集类别 (人、车等)                                   │   │
│  │  • 与深度信息融合，提供 3D 检测结果                                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 文件结构

```
onboard_detector/
├── CMakeLists.txt                    # 构建配置
├── package.xml                       # 包依赖
├── cfg/                              # 配置文件
│   ├── dynamic_detector_param.yaml   # 动态检测器参数
│   └── yolo_detector_param.yaml      # YOLO 检测器参数
├── include/onboard_detector/         # C++ 头文件
│   ├── dynamicDetector.h/.cpp        # 主检测器实现
│   ├── detectors/
│   │   ├── uvDetector.h/.cpp         # UV-Detector 实现
│   │   └── dbscan.h/.cpp             # DBSCAN 实现
│   └── tracking/
│       └── kalmanFilter.h/.cpp       # 卡尔曼滤波器
├── scripts/                          # Python 脚本
│   ├── yolo_detector_node.py         # YOLO 节点
│   ├── module/
│   │   ├── detector.py               # YOLO 检测器
│   │   ├── shufflenetv2.py           # 骨干网络
│   │   └── shufflenetv2.pth          # 预训练权重
│   └── utils/
│       └── tool.py                   # 工具函数
├── src/
│   └── dynamic_detector_node.cpp     # 主节点入口
├── srv/
│   └── GetDynamicObstacles.srv       # 服务定义
└── launch/
    └── dynamic_detector.launch.py    # 启动文件
```

---

## 三、核心算法详解

### 3.1 UV-Detector 算法

**UV-Detector** 是一种快速的基于深度图像的障碍物检测算法。

**核心原理**：

```
深度图像 (Depth Image)
    │
    ├──► U 方向扫描 (水平) ──┐
    │                         ├──► 检测深度不连续区域
    └──► V 方向扫描 (垂直) ──┘         │
                                       ▼
                              2D 边界框 (Bounding Boxes)
                                       │
                                       ▼
                              3D 点云投影 → 计算 3D 尺寸和位置
```

**算法步骤**：

1. **深度图像预处理**
   - 去除无效深度值
   - 降采样提高处理速度

2. **UV 方向扫描**
   - U 方向（水平）：检测列间的深度突变
   - V 方向（垂直）：检测行间的深度突变
   - 标记潜在的障碍物区域

3. **2D 检测框生成**
   - 对标记区域进行连通域分析
   - 提取 2D 边界框

4. **3D 信息恢复**
   - 将 2D 框内的像素反投影到 3D 空间
   - 计算 3D 边界框（位置、尺寸）

**代码实现**（C++）：
```cpp
class UVdetector {
public:
    // 主检测函数
    void detect(const cv::Mat& depthImage, 
                const Eigen::Matrix3d& intrinsicMatrix,
                std::vector<BoundingBox3D>& obstacles);
    
private:
    // U 方向扫描
    void uScan(const cv::Mat& depthImage, cv::Mat& uMask);
    
    // V 方向扫描
    void vScan(const cv::Mat& depthImage, cv::Mat& vMask);
    
    // 3D 投影
    void project2DTo3D(const std::vector<cv::Rect>& boxes2D,
                       const cv::Mat& depthImage,
                       std::vector<BoundingBox3D>& boxes3D);
};
```

### 3.2 DBSCAN 聚类算法

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) 是一种基于密度的聚类算法，用于从点云中提取障碍物。

**核心概念**：

| 概念 | 说明 |
|------|------|
| **核心点** | 在半径 ε 内至少有 MinPts 个邻居的点 |
| **边界点** | 在半径 ε 内邻居少于 MinPts，但在核心点邻域内的点 |
| **噪声点** | 既不是核心点也不是边界点的点 |
| **密度可达** | 从核心点出发，通过一系列核心点连接的点 |

**算法流程**：

```
点云输入
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. 体素网格降采样                                        │
│    • 减少点云数量，提高处理速度                           │
│    • 体素大小: 0.1m x 0.1m x 0.1m                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. 地面点去除                                             │
│    • 基于高度阈值 (ground_height: 0.1m)                  │
│    • 去除地面以下的点                                     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. DBSCAN 聚类                                            │
│    • 对每个未访问点：                                      │
│      - 如果 ε 邻域内点数 ≥ MinPts → 创建新簇              │
│      - 否则标记为噪声                                     │
│    • 扩展簇：将密度可达的点加入簇                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 4. 边界框提取                                             │
│    • 计算每个簇的 3D 边界框                               │
│    • 输出: 位置、尺寸、方向                               │
└─────────────────────────────────────────────────────────┘
```

**参数配置**：
```yaml
dbscan_min_points_cluster: 20      # MinPts: 最小聚类点数
dbscan_search_range_epsilon: 0.1   # ε: 邻域搜索半径 (米)
voxel_occupied_thresh: 5           # 体素占用阈值
```

### 3.3 多目标跟踪算法

**跟踪系统架构**：

```
当前帧检测
    │
    ├──► 卡尔曼滤波器预测 ──► 预测位置
    │                              │
    │                              ▼
    │                       数据关联
    │                              │
    │         ┌────────────────────┼────────────────────┐
    │         │                    │                    │
    │         ▼                    ▼                    ▼
    │    匹配成功              新目标               目标丢失
    │         │                    │                    │
    │         ▼                    ▼                    ▼
    │    更新滤波器           创建新轨迹            删除轨迹
    │         │                    │                    │
    │         └────────────────────┼────────────────────┘
    │                              │
    ▼                              ▼
跟踪结果 (带 ID 的障碍物轨迹)
```

**卡尔曼滤波器**：

- **状态向量**: `[x, y, z, vx, vy, vz, sx, sy, sz]`
  - 位置 (x, y, z)
  - 速度 (vx, vy, vz)
  - 尺寸 (sx, sy, sz)

- **预测模型**: 恒定速度模型
  ```
  x(k+1) = x(k) + vx(k) * dt
  y(k+1) = y(k) + vy(k) * dt
  z(k+1) = z(k) + vz(k) * dt
  ```

**数据关联**：

使用**相似度度量**进行匹配：
```cpp
// 相似度计算考虑：
// 1. 位置距离
// 2. 尺寸差异
// 3. 速度一致性

similarity = w1 * position_distance + 
             w2 * size_difference + 
             w3 * velocity_consistency;

// 阈值判断
if (similarity < similarity_threshold) {
    // 匹配成功
} else if (similarity < retrack_threshold) {
    // 可能重新跟踪
}
```

### 3.4 动态分类算法

**分类逻辑**：

```
对每个跟踪目标：
    │
    ├──► 计算速度大小: v = sqrt(vx² + vy² + vz²)
    │
    ├──► 速度阈值判断
    │    │
    │    ├──► v > dynamic_velocity_threshold (0.15 m/s)
    │    │       └──► 投票 +1 (动态)
    │    │
    │    └──► v ≤ threshold
    │            └──► 投票 -1 (静态)
    │
    ├──► 滑动窗口投票
    │    │
    │    └──► 统计最近 history_size (100) 帧的投票
    │
    └──► 分类决策
         │
         ├──► 动态票数 / 总数 > dynamic_voting_threshold (0.8)
         │       └──► 标记为动态
         │
         └──► 否则
                 └──► 标记为静态
```

**一致性检查**：

```cpp
// 动态一致性检查
if (连续 dynamic_consistency_threshold (5) 帧分类结果一致) {
    // 确认分类结果，避免抖动
    confirm_classification();
}
```

---

## 四、ROS2 接口详解

### 4.1 订阅的话题 (Subscribers)

| 话题名 | 消息类型 | 说明 | 来源 |
|--------|----------|------|------|
| `/camera/depth/image_raw` | `sensor_msgs/Image` | 深度图像 | 深度相机 |
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB 彩色图像 | RGB 相机 |
| `/mavros/vision_pose/pose` | `geometry_msgs/PoseStamped` | 视觉位姿 | 定位系统 |
| `/mavros/odometry/out` | `nav_msgs/Odometry` | 里程计 | 飞控 |
| `/yolo_detector/bounding_boxes` | `vision_msgs/Detection2DArray` | YOLO 检测结果 | YOLO 节点 |

### 4.2 发布的话题 (Publishers)

| 话题名 | 消息类型 | 说明 |
|--------|----------|------|
| `/onboard_detector/dynamic_obstacles` | `visualization_msgs/MarkerArray` | 动态障碍物可视化 |
| `/onboard_detector/dynamic_points` | `sensor_msgs/PointCloud2` | 动态障碍物点云 |
| `/onboard_detector/bounding_boxes` | `visualization_msgs/MarkerArray` | 3D 边界框可视化 |

### 4.3 服务 (Services)

| 服务名 | 服务类型 | 说明 |
|--------|----------|------|
| `/onboard_detector/get_dynamic_obstacles` | `onboard_detector/srv/GetDynamicObstacles` | 获取动态障碍物信息 |

**服务定义** (`GetDynamicObstacles.srv`)：

```srv
# Request
geometry_msgs/Point current_position    # 查询位置
float64 range                           # 查询范围 (米)
---
# Response
geometry_msgs/Vector3[] position        # 障碍物位置列表
geometry_msgs/Vector3[] velocity        # 障碍物速度列表
geometry_msgs/Vector3[] size            # 障碍物尺寸列表
```

### 4.4 消息同步机制

使用 ROS2 `message_filters` 实现多传感器时间同步：

```cpp
// 深度图像 + 位姿同步
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,           // 深度图像
    geometry_msgs::msg::PoseStamped    // 位姿
> depthPoseSync;

std::shared_ptr<message_filters::Synchronizer<depthPoseSync>> depthPoseSync_;

// 深度图像 + 里程计同步
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,           // 深度图像
    nav_msgs::msg::Odometry            // 里程计
> depthOdomSync;

std::shared_ptr<message_filters::Synchronizer<depthOdomSync>> depthOdomSync_;
```

**同步策略**：
- **ApproximateTime**: 允许时间戳不完全匹配，使用最近邻匹配
- **队列大小**: 10 帧
- **最大时间差**: 0.1 秒

---

## 五、配置文件参数说明

### 5.1 动态检测器参数 (`dynamic_detector_param.yaml`)

```yaml
# =============================================================================
# DBSCAN 聚类参数
# =============================================================================
ground_height: 0.1                    # 地面高度阈值 (米)
                                      # 低于此高度的点被视为地面点

dbscan_min_points_cluster: 20         # DBSCAN 最小聚类点数 (MinPts)
                                      # 低于此数的簇被视为噪声

dbscan_search_range_epsilon: 0.1      # DBSCAN 邻域搜索半径 ε (米)
                                      # 决定点的邻域范围

voxel_occupied_thresh: 5              # 体素占用阈值
                                      # 体素内点数超过此值才被认为占用

# =============================================================================
# 多目标跟踪参数
# =============================================================================
history_size: 100                     # 历史帧数
                                      # 用于动态分类投票的滑动窗口大小

prediction_size: 20                   # 预测帧数
                                      # 卡尔曼滤波器预测步数

similarity_threshold: 0.02            # 数据关联相似度阈值
                                      # 低于此值认为匹配成功

retrack_similarity_threshold: 0.015   # 重新跟踪阈值
                                      # 低于此值可能重新激活丢失目标

# =============================================================================
# 动态分类参数
# =============================================================================
dynamic_velocity_threshold: 0.15      # 动态速度阈值 (m/s)
                                      # 超过此速度被认为是动态的

dynamic_voting_threshold: 0.8         # 动态投票阈值
                                      # 动态票数比例超过此值才确认

dynamic_consistency_threshold: 5      # 动态一致性阈值 (帧数)
                                      # 连续多帧一致才确认分类

# =============================================================================
# YOLO 集成参数
# =============================================================================
yolo_enabled: true                    # 是否启用 YOLO 检测
yolo_confidence_threshold: 0.5        # YOLO 置信度阈值
yolo_depth_range: 10.0                # YOLO 深度有效范围 (米)
```

### 5.2 YOLO 检测器参数 (`yolo_detector_param.yaml`)

```yaml
yolo_detector_node:
  ros__parameters:
    # 模型配置
    model_path: "scripts/weights/weight_AP05_0.253207_280-epoch.pth"
    config_path: "scripts/config/coco.yaml"
    
    # 输入配置
    image_topic: "/camera/color/image_raw"
    input_size: [640, 640]              # 输入图像尺寸
    
    # 检测配置
    confidence_threshold: 0.5
    nms_threshold: 0.4                  # 非极大值抑制阈值
    
    # 类别过滤
    target_classes: [0, 1, 2, 3, 5, 7]  # 人、自行车、汽车、摩托车、公交车、卡车
```

---

## 六、启动与使用

### 6.1 启动动态检测器

```bash
# 启动完整检测系统
ros2 launch onboard_detector dynamic_detector.launch.py

# 单独启动动态检测节点
ros2 run onboard_detector dynamic_detector_node

# 单独启动 YOLO 检测节点
ros2 run onboard_detector yolo_detector_node.py
```

### 6.2 启动文件配置

```python
# dynamic_detector.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 参数文件路径
    param_path = os.path.join(
        get_package_share_directory('onboard_detector'),
        'cfg',
        'dynamic_detector_param.yaml'
    )
    
    # 动态检测节点
    dynamic_detector_node = Node(
        package='onboard_detector',
        executable='dynamic_detector_node',
        name='dynamic_detector',
        output='screen',
        parameters=[param_path]
    )
    
    # YOLO 检测节点
    yolo_detector_node = Node(
        package='onboard_detector',
        executable='yolo_detector_node.py',
        name='yolo_detector',
        output='screen'
    )
    
    return LaunchDescription([
        dynamic_detector_node,
        yolo_detector_node
    ])
```

### 6.3 查询动态障碍物

```python
# Python 客户端示例
import rclpy
from rclpy.node import Node
from onboard_detector.srv import GetDynamicObstacles
from geometry_msgs.msg import Point

class DynamicObstacleClient(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_client')
        self.client = self.create_client(
            GetDynamicObstacles, 
            '/onboard_detector/get_dynamic_obstacles'
        )
        
    def get_obstacles(self, position, range_m):
        request = GetDynamicObstacles.Request()
        request.current_position = Point(
            x=position[0], 
            y=position[1], 
            z=position[2]
        )
        request.range = range_m
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        return {
            'positions': response.position,
            'velocities': response.velocity,
            'sizes': response.size
        }
```

---

## 七、技术亮点

| 特点 | 说明 |
|------|------|
| **双检测器架构** | UV-Detector (快速) + DBSCAN (精确)，兼顾速度与精度 |
| **多目标跟踪** | 卡尔曼滤波 + 数据关联，实现稳定跟踪 |
| **动态分类** | 基于速度投票机制，鲁棒区分动静障碍物 |
| **多模态融合** | 深度图像 + RGB 图像 + YOLO 语义信息 |
| **实时性能** | 30Hz 处理频率，满足实时导航需求 |
| **ROS2 原生** | 基于 ROS2 Humble，支持 DDS 通信 |

---

**文档版本**: 1.0  
**最后更新**: 2025-03-23  
**维护者**: NavRL 开发团队
