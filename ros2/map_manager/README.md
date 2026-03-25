# Map Manager 功能包详细解读

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

`map_manager` 是 NavRL 项目的**地图管理模块**，负责构建、维护和提供环境地图信息。它实现了两种主要的地图表示方式：

| 地图类型 | 说明 | 用途 |
|---------|------|------|
| **Occupancy Map** (占据地图) | 基于体素的占据概率地图 | 碰撞检测、路径规划 |
| **ESDF Map** (欧氏距离场) | 到最近障碍物的有符号距离 | 梯度计算、优化控制 |

核心功能包括：

| 功能模块 | 说明 |
|---------|------|
| **多传感器融合建图** | 支持深度相机和 LiDAR 点云 |
| **概率占据更新** | 基于对数几率的贝叶斯更新 |
| **射线投射** | 3D 空间射线与体素求交 |
| **地图膨胀** | 考虑机器人尺寸的障碍物膨胀 |
| **静态障碍物聚类** | 从点云提取静态障碍物边界框 |
| **ESDF 计算** | 欧氏距离变换算法 |

### 1.2 系统定位

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NavRL 地图系统架构                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         传感器输入                                   │  │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│   │  │  深度相机     │  │  LiDAR点云   │  │  位姿/里程计  │              │  │
│   │  │  (Depth)     │  │  (PointCloud)│  │  (Pose/Odom) │              │  │
│   │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │  │
│   │         │                 │                 │                       │  │
│   └─────────┼─────────────────┼─────────────────┼───────────────────────┘  │
│             │                 │                 │                           │
│             ▼                 ▼                 ▼                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Map Manager (地图管理器)                        │  │
│   │                                                                     │  │
│   │  ┌─────────────────────────────────────────────────────────────┐   │  │
│   │  │                  Occupancy Map (占据地图)                    │   │  │
│   │  │                                                             │   │  │
│   │  │  • 概率占据更新 (Log-odds)                                  │   │  │
│   │  │  • 射线投射 (Raycasting)                                    │   │  │
│   │  │  • 地图膨胀 (Inflation)                                     │   │  │
│   │  │  • 局部地图更新 (Local Update)                              │   │  │
│   │  │                                                             │   │  │
│   │  └─────────────────────────────────────────────────────────────┘   │  │
│   │                              │                                      │  │
│   │                              ▼                                      │  │
│   │  ┌─────────────────────────────────────────────────────────────┐   │  │
│   │  │                    ESDF Map (欧氏距离场)                      │   │  │
│   │  │                                                             │   │  │
│   │  │  • 欧氏距离变换 (EDT)                                        │   │  │
│   │  │  • 三线性插值 (Trilinear Interpolation)                      │   │  │
│   │  │  • 梯度计算 (Gradient)                                       │   │  │
│   │  │                                                             │   │  │
│   │  └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      服务接口 (Services)                             │  │
│   │                                                                     │  │
│   │  • /occupancy_map/raycast          →  LiDAR射线检测                 │  │
│   │  • /occupancy_map/check_pos_collision → 位置碰撞检测                │  │
│   │  • /occupancy_map/get_static_obstacles → 获取静态障碍物             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                   Navigation Runner (导航执行)                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、系统架构

### 2.1 模块组成

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Map Manager 架构详解                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Occupancy Map Node (C++)                                │   │
│  │                    占据地图节点                                       │   │
│  │                                                                     │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  1. 数据同步与预处理                                           │   │   │
│   │  │     • 深度图像 + 位姿 时间同步                                 │   │   │
│   │  │     • 点云 + 位姿 时间同步                                     │   │   │
│   │  │     • 相机内参和外参转换                                       │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                      │   │
│   │                              ▼                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  2. 点云生成与滤波                                            │   │   │
│   │  │     • 深度图像反投影到3D点云                                   │   │   │
│   │  │     • 距离滤波 (min/max distance)                             │   │   │
│   │  │     • 边缘滤波 (filter margin)                                │   │   │
│   │  │     • 降采样 (skip pixel)                                     │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                      │   │
│   │                              ▼                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  3. 占据概率更新 (Log-odds)                                    │   │   │
│   │  │                                                             │   │   │
│   │  │   射线投射: 传感器位置 → 点                                   │   │   │
│   │  │        │                                                    │   │   │
│   │  │        ├──► 经过的体素: 标记为空闲 (p_miss)                   │   │   │
│   │  │        └──► 终点体素: 标记为占据 (p_hit)                      │   │   │
│   │  │                                                             │   │   │
│   │  │   贝叶斯更新:                                               │   │   │
│   │  │   log_odd(t) = log_odd(t-1) + log_odd(hit/miss)             │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                      │   │
│   │                              ▼                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  4. 地图膨胀 (Inflation)                                      │   │   │
│   │  │     • 考虑机器人尺寸 (robot_size)                             │   │   │
│   │  │     • 膨胀半径 = 机器人半径 + 安全距离                         │   │   │
│   │  │     • 距离变换算法计算膨胀区域                                 │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                              │                                      │   │
│   │                              ▼                                      │   │
│   │  ┌─────────────────────────────────────────────────────────────┐   │   │
│   │  │  5. 静态障碍物聚类                                            │   │   │
│   │  │     • DBSCAN 聚类点云                                         │   │   │
│   │  │     • K-means 细化边界框                                      │   │   │
│   │  │     • 旋转矩形拟合                                            │   │   │
│   │  └─────────────────────────────────────────────────────────────┘   │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │              ESDF Map Node (C++)                                     │   │
│   │                   ESDF地图节点                                       │   │
│   │                                                                     │   │
│   │  继承自 OccupancyMap，添加 ESDF 计算功能                             │   │
│   │                                                                     │   │
│   │  • 欧氏距离变换 (EDT) 算法                                          │   │
│   │  • 一维距离变换 (1D EDT) 三次遍历                                    │   │
│   │  • 三线性插值查询                                                   │   │
│   │                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 文件结构

```
map_manager/
├── CMakeLists.txt                    # 构建配置
├── package.xml                       # 包依赖
├── cfg/
│   └── map_param.yaml                # 地图参数配置
├── include/map_manager/              # C++ 头文件
│   ├── occupancyMap.h/.cpp           # 占据地图实现
│   ├── ESDFMap.h/.cpp                # ESDF 地图实现
│   ├── raycast/
│   │   ├── raycast.h/.cpp            # 射线投射算法
│   ├── clustering/
│   │   ├── DBSCAN.h                  # DBSCAN 聚类
│   │   ├── Kmeans.h/.cpp             # K-means 聚类
│   │   └── obstacleClustering.h/.cpp # 障碍物聚类
├── src/
│   ├── occupancy_map_node.cpp        # 占据地图节点入口
│   └── esdf_map_node.cpp             # ESDF 地图节点入口
├── srv/                              # 服务定义
│   ├── RayCast.srv                   # 射线检测服务
│   ├── CheckPosCollision.srv         # 碰撞检测服务
│   └── GetStaticObstacles.srv        # 获取静态障碍物服务
└── launch/
    ├── occupancy_map.launch.py       # 占据地图启动文件
    ├── esdf_map.launch.py            # ESDF 地图启动文件
    └── rviz.launch.py                # RViz 启动文件
```

---

## 三、核心算法详解

### 3.1 概率占据地图 (Occupancy Map)

**对数几率 (Log-odds) 更新**：

占据地图使用对数几率表示每个体素的占据概率：

```
log_odd = log(p / (1 - p))

其中:
  p = 占据概率
  log_odd > 0  →  倾向于占据
  log_odd < 0  →  倾向于空闲
  log_odd = 0  →  未知状态
```

**贝叶斯更新公式**：

```
log_odd(t) = log_odd(t-1) + log_odd(measurement)

log_odd(hit)  = log(p_hit / (1 - p_hit))    (观测到占据)
log_odd(miss) = log(p_miss / (1 - p_miss))  (观测到空闲)
```

**参数配置**：
```yaml
p_hit: 0.70     # 命中概率 (观测到占据)
p_miss: 0.35    # 未命中概率 (观测到空闲)
p_min: 0.12     # 最小占据概率 (截断值)
p_max: 0.97     # 最大占据概率 (截断值)
p_occ: 0.80     # 占据阈值 (大于此值认为占据)
```

**更新流程**：

```
传感器数据 (深度图像/点云)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. 点云生成                                              │
│    • 深度图像反投影: (u, v, d) → (x, y, z)              │
│    • 相机坐标系 → 世界坐标系变换                          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. 射线投射 (Raycasting)                                 │
│                                                         │
│    对每个点云点:                                          │
│    • 起点: 传感器位置                                     │
│    • 终点: 点云点位置                                     │
│    • 使用 3D DDA 算法遍历射线经过的体素                   │
│                                                         │
│    经过的体素: 更新为空闲 (p_miss)                        │
│    终点体素: 更新为占据 (p_hit)                           │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. 概率更新                                              │
│    occupancy[voxel] += log_odd(hit/miss)                 │
│    occupancy[voxel] = clamp(occupancy[voxel], p_min, p_max)│
└─────────────────────────────────────────────────────────┘
```

### 3.2 射线投射算法 (Raycasting)

**3D DDA (Digital Differential Analyzer) 算法**：

```cpp
class RayCaster {
public:
    // 设置射线起点和终点
    bool setInput(const Eigen::Vector3d& start, const Eigen::Vector3d& end);
    
    // 逐步遍历射线经过的体素
    bool step(Eigen::Vector3d& ray_pt);
    
private:
    // 当前体素坐标
    int x_, y_, z_;
    
    // 步进方向 (+1 或 -1)
    int stepX_, stepY_, stepZ_;
    
    // 到下一个体素边界的距离
    double tMaxX_, tMaxY_, tMaxZ_;
    
    // 穿越一个体素所需的距离
    double tDeltaX_, tDeltaY_, tDeltaZ_;
};
```

**算法原理**：

```
射线: P(t) = Start + t * (End - Start), t ∈ [0, 1]

对于每个坐标轴 (x, y, z):
  • 计算到下一个体素边界的距离 tMax
  • 计算穿越一个体素所需的距离 tDelta
  
遍历:
  1. 选择 tMax 最小的轴 (最先到达边界)
  2. 沿该轴步进到下一个体素
  3. 更新 tMax += tDelta
  4. 重复直到到达终点
```

### 3.3 地图膨胀 (Map Inflation)

**膨胀原理**：

```
原始占据地图                    膨胀后地图
┌───┬───┬───┐                 ┌───┬───┬───┐
│   │   │   │                 │ I │ I │ I │  I = 膨胀区域
├───┼───┼───┤                 ├───┼───┼───┤
│   │ O │   │    ─────►       │ I │ O │ I │  O = 原始障碍物
├───┼───┼───┤                 ├───┼───┼───┤
│   │   │   │                 │ I │ I │ I │
└───┴───┴───┘                 └───┴───┴───┘

膨胀半径 = 机器人半径 + 安全距离
```

**距离变换算法**：

```cpp
// 计算每个空闲体素到最近障碍物的距离
for each voxel in map:
    if voxel is occupied:
        distance[voxel] = 0
    else:
        distance[voxel] = min distance to any occupied voxel
        
// 标记膨胀区域
for each voxel:
    if distance[voxel] < inflation_radius:
        inflated_map[voxel] = occupied
```

### 3.4 欧氏距离场 (ESDF)

**ESDF (Euclidean Signed Distance Field)** 表示每个体素到最近障碍物的有符号距离：

```
ESDF(x) = {
    +distance,  如果 x 在自由空间 (到最近障碍物的距离)
    -distance,  如果 x 在障碍物内部 (到最近自由空间的距离)
    0,          如果 x 在障碍物表面
}
```

**欧氏距离变换 (EDT) 算法**：

使用 **Felzenszwalb and Huttenlocher** 算法，分三个维度进行一维距离变换：

```
// 3D ESDF 计算流程
for z in range(Z):
    for y in range(Y):
        // 1. X轴方向一维EDT
        EDT_1D along X
        
for z in range(Z):
    for x in range(X):
        // 2. Y轴方向一维EDT
        EDT_1D along Y
        
for y in range(Y):
    for x in range(X):
        // 3. Z轴方向一维EDT
        EDT_1D along Z
```

**一维距离变换**：

```cpp
// 对于每个体素 f[q]，计算:
// g[q] = min_q' (f[q'] + (q - q')²)

template <typename F_get_val, typename F_set_val>
void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim) {
    int n = end - start;
    std::vector<int> v(n);
    std::vector<double> z(n + 1);
    
    int k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    
    for (int q = 1; q < n; q++) {
        double s = ((f_get_val(q) + q*q) - (f_get_val(v[k]) + v[k]*v[k])) / (2*q - 2*v[k]);
        while (s <= z[k]) {
            k--;
            s = ((f_get_val(q) + q*q) - (f_get_val(v[k]) + v[k]*v[k])) / (2*q - 2*v[k]);
        }
        k++;
        v[k] = q;
        z[k] = s;
        z[k+1] = +INF;
    }
    
    k = 0;
    for (int q = start; q < end; q++) {
        while (z[k+1] < q) k++;
        double val = (q - v[k]) * (q - v[k]) + f_get_val(v[k]);
        f_set_val(q, val);
    }
}
```

**三线性插值查询**：

```cpp
// 对于非体素中心位置的查询，使用三线性插值获得更精确的距离值
double getDistanceTrilinear(const Eigen::Vector3d& pos) {
    // 找到周围的8个体素
    Eigen::Vector3d idxd = pos / resolution;
    Eigen::Vector3i idx = idxd.cast<int>();
    
    // 计算插值权重
    double x_ratio = idxd.x() - idx.x();
    double y_ratio = idxd.y() - idx.y();
    double z_ratio = idxd.z() - idx.z();
    
    // 三线性插值
    double c00 = esdf[idx] * (1 - x_ratio) + esdf[idx + (1,0,0)] * x_ratio;
    double c01 = esdf[idx + (0,0,1)] * (1 - x_ratio) + esdf[idx + (1,0,1)] * x_ratio;
    double c10 = esdf[idx + (0,1,0)] * (1 - x_ratio) + esdf[idx + (1,1,0)] * x_ratio;
    double c11 = esdf[idx + (0,1,1)] * (1 - x_ratio) + esdf[idx + (1,1,1)] * x_ratio;
    
    double c0 = c00 * (1 - y_ratio) + c10 * y_ratio;
    double c1 = c01 * (1 - y_ratio) + c11 * y_ratio;
    
    return c0 * (1 - z_ratio) + c1 * z_ratio;
}
```

### 3.5 静态障碍物聚类

**聚类流程**：

```
占据地图点云
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. DBSCAN 聚类                                           │
│    • 密度-based 聚类                                      │
│    • 参数: eps=0.5m, minPts=15                            │
│    • 输出: 初始聚类簇                                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 2. K-means 细化                                          │
│    • 对每个簇进行 K-means 细分                            │
│    • 迭代优化边界框                                        │
│    • 参数: treeLevel=3, kmeansIterNum=10                  │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 3. 旋转矩形拟合                                          │
│    • 尝试多个旋转角度 (angleDiscreteNum=20)               │
│    • 选择面积最小的边界框                                  │
│    • 输出: (centroid, size, yaw)                          │
└─────────────────────────────────────────────────────────┘
```

---

## 四、ROS2 接口详解

### 4.1 订阅的话题 (Subscribers)

| 话题名 | 消息类型 | 说明 | 来源 |
|--------|----------|------|------|
| `/unitree_go2/front_cam/depth_image` | `sensor_msgs/Image` | 深度图像 | 深度相机 |
| `/unitree_go2/lidar/point_cloud` | `sensor_msgs/PointCloud2` | LiDAR 点云 | LiDAR |
| `/unitree_go2/pose` | `geometry_msgs/PoseStamped` | 位姿 | 定位系统 |
| `/unitree_go2/odom` | `nav_msgs/Odometry` | 里程计 | 飞控 |

### 4.2 发布的话题 (Publishers)

| 话题名 | 消息类型 | 说明 |
|--------|----------|------|
| `/occupancy_map/depth_cloud` | `sensor_msgs/PointCloud2` | 深度相机点云可视化 |
| `/occupancy_map/map_vis` | `sensor_msgs/PointCloud2` | 占据地图可视化 |
| `/occupancy_map/inflated_map_vis` | `sensor_msgs/PointCloud2` | 膨胀地图可视化 |
| `/occupancy_map/map_2d` | `nav_msgs/OccupancyGrid` | 2D 占据栅格地图 |
| `/occupancy_map/static_obstacles_vis` | `visualization_msgs/MarkerArray` | 静态障碍物可视化 |
| `/occupancy_map/raycast_vis` | `visualization_msgs/MarkerArray` | 射线投射可视化 |
| `/esdf_map/esdf_vis` | `sensor_msgs/PointCloud2` | ESDF 可视化 |

### 4.3 服务 (Services)

| 服务名 | 服务类型 | 说明 |
|--------|----------|------|
| `/occupancy_map/raycast` | `map_manager/srv/RayCast` | LiDAR 射线检测 |
| `/occupancy_map/check_pos_collision` | `map_manager/srv/CheckPosCollision` | 位置碰撞检测 |
| `/occupancy_map/get_static_obstacles` | `map_manager/srv/GetStaticObstacles` | 获取静态障碍物 |

#### RayCast.srv

```srv
# Request
geometry_msgs/Point position      # 射线起点位置
float64 start_angle               # 起始角度 (相对于目标方向)
float64 range                     # 射线最大距离 (米)
float64 vfov_min                  # 垂直视场角最小值 (度)
float64 vfov_max                  # 垂直视场角最大值 (度)
int32 vbeams                      # 垂直光束数
float64 hres                      # 水平分辨率 (度)
bool visualize                    # 是否可视化
---
# Response
float64[] points                  # 射线击中点坐标 [x1,y1,z1, x2,y2,z2, ...]
```

#### CheckPosCollision.srv

```srv
# Request
float64 x                         # 位置 X 坐标
float64 y                         # 位置 Y 坐标
float64 z                         # 位置 Z 坐标
bool inflated                     # 是否使用膨胀地图
---
# Response
bool occupied                     # 是否被占据 (true = 碰撞)
```

#### GetStaticObstacles.srv

```srv
# Request
---
# Response
geometry_msgs/Vector3[] position  # 障碍物中心位置列表
geometry_msgs/Vector3[] size      # 障碍物尺寸列表 (x, y, z)
float64[] angle                   # 障碍物旋转角度列表 (yaw)
```

### 4.4 消息同步机制

使用 ROS2 `message_filters` 实现多传感器时间同步：

```cpp
// 深度图像 + 位姿同步
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,           // 深度图像
    geometry_msgs::msg::PoseStamped    // 位姿
> depthPoseSync;

// 点云 + 位姿同步
typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::PointCloud2,     // 点云
    geometry_msgs::msg::PoseStamped    // 位姿
> pointcloudPoseSync;
```

---

## 五、配置文件参数说明

### map_param.yaml

```yaml
map_manager_node:
  ros__parameters:
    # =========================================================================
    # 传感器配置
    # =========================================================================
    localization_mode: 0              # 定位模式: 0=pose, 1=odom
    
    # 话题名称
    depth_image_topic: /unitree_go2/front_cam/depth_image
    point_cloud_topic: /unitree_go2/lidar/point_cloud
    pose_topic: /unitree_go2/pose
    odom_topic: /unitree_go2/odom

    # 机器人尺寸 (米)
    robot_size: [0.3, 0.3, 0.1]       # [长, 宽, 高]

    # =========================================================================
    # 深度相机参数
    # =========================================================================
    # 相机内参 [fx, fy, cx, cy]
    depth_intrinsics: [458.1245542807029, 458.1245542807029, 320.0, 240.0]
    
    depth_scale_factor: 1000.0        # 深度值缩放因子 (Intel RealSense = 1000)
    depth_min_value: 0.5              # 最小有效深度 (米)
    depth_max_value: 5.0              # 最大有效深度 (米)
    
    # 滤波参数
    depth_filter_margin: 2            # 图像边缘滤波宽度 (像素)
    depth_skip_pixel: 2               # 降采样步长 (每2像素取1)
    image_cols: 640                   # 图像宽度
    image_rows: 480                   # 图像高度
    
    # 相机外参 (4x4 变换矩阵, 机器人本体→深度相机)
    body_to_depth_sensor: [0.0,  0.0,  1.0,  0.4,
                          -1.0,  0.0,  0.0,  0.0,   
                           0.0, -1.0,  0.0,  0.2,
                           0.0,  0.0,  0.0,  1.0]

    # =========================================================================
    # LiDAR 点云参数
    # =========================================================================
    pointcloud_min_distance: 0.5      # 最小有效距离 (米)
    pointcloud_max_distance: 5.0      # 最大有效距离 (米)
    
    # LiDAR 外参 (4x4 变换矩阵, 机器人本体→LiDAR)
    body_to_pointcloud_sensor: [1.0,  0.0,  0.0,  0.2,
                               0.0,  1.0,  0.0,  0.0,   
                               0.0,  0.0,  1.0,  0.2,
                               0.0,  0.0,  0.0,  1.0]

    # =========================================================================
    # 射线投射参数
    # =========================================================================
    raycast_max_length: 5.0           # 最大射线长度 (米)
    
    # 概率参数
    p_hit: 0.70                       # 命中概率
    p_miss: 0.35                      # 未命中概率
    p_min: 0.12                       # 最小占据概率 (截断)
    p_max: 0.97                       # 最大占据概率 (截断)
    p_occ: 0.80                       # 占据判定阈值

    # =========================================================================
    # 地图参数
    # =========================================================================
    map_resolution: 0.1               # 地图分辨率 (米/体素)
    ground_height: -0.1               # 地面高度 (米)
    
    # 地图尺寸 (米)
    map_size: [60.0, 60.0, 3.0]       # [X, Y, Z]
    
    # 局部更新范围 (米)
    local_update_range: [5.0, 5.0, 5.0]
    local_bound_inflation: 3.0        # 局部边界膨胀 (米)
    clean_local_map: false            # 是否清理局部地图

    # =========================================================================
    # 可视化参数
    # =========================================================================
    local_map_size: [5.0, 5.0, 3.0]   # 局部地图可视化尺寸
    max_height_visualization: 4.0     # 最大可视化高度 (米)
    visualize_global_map: true        # 是否可视化全局地图
    verbose: false                    # 是否输出详细日志

    # =========================================================================
    # 预建地图
    # =========================================================================
    prebuilt_map_directory: "No"      # 预建地图路径 (PCD文件)
    # prebuilt_map_directory: "path/to/static_map.pcd"
```

---

## 六、启动与使用

### 6.1 启动占据地图节点

```bash
# 启动占据地图节点
ros2 launch map_manager occupancy_map.launch.py

# 启动 ESDF 地图节点
ros2 launch map_manager esdf_map.launch.py

# 同时启动两个节点
ros2 launch map_manager occupancy_map.launch.py &
ros2 launch map_manager esdf_map.launch.py
```

### 6.2 启动文件配置

```python
# occupancy_map.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    param_file_path = os.path.join(
        get_package_share_directory('map_manager'),
        'cfg',
        'map_param.yaml'
    )

    occupancy_map_node = Node(
        package='map_manager',
        executable='occupancy_map_node',
        name='map_manager_node',
        output='screen',
        parameters=[param_file_path]
    )

    return LaunchDescription([occupancy_map_node])
```

### 6.3 服务调用示例

```python
# Python 客户端示例 - 射线检测
import rclpy
from rclpy.node import Node
from map_manager.srv import RayCast
from geometry_msgs.msg import Point

class RayCastClient(Node):
    def __init__(self):
        super().__init__('raycast_client')
        self.client = self.create_client(RayCast, '/occupancy_map/raycast')
        
    def raycast(self, position, start_angle, range_m, vbeams=4, hres=10.0):
        request = RayCast.Request()
        request.position = Point(x=position[0], y=position[1], z=position[2])
        request.start_angle = start_angle
        request.range = range_m
        request.vfov_min = -45.0
        request.vfov_max = 45.0
        request.vbeams = vbeams
        request.hres = hres
        request.visualize = True
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        points = []
        for i in range(0, len(response.points), 3):
            points.append([
                response.points[i],
                response.points[i+1],
                response.points[i+2]
            ])
        return points

# 使用示例
client = RayCastClient()
points = client.raycast([0.0, 0.0, 1.0], 0.0, 4.0)
```

### 6.4 碰撞检测示例

```python
from map_manager.srv import CheckPosCollision

class CollisionChecker(Node):
    def __init__(self):
        super().__init__('collision_checker')
        self.client = self.create_client(CheckPosCollision, 
                                         '/occupancy_map/check_pos_collision')
        
    def check_collision(self, x, y, z, use_inflated=True):
        request = CheckPosCollision.Request()
        request.x = x
        request.y = y
        request.z = z
        request.inflated = use_inflated
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        return future.result().occupied
```

---

## 七、技术亮点

| 特点 | 说明 |
|------|------|
| **多传感器融合** | 支持深度相机和 LiDAR 点云同时建图 |
| **概率占据模型** | 基于对数几率的贝叶斯更新，处理传感器噪声 |
| **高效射线投射** | 3D DDA 算法，快速遍历射线经过的体素 |
| **局部地图更新** | 仅更新机器人周围区域，提高实时性 |
| **地图膨胀** | 考虑机器人尺寸，确保安全路径 |
| **ESDF 计算** | 欧氏距离变换，支持梯度计算和优化 |
| **静态障碍物聚类** | DBSCAN + K-means + 旋转矩形拟合 |
| **预建地图支持** | 可加载 PCD 文件作为静态环境 |

`map_manager` 是 NavRL 项目的环境感知基础模块，为导航规划提供准确的地图信息和碰撞检测服务。

---

**文档版本**: 1.0  
**最后更新**: 2025-03-23  
**维护者**: NavRL 开发团队
