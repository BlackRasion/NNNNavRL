# Tasks

## 阶段 1: Go2 机器人模型类实现

- [x] Task 1: 创建 Go2Robot 机器人模型类
  - [x] SubTask 1.1: 创建 `go2_robot.py` 文件，定义 Go2Robot 类
  - [x] SubTask 1.2: 实现 `__init__` 方法，加载 UNITREE_GO2_CFG 配置
  - [x] SubTask 1.3: 实现 `initialize` 方法，初始化机器人物理属性和关节
  - [x] SubTask 1.4: 实现 `action_spec` 属性，定义 3 维速度控制动作空间
  - [x] SubTask 1.5: 实现 `get_state` 方法，返回机器人状态（位置、姿态、速度）
  - [x] SubTask 1.6: 实现 `_reset_idx` 方法，重置指定环境的机器人状态
  - [x] SubTask 1.7: 实现 `apply_action` 方法，应用速度控制动作到机器人
  - [x] SubTask 1.8: 实现 `spawn` 方法，在仿真场景中生成机器人
  - [x] SubTask 1.9: 实现 `set_world_poses` 和 `set_velocities` 方法

## 阶段 2: 配置文件修改

- [x] Task 2: 修改配置文件支持 Go2 机器人
  - [x] SubTask 2.1: 创建 `go2.yaml` 配置文件（基于 drone.yaml）
  - [x] SubTask 2.2: 修改 LiDAR 参数：lidar_vfov: [0, 20], lidar_vbeams: 3
  - [x] SubTask 2.3: 添加 Go2 机器人配置引用
  - [x] SubTask 2.4: 更新 train.yaml 的 defaults 配置引用

## 阶段 3: 训练环境核心改造

- [x] Task 3: 修改 NavigationEnv 环境类
  - [x] SubTask 3.1: 修改 `__init__` 方法，初始化 Go2 机器人参数
  - [x] SubTask 3.2: 修改 `_design_scene` 方法，使用 Go2 机器人替代无人机
  - [x] SubTask 3.3: 修改 LiDAR 配置，调整垂直视场角和射线数量
  - [x] SubTask 3.4: 修改 LiDAR 安装位置，适配 Go2 机器人底盘
  - [x] SubTask 3.5: 修改 `_set_specs` 方法，更新观测空间和动作空间定义
  - [x] SubTask 3.6: 修改 `_reset_idx` 方法，适配 Go2 机器人重置逻辑
  - [x] SubTask 3.7: 修改 `_pre_sim_step` 方法，调用 Go2 机器人的 apply_action
  - [x] SubTask 3.8: 修改 `_compute_state_and_obs` 方法，更新状态观测计算
  - [x] SubTask 3.9: 移除高度惩罚项，保留其他奖励组件
  - [x] SubTask 3.10: 修改 `_compute_reward_and_done` 方法，更新终止条件

## 阶段 4: 状态观测空间重构

- [x] Task 4: 重构状态观测空间
  - [x] SubTask 4.1: 实现自身状态观测（8 维）：相对位置(3) + x距离(1) + y距离(1) + 速度(3)
  - [x] SubTask 4.2: 验证 LiDAR 观测形状为 [1, 36, 3]
  - [x] SubTask 4.3: 验证方向观测正确计算
  - [x] SubTask 4.4: 验证动态障碍物观测形状为 [1, 5, 10]

## 阶段 5: 动作空间与控制器实现

- [x] Task 5: 实现速度控制动作空间
  - [x] SubTask 5.1: 定义 3 维动作空间（Vx, Vy, Vyaw）
  - [x] SubTask 5.2: 创建或适配速度控制器（替代 LeePositionController）
  - [x] SubTask 5.3: 实现动作归一化和反归一化逻辑

## 阶段 6: PPO 网络适配

- [x] Task 6: 修改 PPO 网络适配新观测空间
  - [x] SubTask 6.1: 更新 LiDAR 特征提取器输入维度 [batch, 1, 36, 3]
  - [x] SubTask 6.2: 验证状态特征维度为 8
  - [x] SubTask 6.3: 验证动态障碍物特征维度正确
  - [x] SubTask 6.4: 更新动作空间维度为 3

## 阶段 7: 训练脚本修改

- [x] Task 7: 修改训练和评估脚本
  - [x] SubTask 7.1: 修改 train.py，使用新的速度控制器
  - [x] SubTask 7.2: 修改 eval.py，使用新的速度控制器
  - [x] SubTask 7.3: 移除 LeePositionController 相关代码

## 阶段 8: 验证与测试

- [x] Task 8: 验证系统功能
  - [x] SubTask 8.1: 验证 Go2 机器人正确加载和初始化
  - [x] SubTask 8.2: 验证 LiDAR 传感器数据采集正确
  - [x] SubTask 8.3: 验证状态观测空间维度正确
  - [x] SubTask 8.4: 验证动作执行正确响应
  - [x] SubTask 8.5: 验证奖励计算正确（无高度惩罚）
  - [x] SubTask 8.6: 验证训练流程可以正常启动

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 1, Task 2]
- [Task 4] depends on [Task 3]
- [Task 5] depends on [Task 1]
- [Task 6] depends on [Task 4, Task 5]
- [Task 7] depends on [Task 5, Task 6]
- [Task 8] depends on [Task 3, Task 4, Task 5, Task 6, Task 7]
