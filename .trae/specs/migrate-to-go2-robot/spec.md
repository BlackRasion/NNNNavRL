# Go2 机器人训练环境迁移改造规范

## Why
将现有的无人机 PPO 网络训练项目迁移改造为支持 Unitree Go2 四足机器人训练，使其能够在 Isaac Sim 2023.1.0 和 Isaac Orbit 0.15.1 环境下进行导航任务训练。

## What Changes
- 创建 Go2 机器人模型类，实现与 MultirotorBase 类似的功能接口
- 修改 LiDAR 传感器配置，调整垂直视场角和射线数量
- 重构状态观测空间，适配 Go2 机器人的导航需求
- 实现速度控制模式的动作空间（Vx, Vy, Vyaw）
- 移除高度惩罚项，适配地面机器人特性
- **BREAKING**: 原无人机训练环境将被替换为 Go2 机器人训练环境

## Impact
- Affected specs: 
  - 训练环境核心模块 (env.py)
  - 配置文件 (drone.yaml → go2.yaml)
  - PPO 网络输入维度
- Affected code:
  - `isaac-training/training/scripts/env.py` - 核心环境类
  - `isaac-training/training/cfg/drone.yaml` - 配置文件
  - `isaac-training/training/scripts/train.py` - 训练脚本
  - `isaac-training/training/scripts/eval.py` - 评估脚本
  - `isaac-training/training/scripts/ppo.py` - PPO 网络

## ADDED Requirements

### Requirement: Go2 机器人模型类
系统 SHALL 提供 Go2Robot 类，实现以下功能：
- 从 `unitree.py` 加载 UNITREE_GO2_CFG 配置
- 实现 `action_spec` 属性，定义 3 维速度控制动作空间（Vx, Vy, Vyaw）
- 实现 `get_state()` 方法，返回机器人状态（位置、姿态、速度等）
- 实现 `_reset_idx()` 方法，重置指定环境的机器人状态
- 实现 `apply_action()` 方法，应用速度控制动作
- 实现 `initialize()` 方法，初始化机器人物理属性

#### Scenario: Go2 机器人初始化
- **WHEN** 环境初始化时
- **THEN** Go2 机器人模型正确加载并初始化在指定位置

#### Scenario: Go2 机器人动作执行
- **WHEN** 策略输出速度控制动作 [Vx, Vy, Vyaw]
- **THEN** 机器人按照指定速度移动

### Requirement: LiDAR 传感器配置
系统 SHALL 配置 LiDAR 传感器满足以下规格：
- 垂直视场角范围：0° 至 +20°（总垂直视野 20°）
- 垂直方向射线数量：3
- 水平分辨率：10°（36 条水平射线）
- LiDAR 正确安装在 Go2 机器人底盘上

#### Scenario: LiDAR 数据采集
- **WHEN** 仿真步骤执行时
- **THEN** LiDAR 返回形状为 [num_envs, 1, 36, 3] 的距离数据

### Requirement: 状态观测空间
系统 SHALL 提供以下观测空间：

1. **自身状态观测（8 维）**：
   - 相对目标点位置（3 维）：目标位置 - 当前位置
   - 与目标点 x 方向距离（1 维）
   - 与目标点 y 方向距离（1 维）
   - 速度（3 维）：当前线速度

2. **LiDAR 观测**：
   - 形状：[1, 36, 3]
   - 数据类型：浮点数距离值

3. **方向观测**：
   - 目标方向向量（3 维）

4. **动态障碍物观测**：
   - 形状：[1, 5, 10]
   - 包含最近 5 个动态障碍物的状态信息

#### Scenario: 观测空间生成
- **WHEN** `_compute_state_and_obs()` 方法被调用
- **THEN** 返回符合上述规格的 TensorDict

### Requirement: 动作空间配置
系统 SHALL 提供 3 维连续动作空间：
- Vx：x 方向速度，范围 [-2.0, 2.0] m/s
- Vy：y 方向速度，范围 [-2.0, 2.0] m/s
- Vyaw：偏航角速度，范围 [-π, π] rad/s

#### Scenario: 动作空间定义
- **WHEN** `_set_specs()` 方法执行时
- **THEN** action_spec 定义为 3 维连续动作空间

### Requirement: 奖励函数设计
系统 SHALL 实现以下奖励函数组件：
- 速度奖励：朝向目标方向的速度分量
- 生存奖励：每步固定奖励
- 静态障碍物安全奖励：基于 LiDAR 距离的对数
- 动态障碍物安全奖励：基于最近障碍物距离
- 平滑性惩罚：速度变化的大小
- **移除**：高度惩罚项（地面机器人不需要）

#### Scenario: 奖励计算
- **WHEN** `_compute_state_and_obs()` 方法执行时
- **THEN** 奖励值正确计算，不包含高度惩罚

## MODIFIED Requirements

### Requirement: 配置文件结构
原 `drone.yaml` 配置文件 SHALL 被修改为支持 Go2 机器人：
- 移除无人机模型名称配置
- 添加 Go2 机器人配置引用
- 修改 LiDAR 参数：lidar_vfov: [0, 20], lidar_vbeams: 3

### Requirement: 训练环境初始化
NavigationEnv 类 SHALL 被修改：
- `_design_scene()` 方法：使用 Go2 机器人替代无人机
- `__init__()` 方法：初始化 Go2 机器人而非无人机
- `_reset_idx()` 方法：适配 Go2 机器人重置逻辑
- `_pre_sim_step()` 方法：调用 Go2 机器人的 apply_action
- `_post_sim_step()` 方法：更新 LiDAR 传感器

### Requirement: PPO 网络输入维度
PPO 特征提取器 SHALL 适配新的观测空间：
- LiDAR 输入维度：[batch, 1, 36, 3]
- 状态输入维度：8
- 动态障碍物输入维度：[batch, 1, 5, 10]

## REMOVED Requirements

### Requirement: 无人机高度控制
**Reason**: Go2 为地面机器人，不需要高度控制
**Migration**: 移除所有高度相关惩罚和奖励项

### Requirement: 无人机姿态控制
**Reason**: Go2 使用速度控制模式，不需要直接姿态控制
**Migration**: 使用速度控制器替代 LeePositionController
