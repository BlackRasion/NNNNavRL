# 简化版 Go2 机器人模型实现规范

## Why
当前项目中 `go2_robot.py` 文件为空，需要实现一个简化版的 Go2 机器人模型，用于 PPO 强化学习训练。该简化模型专注于网络训练所需的核心功能，不涉及真实硬件的关节控制和传感器模拟，以简化训练流程并加快模型收敛速度。

## What Changes
- 创建简化版 `Go2Robot` 类，实现基础体积属性定义
- 实现速度控制指令接收与处理功能
- 实现 PPO 训练所需的最小状态信息反馈接口
- 参考 URDF 文件定义机器人物理参数（质量、惯性、碰撞体积）
- 简化速度控制器，移除不必要的 robot_state 参数

## Impact
- Affected specs: 
  - PPO 训练环境 (env.py)
  - 速度控制器 (go2_velocity_controller.py)
- Affected code:
  - `isaac-training/training/scripts/go2_robot.py` - 核心机器人模型类
  - `isaac-training/training/scripts/go2_velocity_controller.py` - 速度控制器

## ADDED Requirements

### Requirement: 简化版 Go2 机器人模型类
系统 SHALL 提供 `Go2Robot` 类，实现以下简化功能：

1. **基础物理属性**：
   - 质量：6.921 kg（来自 URDF）
   - 碰撞体积：0.3762 × 0.0935 × 0.114 m（长×宽×高）
   - 惯性张量：参考 URDF 中的惯性参数

2. **速度控制接口**：
   - 接收 3 维速度命令 (Vx, Vy, Vyaw)
   - 直接应用速度到刚体，无需关节控制

3. **最小状态反馈接口**（仅返回 PPO 训练必需信息）：
   - 位置 (3D)：用于计算相对目标位置
   - 线速度 (3D)：用于观测和速度控制
   - 姿态四元数 (4D)：用于坐标变换（可选保留）
   - 角速度 (3D)：用于状态完整性（可选保留）

4. **动作空间定义**：
   - Vx：x 方向速度，范围 [-2.0, 2.0] m/s
   - Vy：y 方向速度，范围 [-2.0, 2.0] m/s
   - Vyaw：偏航角速度，范围 [-π, π] rad/s

#### Scenario: 机器人初始化
- **WHEN** 环境调用 `spawn()` 方法时
- **THEN** 机器人作为简化刚体模型在指定位置生成

#### Scenario: 速度控制执行
- **WHEN** 策略输出速度命令 [Vx, Vy, Vyaw]
- **THEN** 机器人刚体速度直接更新为指定值

#### Scenario: 状态获取
- **WHEN** 环境调用 `get_state()` 方法时
- **THEN** 返回包含位置、姿态、速度的状态张量（最多 13 维）

### Requirement: 简化速度控制器
系统 SHALL 简化 `Go2VelocityController`：

1. **移除不必要的参数**：
   - `forward()` 方法不再需要 `robot_state` 参数
   - 仅使用 `actions` 参数计算速度命令

2. **简化变换逻辑**：
   - `Go2VelController` 不再从 tensordict 获取 `robot_state`
   - 直接处理动作变换

#### Scenario: 速度控制器简化
- **WHEN** 调用 `Go2VelocityController.forward(actions)`
- **THEN** 直接返回速度命令，无需 robot_state

### Requirement: 与现有训练框架集成
系统 SHALL 确保简化模型与现有训练框架无缝集成：

1. **Go2Robot 必需方法**：
   - `__init__()`: 初始化机器人配置
   - `spawn(translations)`: 在仿真场景中生成机器人
   - `initialize()`: 初始化物理属性
   - `action_spec`: 定义 3 维动作空间
   - `get_state(env_frame)`: 获取机器人状态
   - `_reset_idx(env_ids, training)`: 重置指定环境
   - `apply_action(actions)`: 应用速度动作
   - `set_world_poses(pos, rot, env_ids)`: 设置位姿
   - `set_velocities(vels, env_ids)`: 设置速度
   - `get_velocities()`: 获取当前速度

2. **Go2Robot 必需属性**：
   - `vel_w`: 世界坐标系下的速度

#### Scenario: 环境集成
- **WHEN** `NavigationEnv` 使用 `Go2Robot` 类时
- **THEN** 所有接口调用正常工作，训练流程无报错

## MODIFIED Requirements

### Requirement: 状态观测空间
PPO 网络的 8 维观测 SHALL 在 `env.py` 中计算，而非由机器人模型直接返回：
- 相对位置 (3D)：目标位置 - 当前位置
- x 方向距离 (1D)
- y 方向距离 (1D)
- 速度 (3D)：当前线速度

## REMOVED Requirements

### Requirement: 关节控制
**Reason**: 简化模型不需要真实的关节控制，直接使用刚体速度控制
**Migration**: 移除所有关节相关代码，使用 `RigidObject` 替代 `Articulation`

### Requirement: 传感器模拟
**Reason**: LiDAR 传感器已在 `env.py` 中独立实现，机器人模型不需要内置传感器
**Migration**: 机器人模型仅提供基础刚体属性

### Requirement: 复杂物理仿真
**Reason**: 简化模型不需要腿部动力学、摩擦力等复杂物理
**Migration**: 使用简化的刚体物理模型

### Requirement: 速度控制器中的 robot_state 参数
**Reason**: 速度控制器实际上不使用 robot_state，仅进行动作归一化和缩放
**Migration**: 移除 `Go2VelocityController.forward()` 中的 `robot_state` 参数
