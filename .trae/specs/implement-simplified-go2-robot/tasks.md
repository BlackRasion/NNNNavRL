# Tasks

## 阶段 1: 基础类结构实现

- [x] Task 1: 创建 Go2Robot 类基础结构
  - [x] SubTask 1.1: 创建 `go2_robot.py` 文件，定义 Go2Robot 类
  - [x] SubTask 1.2: 实现 `__init__` 方法，定义基础物理参数（质量、惯性、碰撞体积）
  - [x] SubTask 1.3: 定义类属性：`action_spec`（3 维速度控制空间）

## 阶段 2: 仿真场景集成

- [x] Task 2: 实现机器人生成和初始化
  - [x] SubTask 2.1: 实现 `spawn` 方法，使用 `RigidObject` 创建简化刚体模型
  - [x] SubTask 2.2: 配置碰撞体积（box: 0.3762 × 0.0935 × 0.114 m）
  - [x] SubTask 2.3: 配置质量属性（mass: 6.921 kg）
  - [x] SubTask 2.4: 实现 `initialize` 方法，初始化刚体物理属性

## 阶段 3: 状态管理接口

- [x] Task 3: 实现状态获取和设置接口
  - [x] SubTask 3.1: 实现 `get_state` 方法，返回状态张量（位置、姿态、速度）
  - [x] SubTask 3.2: 实现 `get_velocities` 方法，返回当前速度
  - [x] SubTask 3.3: 实现 `set_world_poses` 方法，设置机器人位姿
  - [x] SubTask 3.4: 实现 `set_velocities` 方法，设置机器人速度
  - [x] SubTask 3.5: 定义 `vel_w` 属性，存储世界坐标系速度

## 阶段 4: 动作控制接口

- [x] Task 4: 实现速度控制接口
  - [x] SubTask 4.1: 实现 `apply_action` 方法，接收速度命令并应用到刚体
  - [x] SubTask 4.2: 实现速度限制（Vx, Vy: [-2.0, 2.0] m/s, Vyaw: [-π, π] rad/s）
  - [x] SubTask 4.3: 实现 `_reset_idx` 方法，重置指定环境的机器人状态

## 阶段 5: 速度控制器简化

- [x] Task 5: 简化 Go2VelocityController
  - [x] SubTask 5.1: 移除 `Go2VelocityController.forward()` 中的 `robot_state` 参数
  - [x] SubTask 5.2: 简化 `Go2VelController._inv_call()` 方法，不再获取 robot_state
  - [x] SubTask 5.3: 验证简化后的速度控制器功能正常

## 阶段 6: 集成验证

- [x] Task 6: 验证与训练框架的集成
  - [x] SubTask 6.1: 验证 `NavigationEnv` 可以正确导入和使用 Go2Robot
  - [x] SubTask 6.2: 验证 `spawn` 方法正确生成机器人
  - [x] SubTask 6.3: 验证 `apply_action` 正确响应速度命令
  - [x] SubTask 6.4: 验证 `get_state` 返回正确格式的状态张量
  - [x] SubTask 6.5: 验证简化后的速度控制器正常工作
  - [x] SubTask 6.6: 验证训练流程可以正常启动

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
- [Task 4] depends on [Task 2]
- [Task 5] depends on [Task 4]
- [Task 6] depends on [Task 1, Task 2, Task 3, Task 4, Task 5]
