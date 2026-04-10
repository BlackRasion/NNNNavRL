"""
简化版 Go2 四足机器人模型

该文件实现了一个简化版的 Go2 四足机器人模型，专门用于 PPO 强化学习训练。
与真实硬件不同，该模型使用刚体动力学模拟，直接控制速度，无需关节控制。

- 质量: 6.921 kg
- 碰撞体积: 0.3762 × 0.0935 × 0.114 m (长×宽×高)
- 惯性张量: ixx=0.02448, iyy=0.098077, izz=0.107
"""

from typing import Optional

import torch
from torchrl.data import UnboundedContinuousTensorSpec
from omni.isaac.core.simulation_context import SimulationContext
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.views import RigidPrimView

# 机器人路径根目录与默认匹配表达式
TEMPLATE_PRIM_ROOT = "/World/envs"
DEFAULT_PRIM_PATH_EXPR_TEMPLATE = "/World/envs/env_.*/{name}_.*/base_link"


class Go2Robot(RobotBase):
    """
    简化版 Go2 四足机器人模型

    该类继承自 RobotBase，实现了一个简化的刚体模型用于强化学习训练。

    核心功能：
    1. 创建简化的刚体物理模型（使用碰撞盒）
    2. 提供速度控制接口（Vx, Vy, Vyaw）
    3. 管理机器人状态（位置、姿态、速度）
    """

    cfg_cls = RobotCfg  # 配置类，用于机器人参数配置

    def __init__(self, name: str = "Go2Robot", cfg: RobotCfg = None) -> None:
        """
        初始化 Go2 机器人模型

        参数:
            name: 机器人名称，用于在仿真场景中标识
            cfg: 机器人配置对象，包含物理参数等

        - 初始化后需要调用 spawn() 和 initialize() 才能使用
        """
        super().__init__(name, cfg, is_articulation=False) # is_articulation=False 表示使用刚体模型而非关节模型

        self.mass = 6.921  # 质量参数（单位：kg）

        self.collision_size = (0.3762, 0.0935, 0.114)  # 碰撞箱体积（单位：米）

        self.inertia = (0.02448, 0.098077, 0.107)  # 惯性张量（单位：kg·m²）

        self.max_linear_vel = 2.25  # 最大线速度（单位：m/s） # 实际被ppo cfg约束

        self.max_angular_vel = 2.0  # 最大角速度（单位：rad/s） 

        self.max_linear_acc = 1.5  # 最大线加速度（单位：m/s²）

        self.max_angular_acc = 2.0  # 最大角加速度（单位：rad/s²）

        # =====================================================================
        # 动作和状态空间定义
        # =====================================================================
        # 动作空间：3维连续空间 [Vx, Vy, Vyaw]
        self.action_spec = UnboundedContinuousTensorSpec(3, device=self.device)

        # 状态空间：13维连续空间 [位置(3), 四元数(4), 线速度(3), 角速度(3)]
        self.state_spec = UnboundedContinuousTensorSpec(13, device=self.device)

        # =====================================================================
        # 状态变量初始化
        # =====================================================================
        self.vel_w = None  # 当前速度（世界坐标系）

        self.pos = None  # 当前位置（世界坐标系）

        self.rot = None  # 当前姿态（四元数，世界坐标系）

        self._dt = None  # 仿真时间步长

        self._last_cmd_vel = None # 上一帧命令速度

    def _create_prim(self, prim_path: str, translation, orientation):
        """
        创建简化的刚体物理模型
        模型结构：prim_path/base_link（包含碰撞体和视觉体）

        参数:
            prim_path: 机器人根节点的路径
            translation: 初始位置 [x, y, z]
            orientation: 初始姿态（四元数）[w, x, y, z]

        返回:
            root_prim: 创建的根节点对象
        """
        root_prim = (
            prim_utils.create_prim(  # 创建根节点（Xform 类型，用于组织层级结构）
                prim_path,
                "Xform",
                translation=translation,
                orientation=orientation,
            )
        )

        base_link_path = f"{prim_path}/base_link"  # 构建 base_link 路径

        size = self.collision_size  # 获取碰撞盒尺寸

        cuboid_cfg = sim_utils.CuboidCfg(  # 配置刚体属性
            size=list(size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 动力学设置
                kinematic_enabled=False,  # 启用动力学（非运动学模式）
                disable_gravity=True,  # 禁用重力（简化训练环境）
                enable_gyroscopic_forces=False,  # 禁用陀螺力（简化物理模型）
                # 求解器设置（影响物理仿真精度）
                solver_position_iteration_count=4,  # 位置迭代次数
                solver_velocity_iteration_count=4,  # 速度迭代次数
                # 速度限制（防止物理仿真不稳定）
                max_angular_velocity=self.max_angular_vel,  # 最大角速度
                max_linear_velocity=self.max_linear_vel,  # 最大线速度
                max_depenetration_velocity=1.0,  # 最大穿透修正速度
                # 阻尼系数（模拟空气阻力等）
                linear_damping=0.5,  # 线性阻尼
                angular_damping=0.5,  # 角阻尼
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=self.mass,  # 质量
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,  # 启用碰撞检测
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.8),  # 颜色（淡蓝色）
                metallic=0.2,  # 金属度
            ),
        )

        cuboid_cfg.func(  # 创建碰撞体
            base_link_path,
            cuboid_cfg,
            translation=(0, 0, 0),  # 相对于根节点的位置
        )

        return root_prim

    def spawn(self, translations=None, orientations=None, prim_paths=None):
        """
        在仿真场景中生成机器人
        必须在 simulation_context.reset() 之前调用。
        参数:
            translations: 初始位置列表，每个元素为 [x, y, z]
            orientations: 初始姿态列表，每个元素为四元数 [w, x, y, z]
            prim_paths: 自定义的 prim 路径列表
        返回:
            prims: 创建的 prim 对象列表
        """
        if translations is None:
            translations = [(0.0, 0.0, 0.4)]

        # 转换并验证位置参数
        translations = torch.atleast_2d(
            torch.as_tensor(translations, device=self.device)
        )
        n = translations.shape[0]

        # 设置默认姿态
        if orientations is None:
            orientations = [None for _ in range(n)]

        # 生成默认 prim 路径（与 initialize 的默认匹配表达式保持同一命名风格）
        if prim_paths is None:
            prim_paths = [f"{TEMPLATE_PRIM_ROOT}/env_{i}/{self.name}_{i}" for i in range(n)]

        # 创建机器人实例
        prims = []
        for prim_path, translation, orientation in zip(
            prim_paths, translations, orientations
        ):
            # 检查路径是否已存在
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(
                    f"Duplicate prim at {prim_path}. "
                    f"该路径已被占用，请使用不同的路径或名称。"
                )

            # 创建 prim
            prim = self._create_prim(prim_path, translation, orientation)
            prims.append(prim)

        # 更新机器人计数
        self.n += n

        return prims

    def initialize(self, prim_paths_expr: Optional[str] = None):
        """
        该方法初始化机器人的物理视图和状态变量。

        参数:
            prim_paths_expr: prim 路径表达式，用于匹配机器人实例
                           默认：None - 使用自动生成的路径表达式
        """
        if prim_paths_expr is None:  # 设置 prim 路径表达式
            prim_paths_expr = DEFAULT_PRIM_PATH_EXPR_TEMPLATE.format(name=self.name)
        self.prim_paths_expr = prim_paths_expr

        self._view = RigidPrimView(  # 创建刚体视图（用于高效的物理操作）
            self.prim_paths_expr,
            reset_xform_properties=False,
            shape=(-1, self.n), 
        )

        # 初始化视图
        self._view.initialize()
        self._view.post_reset()

        # 计算形状
        num_envs = self._view.count // self.n
        self.shape = (num_envs, self.n)

        # 保存 prim 路径
        self.prim_paths = self._view.prim_paths
        self.initialized = True

        # 获取仿真时间步长
        self._dt = SimulationContext.instance().get_physics_dt()

        # 初始化状态变量
        self.pos, self.rot = self.get_world_poses(clone=True)
        self.vel_w = torch.zeros(*self.shape, 6, device=self.device)

    def _clamp_velocity_commands(self, commands: torch.Tensor) -> torch.Tensor:
        commands = torch.nan_to_num(commands, nan=0.0, posinf=0.0, neginf=0.0) # 将NaN，Inf，-Inf替换为0
        # 限制速度命令在最大速度范围内
        limits = torch.tensor([
            self.max_linear_vel, 
            self.max_linear_vel, 
            self.max_angular_vel
        ], device=commands.device)
        commands = torch.clamp(commands, -limits, limits)
        return commands

    def _apply_rate_limit(self, commands: torch.Tensor) -> torch.Tensor:
        # 如果是首次调用或形状不匹配，直接返回当前命令
        if self._last_cmd_vel is None or self._last_cmd_vel.shape != commands.shape:
            self._last_cmd_vel = commands.clone()
            return commands

        dt = self._dt if self._dt is not None else 0.016
        max_dv = self.max_linear_acc * dt # 单步最大线速度变化
        max_dw = self.max_angular_acc * dt # 单步最大角速度变化

        max_delta = torch.tensor([max_dv, max_dv, max_dw], device=commands.device)
        
        delta = commands - self._last_cmd_vel # 计算速度变化量
        delta = torch.clamp(delta, -max_delta, max_delta) # 限制速度变化量在最大范围内

        limited = self._last_cmd_vel + delta # 计算限制后的速度命令
        self._last_cmd_vel = limited.clone() # 更新上次命令速度
        return limited

    @staticmethod
    def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
        # 从四元数中提取 yaw 角度
        # 公式：yaw = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2^2 + q3^2))
        return torch.atan2(
            2.0 * (quat[..., 0] * quat[..., 3] + quat[..., 1] * quat[..., 2]),
            1.0 - 2.0 * (quat[..., 2] ** 2 + quat[..., 3] ** 2),
        )

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        """
        应用实际速度控制命令到机器人。

        参数:
            actions: 速度命令张量，形状为 [num_envs, 3]，机器人自身坐标系下

        返回:
            applied_actions: 实际应用的速度命令 [Vx, Vy, Vyaw]，世界坐标系下
        """
        if actions.dim() != 2 or actions.shape[-1] != 3:
            raise ValueError(
                "当前仅支持动作张量形状 [num_envs, 3]，"
                f"收到 {actions.shape}"
            )

        actions = actions.unsqueeze(1)

        velocity_cmd = self._clamp_velocity_commands(actions) # 限制速度命令在最大速度范围内
        velocity_cmd = self._apply_rate_limit(velocity_cmd) # 限制速度变化量在最大范围内

        vx_body = velocity_cmd[..., 0]
        vy_body = velocity_cmd[..., 1]
        vyaw = velocity_cmd[..., 2]

        _, rot = self.get_world_poses(clone=False)
        yaw = self._quat_to_yaw(rot) # 从四元数中提取 yaw 角度
        cos_yaw = torch.cos(yaw) 
        sin_yaw = torch.sin(yaw)
        # 计算世界坐标系下的速度
        vx_world = vx_body * cos_yaw - vy_body * sin_yaw
        vy_world = vx_body * sin_yaw + vy_body * cos_yaw

        new_vel = torch.zeros(*self.shape, 6, device=self.device)
        new_vel[..., 0] = vx_world
        new_vel[..., 1] = vy_world
        new_vel[..., 2] = 0.0  # 线速度 z（固定为 0）
        new_vel[..., 3] = 0.0  # 角速度 roll（固定为 0）
        new_vel[..., 4] = 0.0  # 角速度 pitch（固定为 0）
        new_vel[..., 5] = vyaw

        self.set_velocities(new_vel)  # 应用速度
        return torch.stack([vx_body, vy_body, vyaw], dim=-1)

    def _reset_idx(self, env_ids: torch.Tensor):
        """
        在回合结束时重置机器人的速度，为新的训练回合做准备。

        参数:
            env_ids: 需要重置的 ID 
                    默认：None - 重置所有环境

        返回:
            env_ids: 实际重置的环境 ID

        注意:
            - 该方法只重置速度，不重置位置和姿态
        """
        # 如果未指定环境 ID，则重置所有环境
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)

        # 重置速度缓存
        if self.vel_w is not None:
            self.vel_w[env_ids] = 0.0
        if self._last_cmd_vel is not None:
            self._last_cmd_vel[env_ids] = 0.0

        # 设置速度为零
        zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
        self.set_velocities(zero_vel, env_indices=env_ids)

        return env_ids

    def _reset_idx_vel(self, env_ids: torch.Tensor):
        return self._reset_idx(env_ids)

    def set_world_poses(self, positions=None, orientations=None, env_indices=None):
        """
        设置机器人的世界坐标位姿

        参数:
            positions: 位置张量 [x, y, z]，形状为 [..., 3]
                      默认：None - 保持当前位置
            orientations: 姿态四元数 [w, x, y, z]，形状为 [..., 4]
                         默认：None - 保持当前姿态
            env_indices: 环境索引，指定要设置的环境
                        默认：None - 设置所有环境
        """
        return self._view.set_world_poses(
            positions, orientations, env_indices=env_indices
        )

    def set_velocities(self, velocities, env_indices=None):
        """
        直接设置机器人的线速度和角速度。

        参数:
            velocities: 速度张量 [vx, vy, vz, wx, wy, wz]，形状为 [num_envs, 6]
            env_indices: 环境索引，指定要设置的环境
                        默认：None - 设置所有环境
        注意:
            - 速度在世界坐标系中表示
            - 该方法会立即生效，不考虑动力学约束
        """
        return self._view.set_velocities(velocities, env_indices=env_indices)

    def get_state(self):
        """
        获取机器人当前状态

        该方法返回机器人的完整状态信息，包括位置、姿态和速度。

        返回:
            state: 状态张量，形状为 [..., 13]
                  - state[..., 0:3]: 位置 [x, y, z]
                  - state[..., 3:7]: 姿态四元数 [w, x, y, z]
                  - state[..., 7:10]: 线速度 [vx, vy, vz]
                  - state[..., 10:13]: 角速度 [wx, wy, wz]

        注意:
            - 位置在世界坐标系中表示
            - 速度在世界坐标系中表示
            - 四元数格式为 [w, x, y, z]
        """
        # 获取当前位置和姿态
        pos, rot = self.get_world_poses()

        # 获取当前速度
        vel_w = self.get_velocities()

        # 组合状态向量
        state = torch.cat(
            [
                pos,  # 位置 (3)
                rot,  # 姿态 (4)
                vel_w[..., :3],  # 线速度 (3)
                vel_w[..., 3:],  # 角速度 (3)
            ],
            dim=-1,
        )

        # 更新内部状态缓存（用于其他方法）
        self.pos = pos
        self.rot = rot
        self.vel_w = vel_w

        return state

    def get_velocities(self, clone: bool = False):
        """
        获取机器人的当前速度

        参数:
            clone: 是否返回克隆的副本
                  默认：False - 返回视图的引用
                  建议：需要修改速度时使用 clone=True

        返回:
            velocities: 速度张量 [vx, vy, vz, wx, wy, wz]，形状为 [..., 6]
        """
        return self._view.get_velocities(clone=clone)

    def get_world_poses(self, clone: bool = False):
        """
        获取机器人的当前世界坐标位姿

        参数:
            clone: 是否返回克隆的副本
                  默认：False - 返回视图的引用
                  建议：需要修改位姿时使用 clone=True

        返回:
            positions: 位置张量 [x, y, z]，形状为 [..., 3]
            orientations: 姿态四元数 [w, x, y, z]，形状为 [..., 4]
        """
        return self._view.get_world_poses(clone=clone) 
