"""
Go2Robot - Unitree Go2 四足机器人模型类

该文件实现了 Go2Robot 类，用于在 Isaac Sim 仿真环境中控制 Unitree Go2 四足机器人。
支持速度控制 (Vx, Vy, Vyaw)，适用于导航任务。

基于 UNITREE_GO2_CFG 官方配置开发。
"""

import torch
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.articulation import ArticulationCfg
from torchrl.data import UnboundedContinuousTensorSpec
from omni.isaac.orbit_assets.unitree import UNITREE_GO2_CFG


class Go2Robot:
    """
    Go2 四足机器人控制器
    
    封装 Isaac Sim Articulation 类，提供速度控制接口用于强化学习训练。
    
    主要功能:
    - 机器人实例生成和初始化
    - 状态获取（位置、姿态、速度）
    - 速度控制命令执行
    - 环境重置
    """

    def __init__(self):
        """
        初始化 Go2Robot 实例
        
        设置内部状态变量，但不创建仿真实例。
        实际的机器人实例在 spawn() 方法中创建。
        """
        self._articulation: Articulation = None
        self._is_initialized = False
        self._prim_paths = []
        
        self._vel_w: torch.Tensor = None
        self._cmd_velocities: torch.Tensor = None
        self._num_envs: int = 0
        self._device: str = "cuda:0"

    @property
    def action_spec(self) -> UnboundedContinuousTensorSpec:
        """
        动作空间规格
        
        返回:
            UnboundedContinuousTensorSpec: 形状为 (3,) 的动作空间规格
            对应 (Vx, Vy, Vyaw) 速度控制命令
        """
        return UnboundedContinuousTensorSpec((3,), device=self._device)

    @property
    def vel_w(self) -> torch.Tensor:
        """
        世界坐标系下的速度
        
        返回:
            torch.Tensor: 形状为 [num_envs, 6] 的速度张量
            包含线速度 (3) 和角速度 (3)
        """
        if self._articulation is None or not self._is_initialized:
            return self._vel_w
        return self._articulation.data.root_state_w[:, 7:13]

    @property
    def num_envs(self) -> int:
        """环境数量"""
        return self._num_envs

    @property
    def device(self) -> str:
        """计算设备"""
        return self._device

    def spawn(self, translations: list = None) -> list:
        """
        生成机器人实例
        
        基于 UNITREE_GO2_CFG 配置在仿真环境中创建 Go2 机器人实例。
        
        参数:
            translations: 机器人初始位置列表，每个元素为 (x, y, z) 元组
            
        返回:
            list: 机器人 prim 路径列表
        """
        if translations is None:
            translations = [(0.0, 0.0, 0.4)]
        
        self._num_envs = len(translations)
        self._prim_paths = []
        
        for i, translation in enumerate(translations):
            prim_path = f"/World/envs/env_{i}/Go2Robot_0"
            
            UNITREE_GO2_CFG.spawn.func(
                prim_path,
                UNITREE_GO2_CFG.spawn,
                translation=translation,
            )
            self._prim_paths.append(prim_path)
        
        return self._prim_paths

    def initialize(self):
        """
        初始化机器人
        
        在仿真环境启动后调用，完成机器人的内部状态设置。
        创建 Articulation 实例并初始化内部缓冲区。
        """
        if self._is_initialized:
            return
        
        prim_path_expr = "/World/envs/env_.*/Go2Robot_0"
        
        cfg = ArticulationCfg(
            prim_path=prim_path_expr,
            spawn=UNITREE_GO2_CFG.spawn,
            init_state=UNITREE_GO2_CFG.init_state,
            actuators=UNITREE_GO2_CFG.actuators,
            soft_joint_pos_limit_factor=UNITREE_GO2_CFG.soft_joint_pos_limit_factor,
        )
        
        self._articulation = Articulation(cfg=cfg)
        
        sim = sim_utils.SimulationContext.instance()
        if sim is not None:
            self._device = sim.device
        
        self._is_initialized = True
        self._num_envs = self._articulation.num_instances
        
        self._vel_w = torch.zeros(self._num_envs, 6, device=self._device)
        self._cmd_velocities = torch.zeros(self._num_envs, 3, device=self._device)
        
        print(f"[Go2Robot]: 初始化完成，环境数量: {self._num_envs}")

    def get_state(self, env_frame: bool = False) -> torch.Tensor:
        """
        获取机器人完整状态
        
        参数:
            env_frame: 是否返回环境坐标系下的状态（暂未实现）
            
        返回:
            torch.Tensor: 形状为 [num_envs, state_dim] 的状态张量
            包含位置 (3)、姿态四元数 (4)、线速度 (3)、角速度 (3) 等
        """
        if self._articulation is None or not self._is_initialized:
            return torch.zeros(self._num_envs, 13, device=self._device)
        
        return self._articulation.data.root_state_w

    def get_velocities(self) -> torch.Tensor:
        """
        获取机器人速度
        
        返回:
            torch.Tensor: 形状为 [num_envs, 6] 的速度张量
            包含线速度 (3) 和角速度 (3)
        """
        if self._articulation is None or not self._is_initialized:
            return self._vel_w
        return self._articulation.data.root_state_w[:, 7:13]

    def apply_action(self, actions: torch.Tensor):
        """
        应用速度控制命令
        
        将策略网络输出的速度命令应用到机器人。
        速度命令在机器人本体坐标系下，需要转换到世界坐标系。
        
        参数:
            actions: 速度命令张量，形状为 [num_envs, 3]
                     对应 (Vx, Vy, Vyaw) - 前进速度、横向速度、角速度
        """
        if self._articulation is None or not self._is_initialized:
            return
        
        self._cmd_velocities = actions.clone()
        
        quat_w = self._articulation.data.root_quat_w
        
        lin_vel_b = torch.zeros(self._num_envs, 3, device=self._device)
        lin_vel_b[:, 0] = actions[:, 0]  # Vx
        lin_vel_b[:, 1] = actions[:, 1]  # Vy
        
        lin_vel_w = math_utils.quat_apply(quat_w, lin_vel_b)
        
        ang_vel_w = torch.zeros(self._num_envs, 3, device=self._device)
        ang_vel_w[:, 2] = actions[:, 2]  # Vyaw (绕 z 轴角速度)
        
        root_velocity = torch.cat([lin_vel_w, ang_vel_w], dim=-1)
        
        self._articulation.write_root_velocity_to_sim(root_velocity)

    def set_world_poses(
        self,
        positions: torch.Tensor,
        orientations: torch.Tensor,
        env_ids: torch.Tensor = None
    ):
        """
        设置机器人位姿
        
        参数:
            positions: 位置张量，形状为 [num_envs, 1, 3] 或 [num_envs, 3]
            orientations: 姿态四元数张量，形状为 [num_envs, 1, 4] 或 [num_envs, 4]
            env_ids: 环境索引，如果为 None 则设置所有环境
        """
        if self._articulation is None or not self._is_initialized:
            return
        
        if positions.dim() == 3:
            positions = positions.squeeze(1)
        if orientations.dim() == 3:
            orientations = orientations.squeeze(1)
        
        root_poses = torch.cat([positions, orientations], dim=-1)
        
        if env_ids is not None:
            self._articulation.write_root_pose_to_sim(root_poses, env_ids=env_ids.tolist())
        else:
            self._articulation.write_root_pose_to_sim(root_poses)

    def set_velocities(
        self,
        velocities: torch.Tensor,
        env_ids: torch.Tensor = None
    ):
        """
        设置机器人速度
        
        参数:
            velocities: 速度张量，形状为 [num_envs, 6]
                        包含线速度 (3) 和角速度 (3)
            env_ids: 环境索引，如果为 None 则设置所有环境
        """
        if self._articulation is None or not self._is_initialized:
            return
        
        if env_ids is not None:
            self._articulation.write_root_velocity_to_sim(velocities, env_ids=env_ids.tolist())
        else:
            self._articulation.write_root_velocity_to_sim(velocities)

    def _reset_idx(self, env_ids: torch.Tensor, training: bool = True):
        """
        重置指定环境的机器人状态
        
        参数:
            env_ids: 需要重置的环境索引
            training: 是否为训练模式
        """
        if self._articulation is None or not self._is_initialized:
            return
        
        if env_ids is None or len(env_ids) == 0:
            return
        
        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids
        self._articulation.reset(env_ids_list)
        
        if self._vel_w is not None:
            self._vel_w[env_ids_list] = 0.0
        if self._cmd_velocities is not None:
            self._cmd_velocities[env_ids_list] = 0.0
