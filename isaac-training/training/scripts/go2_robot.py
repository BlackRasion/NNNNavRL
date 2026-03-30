"""
Go2Robot - Unitree Go2 四足机器人模型类

该文件实现了 Go2Robot 类，用于在 Isaac Sim 仿真环境中控制 Unitree Go2 四足机器人。
支持速度控制 (Vx, Vy, Vyaw)，适用于导航任务。
"""
import os
import torch
import numpy as np
from typing import Sequence, Optional, Tuple
from dataclasses import dataclass
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.robots.config import RigidBodyPropertiesCfg, ArticulationRootPropertiesCfg
from omni_drones.views import ArticulationView, RigidPrimView
from omni.isaac.core.simulation_context import SimulationContext
import omni.isaac.core.utils.prims as prim_utils
import omni_drones.utils.kit as kit_utils


TEMPLATE_PRIM_PATH = "/World/envs/env_0"


@dataclass
class Go2RobotCfg(RobotCfg):
    rigid_props: RigidBodyPropertiesCfg = RigidBodyPropertiesCfg(
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
        disable_gravity=False,
        retain_accelerations=False,
    )
    articulation_props: ArticulationRootPropertiesCfg = ArticulationRootPropertiesCfg(
        enable_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
    )


def _get_go2_usd_path():
    """获取 Go2 USD 文件的绝对路径"""
    current_file = os.path.abspath(__file__)
    training_scripts_dir = os.path.dirname(current_file)
    training_dir = os.path.dirname(training_scripts_dir)
    isaac_training_dir = os.path.dirname(training_dir)
    project_root = os.path.dirname(isaac_training_dir)
    usd_path = os.path.join(project_root, "third_party", "Go2", "go2.usd")
    return os.path.normpath(usd_path)


UNITREE_GO2_CFG = {
    "usd_path": _get_go2_usd_path(),
    "init_state": {
        "pos": (0.0, 0.0, 0.4),
        "joint_pos": {
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_vel": {".*": 0.0},
    },
    "soft_joint_pos_limit_factor": 0.9,
    "actuators": {
        "stiffness": 25.0,
        "damping": 0.5,
        "friction": 0.0,
        "effort_limit": 23.5,
        "saturation_effort": 23.5,
        "velocity_limit": 30.0,
    },
}


class Go2Robot(RobotBase):
    """
    Unitree Go2 四足机器人模型类
    
    继承自 RobotBase，实现速度控制接口。
    动作空间为 3 维速度控制 (Vx, Vy, Vyaw)。
    """
    
    cfg_cls = Go2RobotCfg
    
    def __init__(
        self,
        name: str = None,
        cfg: Go2RobotCfg = None,
        is_articulation: bool = True
    ) -> None:
        if name is None:
            name = self.__class__.__name__
        if cfg is None:
            cfg = self.cfg_cls()
        
        self._load_go2_config()
        
        super().__init__(name, cfg, is_articulation)
        
        self.action_spec = BoundedTensorSpec(
            low=torch.tensor([-2.0, -2.0, -3.14], device=self.device),
            high=torch.tensor([2.0, 2.0, 3.14], device=self.device),
            shape=(3,),
            device=self.device
        )
        
        self.state_spec = UnboundedContinuousTensorSpec(13, device=self.device)
        
        self._init_gait_params()
        
        self._last_actions = torch.zeros(3, device=self.device)
        self._joint_pos_targets = None
        self._default_joint_pos = None

    def _load_go2_config(self):
        """加载 Go2 机器人配置"""
        self.usd_path = UNITREE_GO2_CFG["usd_path"]
        self._init_state = UNITREE_GO2_CFG["init_state"]
        self._actuator_cfg = UNITREE_GO2_CFG["actuators"]
        
        self.joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.num_joints = len(self.joint_names)

    def _init_gait_params(self):
        """初始化步态参数"""
        self.gait_params = {
            "standing_height": 0.3,
            "step_height": 0.08,
            "step_length": 0.1,
            "gait_frequency": 2.0,
            "hip_offset": 0.1,
            "thigh_offset_standing": 0.8,
            "calf_offset_standing": -1.5,
        }
        
        self._default_joint_pos = torch.tensor([
            0.1, -0.1, 0.1, -0.1,
            0.8, 0.8, 1.0, 1.0,
            -1.5, -1.5, -1.5, -1.5,
        ], device=self.device)

    def spawn(
        self,
        translations=[(0.0, 0.0, 0.5)],
        orientations=None,
        prim_paths: Sequence[str] = None
    ):
        """
        在仿真场景中生成机器人
        
        Args:
            translations: 机器人初始位置列表
            orientations: 机器人初始朝向列表
            prim_paths: 机器人 prim 路径列表
            
        Returns:
            生成的 prim 列表
        """
        if SimulationContext.instance()._physics_sim_view is not None:
            raise RuntimeError(
                "Cannot spawn robots after simulation_context.reset() is called."
            )
        
        translations = torch.atleast_2d(
            torch.as_tensor(translations, device=self.device)
        )
        n = translations.shape[0]
        
        if orientations is None:
            orientations = [None for _ in range(n)]
        
        if prim_paths is None:
            prim_paths = [f"{TEMPLATE_PRIM_PATH}/{self.name}_{i}" for i in range(n)]
        
        if not len(translations) == len(prim_paths):
            raise ValueError("translations and prim_paths must have the same length")
        
        prims = []
        for prim_path, translation, orientation in zip(prim_paths, translations, orientations):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            prim = self._create_prim(prim_path, translation, orientation)
            
            kit_utils.set_nested_rigid_body_properties(
                prim_path,
                linear_damping=self.rigid_props.linear_damping,
                angular_damping=self.rigid_props.angular_damping,
                max_linear_velocity=self.rigid_props.max_linear_velocity,
                max_angular_velocity=self.rigid_props.max_angular_velocity,
                max_depenetration_velocity=self.rigid_props.max_depenetration_velocity,
                enable_gyroscopic_forces=True,
                disable_gravity=self.rigid_props.disable_gravity,
                retain_accelerations=self.rigid_props.retain_accelerations,
            )
            
            if self.is_articulation:
                kit_utils.set_articulation_properties(
                    prim_path,
                    enable_self_collisions=self.articulation_props.enable_self_collisions,
                    solver_position_iteration_count=self.articulation_props.solver_position_iteration_count,
                    solver_velocity_iteration_count=self.articulation_props.solver_velocity_iteration_count,
                )
            prims.append(prim)
        
        self.n += n
        return prims

    def _create_prim(self, prim_path, translation, orientation):
        """创建机器人 prim"""
        from pxr import UsdPhysics, PhysxSchema
        
        prim = prim_utils.create_prim(
            prim_path,
            usd_path=self.usd_path,
            translation=translation,
            orientation=orientation,
        )
        
        if self.is_articulation:
            stage = prim_utils.get_current_stage()
            prim = stage.GetPrimAtPath(prim_path)
            
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                UsdPhysics.ArticulationRootAPI.Apply(prim)
            if not prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                PhysxSchema.PhysxArticulationAPI.Apply(prim)
        
        return prim

    def initialize(
        self,
        prim_paths_expr: str = None,
        track_contact_forces: bool = False
    ):
        """
        初始化机器人物理属性
        
        Args:
            prim_paths_expr: 机器人 prim 路径表达式
            track_contact_forces: 是否跟踪接触力
        """
        if SimulationContext.instance()._physics_sim_view is None:
            raise RuntimeError(
                f"Cannot initialize {self.__class__.__name__} before the simulation context resets."
                "Call simulation_context.reset() first."
            )
        
        if prim_paths_expr is None:
            prim_paths_expr = f"/World/envs/.*/{self.name}_*"
        self.prim_paths_expr = prim_paths_expr
        
        if self.is_articulation:
            self._view = ArticulationView(
                self.prim_paths_expr,
                reset_xform_properties=False,
                shape=(-1, self.n)
            )
            self.articulation = self
        else:
            raise ValueError("Go2Robot must be an articulation")
        
        self._view.initialize()
        self._view.post_reset()
        
        self.shape = torch.arange(self._view.count).reshape(-1, self.n).shape
        self.prim_paths = self._view.prim_paths
        
        self.base_link = RigidPrimView(
            prim_paths_expr=f"{self.prim_paths_expr}/base_link",
            name="base_link",
            track_contact_forces=track_contact_forces,
            shape=self.shape,
        )
        self.base_link.initialize()
        
        self._init_joint_indices()
        self._init_state_buffers()
        
        self.initialized = True

    def _init_joint_indices(self):
        """初始化关节索引映射"""
        dof_names = self._view._dof_names
        self._joint_indices = {}
        for i, name in enumerate(dof_names):
            self._joint_indices[name] = i
        
        self.hip_indices = torch.tensor([
            self._joint_indices[f"{leg}_hip_joint"]
            for leg in ["FL", "FR", "RL", "RR"]
        ], device=self.device)
        
        self.thigh_indices = torch.tensor([
            self._joint_indices[f"{leg}_thigh_joint"]
            for leg in ["FL", "FR", "RL", "RR"]
        ], device=self.device)
        
        self.calf_indices = torch.tensor([
            self._joint_indices[f"{leg}_calf_joint"]
            for leg in ["FL", "FR", "RL", "RR"]
        ], device=self.device)

    def _init_state_buffers(self):
        """初始化状态缓冲区"""
        self._pos, self._rot = self.get_world_poses(True)
        self._vel = self.get_velocities(True)
        
        self._joint_pos_targets = self._default_joint_pos.clone().unsqueeze(0).expand(
            self.shape[0], -1
        )
        
        self._gait_phase = torch.zeros(self.shape[0], device=self.device)
        self._last_actions = torch.zeros(*self.shape, 3, device=self.device)

    def get_state(self, check_nan: bool = False, env_frame: bool = True) -> torch.Tensor:
        """
        获取机器人状态
        
        Args:
            check_nan: 是否检查 NaN 值
            env_frame: 是否转换到环境坐标系
            
        Returns:
            状态张量 [位置(3), 姿态四元数(4), 速度(3), 角速度(3)] 共 13 维
        """
        pos, rot = self.get_world_poses(True)
        vel = self.get_velocities(True)
        
        if env_frame and hasattr(self, "_envs_positions"):
            pos = pos - self._envs_positions
        
        self._pos = pos
        self._rot = rot
        self._vel = vel
        
        state = torch.cat([pos, rot, vel[..., :3], vel[..., 3:]], dim=-1)
        
        if check_nan:
            assert not torch.isnan(state).any(), "State contains NaN values"
        
        return state

    def _reset_idx(self, env_ids: torch.Tensor, train: bool = True):
        """
        重置指定环境的机器人状态
        
        Args:
            env_ids: 需要重置的环境 ID
            train: 是否为训练模式
        """
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
        
        self._last_actions[env_ids] = 0.0
        self._gait_phase[env_ids] = 0.0
        
        if self._joint_pos_targets is not None:
            self._joint_pos_targets[env_ids] = self._default_joint_pos.unsqueeze(0).expand(
                len(env_ids), -1
            )
        
        return env_ids

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        """
        应用速度控制动作到机器人
        
        将速度命令 (Vx, Vy, Vyaw) 转换为关节位置目标。
        
        Args:
            actions: 速度命令 [Vx, Vy, Vyaw]
                     Vx: 前进速度 [-2.0, 2.0] m/s
                     Vy: 横向速度 [-2.0, 2.0] m/s
                     Vyaw: 偏航角速度 [-3.14, 3.14] rad/s
        
        Returns:
            应用的动作
        """
        actions = actions.expand(*self.shape, 3)
        
        vx = actions[..., 0]
        vy = actions[..., 1]
        vyaw = actions[..., 2]
        
        self._last_actions = actions.clone()
        
        joint_targets = self._velocity_to_joint_targets(vx, vy, vyaw)
        
        self._joint_pos_targets = joint_targets
        
        self._view.set_joint_position_targets(joint_targets)
        
        return actions

    def _velocity_to_joint_targets(
        self,
        vx: torch.Tensor,
        vy: torch.Tensor,
        vyaw: torch.Tensor
    ) -> torch.Tensor:
        """
        将速度命令转换为关节位置目标
        
        使用简化的步态控制器，根据速度命令调整关节角度。
        
        Args:
            vx: 前进速度
            vy: 横向速度
            vyaw: 偏航角速度
            
        Returns:
            关节位置目标 [num_envs, 12]
        """
        batch_size = vx.shape[0]
        
        self._gait_phase = (self._gait_phase + self.dt * self.gait_params["gait_frequency"]) % (2 * np.pi)
        
        phase = self._gait_phase
        
        step_scale = torch.sqrt(vx**2 + vy**2 + 0.01).clamp(max=1.0)
        
        hip_swing = self.gait_params["hip_offset"] * step_scale * torch.sin(phase)
        thigh_swing = 0.2 * step_scale * torch.sin(phase)
        calf_swing = 0.15 * step_scale * torch.sin(phase)
        
        hip_offset = torch.zeros(batch_size, 4, device=self.device)
        thigh_offset = torch.zeros(batch_size, 4, device=self.device)
        calf_offset = torch.zeros(batch_size, 4, device=self.device)
        
        hip_offset[:, 0] = hip_swing
        hip_offset[:, 1] = -hip_swing
        hip_offset[:, 2] = hip_swing
        hip_offset[:, 3] = -hip_swing
        
        thigh_offset[:, 0] = thigh_swing
        thigh_offset[:, 1] = thigh_swing
        thigh_offset[:, 2] = -thigh_swing
        thigh_offset[:, 3] = -thigh_swing
        
        calf_offset[:, 0] = calf_swing
        calf_offset[:, 1] = calf_swing
        calf_offset[:, 2] = -calf_swing
        calf_offset[:, 3] = -calf_swing
        
        turn_hip = 0.1 * vyaw.unsqueeze(-1).expand(-1, 4)
        turn_hip[:, 0] = -turn_hip[:, 0]
        turn_hip[:, 2] = -turn_hip[:, 2]
        
        hip_targets = self._default_joint_pos[self.hip_indices] + hip_offset + turn_hip
        thigh_targets = self._default_joint_pos[self.thigh_indices] + thigh_offset
        calf_targets = self._default_joint_pos[self.calf_indices] + calf_offset
        
        joint_targets = torch.zeros(batch_size, self.num_joints, device=self.device)
        joint_targets[:, self.hip_indices] = hip_targets
        joint_targets[:, self.thigh_indices] = thigh_targets
        joint_targets[:, self.calf_indices] = calf_targets
        
        return joint_targets

    def set_world_poses(
        self,
        positions: torch.Tensor = None,
        orientations: torch.Tensor = None,
        env_indices: torch.Tensor = None
    ):
        """
        设置机器人位姿
        
        Args:
            positions: 位置张量
            orientations: 姿态四元数张量
            env_indices: 环境索引
        """
        return self._view.set_world_poses(positions, orientations, env_indices)

    def set_velocities(self, velocities: torch.Tensor, env_indices: torch.Tensor = None):
        """
        设置机器人速度
        
        Args:
            velocities: 速度张量 [线性速度(3), 角速度(3)]
            env_indices: 环境索引
        """
        return self._view.set_velocities(velocities, env_indices)

    def get_world_poses(self, clone: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取机器人位姿
        
        Args:
            clone: 是否克隆张量
            
        Returns:
            位置和姿态四元数
        """
        return self._view.get_world_poses(clone)

    def get_velocities(self, clone: bool = False) -> torch.Tensor:
        """
        获取机器人速度
        
        Args:
            clone: 是否克隆张量
            
        Returns:
            速度张量
        """
        return self._view.get_velocities(clone)

    @property
    def pos(self) -> torch.Tensor:
        """返回机器人位置"""
        pos, _ = self.get_world_poses(True)
        return pos

    @property
    def vel_w(self) -> torch.Tensor:
        """返回机器人世界坐标系下的速度"""
        return self.get_velocities(True)

    @property
    def joint_positions(self) -> torch.Tensor:
        """返回关节位置"""
        return self._view.get_joint_positions(clone=True)

    @property
    def joint_velocities(self) -> torch.Tensor:
        """返回关节速度"""
        return self._view.get_joint_velocities(clone=True)

    def set_joint_positions(self, positions: torch.Tensor, env_indices: torch.Tensor = None):
        """设置关节位置"""
        return self._view.set_joint_positions(positions, env_indices)

    def set_joint_position_targets(self, targets: torch.Tensor, env_indices: torch.Tensor = None):
        """设置关节位置目标"""
        return self._view.set_joint_position_targets(targets, env_indices)

    def get_dof_limits(self) -> torch.Tensor:
        """获取关节自由度限制"""
        return self._view.get_dof_limits()
