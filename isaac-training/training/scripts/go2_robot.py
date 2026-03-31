"""
简化版 Go2 机器人模型
用于 PPO 强化学习训练，不涉及真实硬件的关节控制和传感器模拟

物理参数参考: /isaac-training/third_party/go2_description.urdf
- 质量: 6.921 kg
- 碰撞体积: 0.3762 × 0.0935 × 0.114 m (长×宽×高)
- 惯性张量: ixx=0.02448, iyy=0.098077, izz=0.107
"""

import torch
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec
from omni.isaac.core.simulation_context import SimulationContext
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils

from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.views import RigidPrimView

TEMPLATE_PRIM_PATH = "/World/envs/env_0"


class Go2Robot(RobotBase):
    """
    简化版 Go2 四足机器人模型
    
    使用刚体模型模拟机器人，直接控制速度，无需关节控制
    """
    
    cfg_cls = RobotCfg
    
    def __init__(self, name: str = "Go2Robot", cfg: RobotCfg = None) -> None:
        super().__init__(name, cfg, is_articulation=False)
        
        self.mass = 6.921
        self.collision_size = (0.3762, 0.0935, 0.114)
        self.inertia = (0.02448, 0.098077, 0.107)
        
        self.max_linear_vel = 2.0
        self.max_angular_vel = 3.14159
        
        self.action_spec = BoundedTensorSpec(
            -1, 1, 3, device=self.device
        )
        
        self.state_spec = UnboundedContinuousTensorSpec(13, device=self.device)
        
        self.vel_w = None
        self.pos = None
        self.rot = None
        
        self._dt = None

    def _create_prim(self, prim_path, translation, orientation):
        """
        创建简化刚体模型（使用碰撞盒）
        创建结构: prim_path/base_link (包含碰撞体)
        """
        root_prim = prim_utils.create_prim(
            prim_path,
            "Xform",
            translation=translation,
            orientation=orientation,
        )
        
        base_link_path = f"{prim_path}/base_link"
        
        size = self.collision_size
        
        cuboid_cfg = sim_utils.CuboidCfg(
            size=list(size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=4,
                max_angular_velocity=100.0,
                max_linear_velocity=100.0,
                max_depenetration_velocity=10.0,
                linear_damping=0.5,
                angular_damping=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=self.mass,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.4, 0.8),
                metallic=0.2,
            ),
        )
        
        cuboid_cfg.func(
            base_link_path,
            cuboid_cfg,
            translation=(0, 0, 0),
        )
        
        return root_prim

    def spawn(
        self,
        translations=[(0.0, 0.0, 0.4)],
        orientations=None,
        prim_paths=None
    ):
        """
        在仿真场景中生成机器人
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
            raise ValueError

        prims = []
        for prim_path, translation, orientation in zip(prim_paths, translations, orientations):
            if prim_utils.is_prim_path_valid(prim_path):
                raise RuntimeError(f"Duplicate prim at {prim_path}.")
            prim = self._create_prim(prim_path, translation, orientation)
            prims.append(prim)

        self.n += n
        return prims

    def initialize(self, prim_paths_expr: str = None):
        """
        初始化机器人物理属性
        """
        if SimulationContext.instance()._physics_sim_view is None:
            raise RuntimeError(
                f"Cannot initialize {self.__class__.__name__} before the simulation context resets."
                "Call simulation_context.reset() first."
            )
        
        if prim_paths_expr is None:
            prim_paths_expr = f"/World/envs/.*/{self.name}_*/base_link"
        self.prim_paths_expr = prim_paths_expr
        
        self._view = RigidPrimView(
            self.prim_paths_expr,
            reset_xform_properties=False,
            shape=(-1, self.n),
        )
        
        self._view.initialize()
        self._view.post_reset()
        self.shape = torch.arange(self._view.count).reshape(-1, self.n).shape
        
        self.prim_paths = self._view.prim_paths
        self.initialized = True
        
        self._dt = SimulationContext.instance().get_physics_dt()
        
        self.pos, self.rot = self.get_world_poses(True)
        self.vel_w = torch.zeros(*self.shape, 6, device=self.device)

    def apply_action(self, actions: torch.Tensor) -> torch.Tensor:
        """
        应用速度控制动作
        
        参数:
            actions: 速度命令 [Vx, Vy, Vyaw]，范围 [-1, 1]
        
        返回:
            应用的速度命令
        """
        actions = actions.expand(*self.shape, 3)
        
        vx = actions[..., 0] * self.max_linear_vel
        vy = actions[..., 1] * self.max_linear_vel
        vyaw = actions[..., 2] * self.max_angular_vel
        
        current_vel = self.get_velocities(True)
        new_vel = current_vel.clone()
        new_vel[..., 0] = vx
        new_vel[..., 1] = vy
        new_vel[..., 2] = 0
        new_vel[..., 3] = 0
        new_vel[..., 4] = 0
        new_vel[..., 5] = vyaw
        
        self.set_velocities(new_vel)
        
        return torch.stack([vx, vy, vyaw], dim=-1)

    def get_state(self, check_nan: bool = False, env_frame: bool = True):
        """
        获取机器人状态
        
        返回:
            状态张量 [位置(3), 四元数(4), 线速度(3), 角速度(3)] = 13维
        """
        self.pos[:], self.rot[:] = self.get_world_poses(True)
        
        if env_frame and hasattr(self, "_envs_positions"):
            self.pos.sub_(self._envs_positions)
        
        vel_w = self.get_velocities(True)
        self.vel_w[:] = vel_w
        
        state = torch.cat([
            self.pos,
            self.rot,
            vel_w[..., :3],
            vel_w[..., 3:],
        ], dim=-1)
        
        if check_nan:
            assert not torch.isnan(state).any()
        
        return state

    def _reset_idx(self, env_ids: torch.Tensor, train: bool = True):
        """
        重置指定环境的机器人状态
        """
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
        
        self.vel_w[env_ids] = 0.0
        
        zero_vel = torch.zeros(len(env_ids), 6, device=self.device)
        self.set_velocities(zero_vel, env_indices=env_ids)
        
        return env_ids

    def set_world_poses(self, positions=None, orientations=None, env_indices=None):
        """
        设置机器人位姿
        """
        return self._view.set_world_poses(positions, orientations, env_indices=env_indices)

    def set_velocities(self, velocities, env_indices=None):
        """
        设置机器人速度
        """
        return self._view.set_velocities(velocities, env_indices=env_indices)

    def get_velocities(self, clone: bool = False):
        """
        获取当前速度
        """
        return self._view.get_velocities(clone=clone)

    def get_world_poses(self, clone: bool = False):
        """
        获取当前位姿
        """
        return self._view.get_world_poses(clone=clone)
