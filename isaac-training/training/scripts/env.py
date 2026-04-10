"""
Go2 机器人导航强化学习环境

该文件实现了 NavigationEnv 类，是项目的核心环境模块。
基于 NVIDIA Isaac Sim 仿真平台，支持：
- 地面移动机器人物理仿真
- 构建静态和动态障碍物
- 多模态观测（LiDAR静态障碍物 + 机器人自身状态 + 动态障碍物）
- 观测空间格式定义
- 动作空间格式定义
- 奖励设计
"""

import torch
import einops
import numpy as np
import time
from typing import List, Tuple
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    UnboundedContinuousTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
)
from omni_drones.envs.isaac_env import IsaacEnv
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.orbit.terrains import (
    TerrainImporterCfg,
    TerrainImporter,
    TerrainGeneratorCfg,
    HfDiscreteObstaclesTerrainCfg,
)
from omni_drones.utils.torch import euler_to_quaternion
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
import omni.isaac.core.utils.prims as prim_utils
from go2_robot import Go2Robot
from utils import vec_to_new_frame, construct_input


class NavigationEnv(IsaacEnv):
    """
    导航环境类 - 地面移动机器人强化学习训练环境

    继承自 IsaacEnv，实现 地面移动机器人在复杂动态环境中的自主导航任务。
    该环境支持多环境并行仿真，提供多模态观测空间和丰富的奖励信号。

    核心特性：
    -----------
    1. 多模态观测：
       - LiDAR点云数据（36x3分辨率，4m探测范围）
       - 机器人内部状态（相对位置、速度等）
       - 动态障碍物状态（位置、速度、尺寸）

    2. 动态环境：   
       - 静态障碍物地形（随机生成的立方体障碍物）
       - 动态障碍物（立方体和圆柱体，自主运动）
       - 边界围墙（防止机器人绕行）

    3. 目标坐标系：
       - 所有观测转换到目标坐标系，实现旋转不变性
       - 便于策略学习，提高泛化能力

    4. 奖励设计

    仿真步骤流程：
    --------------
    1. `_pre_sim_step`: 应用动作到机器人
    2. 物理引擎推进仿真
    3. `_post_sim_step`: 更新LiDAR传感器、移动动态障碍物
    4. `_compute_state_and_obs`: 计算观测和状态
    5. `_compute_reward_and_done`: 计算奖励和终止条件

    属性：
    ------
    - go2: 地面移动机器人物体实例
    - lidar: LiDAR传感器实例
    - target_pos: 目标位置 [num_envs, 1, 3]
    - target_dir: 目标方向向量 [num_envs, 1, 3]
    - dyn_obs_list: 动态障碍物列表
    - stats: 训练统计信息（回报、回合长度等）
    """

    def __init__(self, cfg: object):
        """
        初始化导航环境
        """
        # =========================================================================
        # LiDAR 传感器参数配置
        # =========================================================================
        self.lidar_range = cfg.sensor.lidar_range   # 探测范围：LiDAR能检测到的最远距离（米）
        
        self.lidar_vfov = (      # 垂直视场角：限制在[0, 89]度范围内，避免无效角度
            max(0.0, cfg.sensor.lidar_vfov[0]),
            min(89.0, cfg.sensor.lidar_vfov[1]),
        )

        self.lidar_vbeams = cfg.sensor.lidar_vbeams # 垂直光束数：垂直方向的激光射线数量

        self.lidar_hres = cfg.sensor.lidar_hres   # 水平分辨率：相邻射线间的角度间隔（度）

        self.lidar_hbeams = int(360 / self.lidar_hres)  # 水平光束数：360度 / 水平分辨率

        # =========================================================================
        # Go2 机器人初始化
        # =========================================================================
        super().__init__(cfg, cfg.headless) # 调用父类初始化（创建仿真世界、场景等）

        self.go2.initialize()   # 初始化机器人控制器和状态

        self.init_vels = torch.zeros_like(self.go2.get_velocities())   # 初始速度张量（用于重置时设置零速度）

        # =========================================================================
        # LiDAR 传感器初始化
        # =========================================================================
        ray_caster_cfg = RayCasterCfg(  # RayCaster 配置：模拟 LiDAR 射线检测
            prim_path=f"/World/envs/env_.*/{self.go2.name}_.*/base_link",    # 传感器挂载路径：所有环境的机器人基座
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)), # 传感器相对于基座的偏移（无偏移）
            attach_yaw_only=True,   # 只跟随偏航角（不跟随俯仰和翻滚）
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres,  # 水平分辨率
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams),   # 垂直角度分布：在vfov范围内均匀分布vbeams条射线
            ),
            debug_vis=False,  # 不显示调试可视化
            mesh_prim_paths=["/World/ground"],  # 检测地面网格
        )

        # 创建 LiDAR 传感器实例
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()

        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams)  # LiDAR 分辨率：(水平光束数, 垂直光束数)

        # =========================================================================
        # 状态变量初始化
        # =========================================================================
        with torch.device(self.cfg.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)  # 目标位置：世界坐标系 
            self.target_dir = torch.zeros(self.num_envs, 1, 3)  # 目标方向向量：用于坐标变换 
            self.reward = torch.zeros(self.num_envs, 1)  # 奖励：初始化为零 
        
        # 随机选择目标掩码索引
        self._target_mask_idx = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.cfg.device
        )

    def _design_scene(self):
        """
        设计仿真训练场景

        创建完整的训练环境，包括：
        1. 地面移动机器人
        2. 光照系统
        3. 地平面
        4. 静态障碍物地形
        5. 边界围墙
        6. 动态障碍物（可选）

        返回：
        --------
        list
            机器人 prim 路径列表
        """
        # =========================================================================
        # 1. 创建 地面移动机器人
        # =========================================================================
        self.go2 = Go2Robot()
        # 在高度0.4米处生成机器人（确保机器人站立在地面上）
        self.go2.spawn(translations=[(0.0, 0.0, 0.4)])

        # =========================================================================
        # 2. 添加光照系统
        # =========================================================================
        # 主光源：模拟太阳光（方向光）
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(
                color=(0.75, 0.75, 0.75), intensity=3000.0  # 浅灰色光  # 光照强度
            ),
        )

        # 环境光：模拟天空散射光（穹顶光）
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                color=(0.2, 0.2, 0.3), intensity=2000.0  # 深蓝灰色
            ),
        )

        # 生成光源
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)

        # =========================================================================
        # 3. 创建地平面
        # =========================================================================
        cfg_ground = sim_utils.GroundPlaneCfg(
            color=(0.1, 0.1, 0.1), size=(300.0, 300.0)  # 深灰色地面  # 300m x 300m
        )
        cfg_ground.func(
            "/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01)
        )

        # =========================================================================
        # 4. 创建带静态障碍物的地形
        # =========================================================================
        self.map_range = [20.0, 20.0, 4.5]      # 地图范围 （米）

        # 地形生成器配置
        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,  # 随机种子（固定以确保可重复性）
                size=(self.map_range[0] * 2, self.map_range[1] * 2),  # 地形尺寸
                border_width=5.0,  # 边界宽度
                num_rows=1,  # 地形行数
                num_cols=1,  # 地形列数
                horizontal_scale=0.1,  # 水平缩放
                vertical_scale=0.1,  # 垂直缩放
                slope_threshold=0.75,  # 坡度阈值
                use_cache=False,  # 不使用缓存
                color_scheme="height",  # 颜色方案：按高度着色
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,  # 静态障碍物数量
                        obstacle_height_mode="range",  # 高度模式：范围随机
                        obstacle_width_range=(0.4, 1.1),  # 宽度范围（米）
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],    # 高度等级（米）
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],   # 高度概率分布
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material=None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        TerrainImporter(terrain_cfg)

        # =========================================================================
        # 5. 创建边界围墙
        # =========================================================================
        self._create_boundary_walls()

        # =========================================================================
        # 6. 创建动态障碍物（如果启用）
        # =========================================================================
        if self.cfg.env_dyn.num_obstacles == 0:
            return

        self._create_dynamic_obstacles()

    def _create_boundary_walls(self):
        """
        创建四面围墙完全包围障碍物区域，防止机器人绕行。
        """
        # 围墙参数
        wall_height = 6.0  # 围墙高度（米）
        wall_thickness = 0.4  # 围墙厚度（米）
        wall_position = self.map_range[0] + 4  # 围墙位置（距离中心） 24m
        wall_length = (wall_position + 0.1) * 2  # 围墙长度

        wall_material = sim_utils.PreviewSurfaceCfg(    # 围墙材质配置（与地面颜色一致）
            diffuse_color=(0.1, 0.1, 0.1),
            metallic=0.0,
        )

        wall_collision = sim_utils.CollisionPropertiesCfg(  # 围墙碰撞属性配置
            collision_enabled=True,
        )

        # 创建四面围墙的配置列表
        walls_config = [
            (
                "wall_north",
                [wall_length, wall_thickness, wall_height],
                (0.0, wall_position, wall_height / 2.0),
            ),
            (
                "wall_south",
                [wall_length, wall_thickness, wall_height],
                (0.0, -wall_position, wall_height / 2.0),
            ),
            (
                "wall_east",
                [wall_thickness, wall_length, wall_height],
                (wall_position, 0.0, wall_height / 2.0),
            ),
            (
                "wall_west",
                [wall_thickness, wall_length, wall_height],
                (-wall_position, 0.0, wall_height / 2.0),
            ),
        ]

        # 批量创建围墙
        for wall_name, wall_size, wall_translation in walls_config:
            wall_cfg = sim_utils.CuboidCfg(
                size=wall_size,
                visual_material=wall_material,
                collision_props=wall_collision,
            )
            wall_cfg.func(
                f"/World/{wall_name}",
                wall_cfg,
                translation=wall_translation,
            )

    def _create_dynamic_obstacles(self):
        """
        创建动态障碍物

        动态障碍物分类：
        - 3D立方体，可在空中漂浮
        - 2D圆柱体，只能地面移动

        尺寸分类：
        - 宽度分为 N_w=4 个区间
        - 高度分为 N_h=2 个区间：cuboid, cylinder
        """
        # 障碍物分类参数
        N_w = 4  # 宽度区间数
        N_h = 2  # 高度分类数：cuboid, cylinder
        max_obs_width = 1.0  # 最大宽度（米）

        self.max_obs_3d_height = 0.75  # 3D立方体高度
        self.max_obs_2d_height = 5.0  # 2D圆柱体高度

        # 宽度分辨率：每个宽度区间的宽度
        self.dyn_obs_width_res = max_obs_width / float(N_w)

        # 障碍物类别总数    
        dyn_obs_category_num = N_w * N_h

        # 每个类别的障碍物数量 
        self.dyn_obs_num_of_each_category = int(
            self.cfg.env_dyn.num_obstacles / dyn_obs_category_num
        )

        # 确保障碍物总数是类别数的整数倍
        self.cfg.env_dyn.num_obstacles = (
            self.dyn_obs_num_of_each_category * dyn_obs_category_num
        )

        # =========================================================================
        # 初始化动态障碍物状态变量
        # =========================================================================
        # 障碍物状态（N，13）
        self.dyn_obs_state = torch.zeros(
            (self.cfg.env_dyn.num_obstacles, 13),
            dtype=torch.float,
            device=self.cfg.device,
        )

        self.dyn_obs_state[:, 3] = 1.0 # 四元数 w 分量初始化为1 （无旋转）

        # 障碍物目标位置：用于随机运动
        self.dyn_obs_goal = torch.zeros(
            (self.cfg.env_dyn.num_obstacles, 3),
            dtype=torch.float,
            device=self.cfg.device,
        )

        # 障碍物原点位置：障碍物在原点附近运动
        self.dyn_obs_origin = torch.zeros(
            (self.cfg.env_dyn.num_obstacles, 3),
            dtype=torch.float,
            device=self.cfg.device,
        )

        # 障碍物速度
        self.dyn_obs_vel = torch.zeros(
            (self.cfg.env_dyn.num_obstacles, 3),
            dtype=torch.float,
            device=self.cfg.device,
        )

        # 步数计数器
        self.dyn_obs_step_count = 0

        # 障碍物尺寸（N，3[width, width, height]）
        self.dyn_obs_size = torch.zeros(
            (self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device
        )

        # =========================================================================
        # 生成障碍物位置
        # =========================================================================
        # 计算期望的障碍物间距（基于均匀分布假设）
        obs_dist = 2 * np.sqrt(
            self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles
        )
        prev_pos_list = []

        # 各类别的障碍物数量
        cuboid_category_num = cylinder_category_num = int(
            dyn_obs_category_num / N_h
        )  

        # 为每个类别生成障碍物
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # 为该类别的每个障碍物生成原点位置
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # 随机采样位置，确保障碍物之间保持一定距离
                ox, oy, oz = self._sample_obstacle_position(
                    category_idx, cuboid_category_num, prev_pos_list, obs_dist
                )

                # 保存障碍物原点和初始位置
                origin = [ox, oy, oz]
                idx = origin_idx + category_idx * self.dyn_obs_num_of_each_category
                self.dyn_obs_origin[idx] = torch.tensor(
                    origin, dtype=torch.float, device=self.cfg.device
                )
                self.dyn_obs_state[idx, :3] = torch.tensor( 
                    origin, dtype=torch.float, device=self.cfg.device
                )

                # 创建障碍物的父节点
                prim_utils.create_prim(
                    f"/World/Origin{idx}", "Xform", translation=origin
                )

            # 根据类别创建不同形状的障碍物
            self._create_obstacle_by_category(
                category_idx, cuboid_category_num, max_obs_width, N_w
            )

    def _sample_obstacle_position(self, category_idx: int, cuboid_category_num: int, prev_pos_list: List[np.ndarray], obs_dist: float) -> Tuple[float, float, float]:
        """
        采样障碍物位置，确保障碍物之间保持一定距离

        参数：
        ----------
        category_idx : int 障碍物类别索引
        cuboid_category_num : int 立方体障碍物类别数量
        prev_pos_list : List[np.ndarray] 已生成的障碍物位置列表
        obs_dist : float 期望的障碍物间距

        返回：
        ----------
        Tuple[float, float, float] 障碍物位置 (x, y, z)
        """
        start_time = time.time()
        curr_obs_dist = obs_dist  # 当前障碍物间距阈值

        while True:
            # 随机采样 x, y 坐标
            ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
            oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])

            # 根据类别确定 z 坐标
            if category_idx < cuboid_category_num:
                # 3D 立方体：z 坐标随机[1.5, map_range[2]]
                oz = np.random.uniform(low=1.5, high=self.map_range[2])
            else:
                # 2D 圆柱体：z 坐标固定在中间高度 
                oz = self.max_obs_2d_height / 2.0

            # 检查与已有障碍物的距离
            curr_pos = np.array([ox, oy])
            valid = self._check_position_validity(
                prev_pos_list, curr_pos, curr_obs_dist
            )

            # 如果超时，降低距离要求
            curr_time = time.time()
            if curr_time - start_time > 0.1:
                curr_obs_dist *= 0.8
                start_time = time.time()

            # 如果位置有效，返回
            if valid:
                prev_pos_list.append(curr_pos)
                return ox, oy, oz

    def _check_position_validity(self, prev_pos_list: List[np.ndarray], curr_pos: np.ndarray, min_dist: float) -> bool:
        """
        检查位置是否满足最小距离要求

        参数：
        ----------
        prev_pos_list : List[np.ndarray]
            已有障碍物位置列表
        curr_pos : np.ndarray
            当前候选位置
        min_dist : float
            最小距离阈值

        返回：
        ----------
        bool
            位置是否有效
        """
        for prev_pos in prev_pos_list:
            if np.linalg.norm(curr_pos - prev_pos) <= min_dist:
                return False
        return True

    def _create_obstacle_by_category(self, category_idx: int, cuboid_category_num: int, max_obs_width: float, N_w: int):
        """
        根据类别创建障碍物

        参数：
        ----------
        category_idx : int 障碍物类别索引
        cuboid_category_num : int 立方体障碍物类别数量
        max_obs_width : float 最大宽度
        N_w : int 宽度区间数
        """
        # 障碍物列表
        self.dyn_obs_list = getattr(self, "dyn_obs_list", [])

        if category_idx < cuboid_category_num:
            # 创建 3D 立方体障碍物
            obs_width = float(category_idx + 1) * max_obs_width / float(N_w)
            obs_height = self.max_obs_3d_height

            cuboid_cfg = RigidObjectCfg(
                prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                spawn=sim_utils.CuboidCfg(
                    size=[obs_width, obs_width, obs_height],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0), metallic=0.2  # 绿色
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
            dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
        else:
            # 创建 2D 圆柱体障碍物
            radius = (
                float(category_idx - cuboid_category_num + 1)
                * max_obs_width
                / float(N_w)
                / 2.0
            )
            obs_width = radius * 2
            obs_height = self.max_obs_2d_height

            cylinder_cfg = RigidObjectCfg(
                prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                spawn=sim_utils.CylinderCfg(
                    radius=radius,
                    height=obs_height,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=False
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0), metallic=0.2  # 绿色
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(),
            )
            dynamic_obstacle = RigidObject(cfg=cylinder_cfg)

        # 添加到障碍物列表
        self.dyn_obs_list.append(dynamic_obstacle)

        # 设置障碍物尺寸
        self.dyn_obs_size[
            category_idx
            * self.dyn_obs_num_of_each_category : (category_idx + 1)
            * self.dyn_obs_num_of_each_category
        ] = torch.tensor(
            [obs_width, obs_width, obs_height],
            dtype=torch.float,
            device=self.cfg.device,
        )

    def move_dynamic_obstacle(self):
        """
        移动动态障碍物

        每个障碍物在局部范围内随机移动，模拟动态环境中的移动障碍物。

        运动逻辑：
        ----------
        1. 目标采样：当障碍物接近当前目标时，在局部范围内随机采样新目标
        2. 速度更新：每约2秒随机改变速度大小和方向
        3. 位置更新：根据速度和时间步长更新位置
        4. 边界限制：确保障碍物在地图范围内
        5. 同步仿真：将更新后的状态同步到仿真器进行可视化

        参数配置：
        ----------
        - cfg.env_dyn.local_range: 局部运动范围 [x, y, z]
        - cfg.env_dyn.vel_range: 速度范围 [最小, 最大]
        - cfg.sim.dt: 仿真时间步长
        """
        # =========================================================================
        # 步骤1：为需要更新的障碍物随机采样新目标
        # =========================================================================
        # 计算当前位置与目标的距离
        if self.dyn_obs_step_count != 0:
            dyn_obs_goal_dist = torch.sqrt(
                torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal) ** 2, dim=1)
            )
        else:
            dyn_obs_goal_dist = torch.zeros(  # 初始时刻设为0
                self.dyn_obs_state.size(0), device=self.cfg.device
            )

        # 距离小于阈值（0.5米）则需要新目标
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5
        num_new_goal = int(dyn_obs_new_goal_mask.sum().item())

        # 在局部范围内随机采样新目标
        if num_new_goal > 0:
            sample_x_local = -self.cfg.env_dyn.local_range[
                0
            ] + 2.0 * self.cfg.env_dyn.local_range[0] * torch.rand(
                num_new_goal, 1, dtype=torch.float, device=self.cfg.device
            )
            sample_y_local = -self.cfg.env_dyn.local_range[
                1
            ] + 2.0 * self.cfg.env_dyn.local_range[1] * torch.rand(
                num_new_goal, 1, dtype=torch.float, device=self.cfg.device
            )
            sample_z_local = -self.cfg.env_dyn.local_range[
                2
            ] + 2.0 * self.cfg.env_dyn.local_range[2] * torch.rand(
                num_new_goal, 1, dtype=torch.float, device=self.cfg.device
            )
            sample_goal_local = torch.cat(
                [sample_x_local, sample_y_local, sample_z_local], dim=1
            )

            # 将局部目标转换到全局坐标
            self.dyn_obs_goal[dyn_obs_new_goal_mask] = (
                self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
            )

        # 限制在地图范围内
        self.dyn_obs_goal[:, 0] = torch.clamp(
            self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0]
        )
        self.dyn_obs_goal[:, 1] = torch.clamp(
            self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1]
        )
        self.dyn_obs_goal[:, 2] = torch.clamp(
            self.dyn_obs_goal[:, 2], min=0.0, max=self.map_range[2]
        )

        # 2D 圆柱体（后半部分）z坐标固定
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0) / 2) :, 2] = (
            self.max_obs_2d_height / 2.0
        )

        # =========================================================================
        # 步骤2：每约2秒随机改变速度
        # =========================================================================
        update_interval = int(2.0 / self.cfg.sim.dt)  # 更新间隔（步数） 2.0/0.016 = 128
        if self.dyn_obs_step_count % update_interval == 0:
            # 随机采样速度大小
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (
                self.cfg.env_dyn.vel_range[1] - self.cfg.env_dyn.vel_range[0]
            ) * torch.rand(
                self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device
            )

            # 计算速度方向（朝向目标）
            direction_to_goal = self.dyn_obs_goal - self.dyn_obs_state[:, :3]
            direction_normalized = direction_to_goal / torch.norm(
                direction_to_goal, dim=1, keepdim=True
            ).clamp_min(1e-6)

            # 速度 = 速度大小 × 方向
            self.dyn_obs_vel = self.dyn_obs_vel_norm * direction_normalized

        # =========================================================================
        # 步骤3：更新位置
        # =========================================================================
        # 位置 = 位置 + 速度 × 时间步长
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt

        # =========================================================================
        # 步骤4：同步到仿真器进行可视化
        # =========================================================================
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            # 获取该类别的障碍物状态
            start_idx = category_idx * self.dyn_obs_num_of_each_category
            end_idx = (category_idx + 1) * self.dyn_obs_num_of_each_category

            # 写入状态到仿真器
            dynamic_obstacle.write_root_state_to_sim(
                self.dyn_obs_state[start_idx:end_idx]
            )
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        # 增加步数计数
        self.dyn_obs_step_count += 1

    def _set_specs(self):
        """
        定义观测、动作、奖励等空间规格
        观测空间组成：
        --------------
        1. state: 地面移动机器人内部状态 (7维)
           - 相于目标位置的水平单位向量 (2维)
           - 水平距离 (1维)
           - 水平速度 (2维)
           - 目标相对角度 (1维)
           - yaw角速度 (1维)

        2. lidar: LiDAR扫描数据 (1×36×3)
           - 1个通道
           - 36个水平光束
           - 3个垂直光束

        3. direction: 目标方向向量 (3维) - 没有用于PPO输入
           - 用于坐标变换

        4. dynamic_obstacle: 动态障碍物状态 (1×N×10)
           - N个最近障碍物
           - 每个障碍物10维状态

        动作空间：
        ----------
        - 地面移动机器人动作空间 (3维速度控制)
          - x速度
          - y速度
          - yaw速度
        """
        # 观测维度定义
        observation_dim = 7  # 地面移动机器人内部状态维度
        num_dim_each_dyn_obs_state = 10  # 每个动态障碍物的状态维度

        # =========================================================================
        # 观测空间格式定义
        # =========================================================================
        self.observation_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "observation": CompositeSpec(
                            {
                                # 地面移动机器人内部状态（7维） [相对于目标位置的水平单位向量(2), 水平距离(1), 水平速度(2), 目标相对角度(1), yaw角速度(1)] = 7维
                                "state": UnboundedContinuousTensorSpec(
                                    (observation_dim,), device=self.cfg.device
                                ),
                                # LiDAR扫描数据 [1, 36, 3] (通道, 水平光束, 垂直光束)
                                "lidar": UnboundedContinuousTensorSpec(
                                    (1, self.lidar_hbeams, self.lidar_vbeams),
                                    device=self.cfg.device,
                                ),
                                # 目标方向向量（用于坐标变换）
                                "direction": UnboundedContinuousTensorSpec(
                                    (1, 3), device=self.cfg.device
                                ),
                                # 动态障碍物状态 [1, N, 10]
                                "dynamic_obstacle": UnboundedContinuousTensorSpec(
                                    (
                                        1,
                                        self.cfg.algo.feature_extractor.dyn_obs_num,
                                        num_dim_each_dyn_obs_state,
                                    ),
                                    device=self.cfg.device,
                                ),
                            }
                        ),
                    }
                ).expand(self.num_envs)
            },
            shape=[self.num_envs],
            device=self.cfg.device,
        )

        # =========================================================================
        # 动作空间格式定义
        # =========================================================================
        self.action_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "action": self.go2.action_spec, # Go2机器人动作空间 (3维速度控制)
                        }
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.cfg.device)
        )

        # =========================================================================
        # 奖励空间格式定义
        # =========================================================================
        self.reward_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {"reward": UnboundedContinuousTensorSpec((1,))}
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.cfg.device)
        )

        # =========================================================================
        # 终止条件空间定义
        # =========================================================================
        self.done_spec = (
            CompositeSpec(
                {
                    # 是否结束（终止或截断）
                    "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                    # 是否终止（碰撞）
                    "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                    # 是否截断（超时）
                    "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                }
            )
            .expand(self.num_envs)
            .to(self.cfg.device)
        )

        # =========================================================================
        # 统计信息空间定义
        # =========================================================================
        stats_spec = (
            CompositeSpec(
                {
                    "return": UnboundedContinuousTensorSpec((1,)),  # 回合累积回报
                    "episode_len": UnboundedContinuousTensorSpec(
                        (1,)
                    ),  # 回合长度（步数）
                    "reach_goal": UnboundedContinuousTensorSpec((1,)),  # 是否到达目标
                    "collision": UnboundedContinuousTensorSpec((1,)),  # 是否发生碰撞
                    "truncated": UnboundedContinuousTensorSpec((1,)),  # 是否截断
                    "reward_distance": UnboundedContinuousTensorSpec((1,)),
                    "reward_progress": UnboundedContinuousTensorSpec((1,)),
                    "reward_velocity": UnboundedContinuousTensorSpec((1,)),
                    "reward_heading": UnboundedContinuousTensorSpec((1,)),
                    "reward_safety_static": UnboundedContinuousTensorSpec((1,)),
                    "reward_safety_dynamic": UnboundedContinuousTensorSpec((1,)),
                    "reward_angular_penalty": UnboundedContinuousTensorSpec((1,)),
                    "reward_collision": UnboundedContinuousTensorSpec((1,)),
                    "reward_goal": UnboundedContinuousTensorSpec((1,)),
                }
            )
            .expand(self.num_envs)
            .to(self.cfg.device)
        )

        # =========================================================================
        # 额外信息空间定义
        # =========================================================================
        info_spec = (
            CompositeSpec(
                {
                    # 机器人完整状态 [位置(3), 四元数(4), 速度(3), 角速度(3)]
                    "robot_state_info": UnboundedContinuousTensorSpec(
                        (1, 13), device=self.cfg.device
                    ),
                }
            )
            .expand(self.num_envs)
            .to(self.cfg.device)
        )

        # 将 stats 和 info 添加到观测规格中
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec

        # 初始化统计和信息张量
        self.stats = TensorDict(
            {
                "return": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "episode_len": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "reach_goal": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "collision": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "truncated": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "reward_distance": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "reward_progress": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "reward_velocity": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "reward_heading": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "reward_safety_static": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "reward_safety_dynamic": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "reward_angular_penalty": torch.zeros(self.num_envs, 1, device=self.cfg.device),
                "reward_collision": torch.zeros(self.num_envs, 1, device=self.cfg.device), 
                "reward_goal": torch.zeros(self.num_envs, 1, device=self.cfg.device),
            },
            batch_size=[self.num_envs],
                device=self.cfg.device,
        )

        self.info = TensorDict(
            {
                "robot_state_info": torch.zeros(self.num_envs, 1, 13, device=self.cfg.device),
            },
            batch_size=[self.num_envs],
            device=self.cfg.device,
        )

    def reset_target(self, env_ids: torch.Tensor):
        """
        重置目标位置

        根据训练或评估模式设置目标位置：
        - 训练模式：随机生成目标位置
        - 评估模式：固定目标位置（所有环境在x轴一条线上，y=-22.5）

        参数：
        ----------
        env_ids : torch.Tensor
            需要重置的机器人 ID

        目标位置设计：
        --------------
        """
        if self.training:
            # 训练模式：随机选择四个方向之一
            # 目标点和起始点限制在围墙内部（±18米范围内）

            # 掩码：控制 x/y/z 哪个方向有偏移
            # [1, 0, 1]: x随机, y固定, z固定
            # [0, 1, 1]: x固定, y随机, z固定
            masks = torch.tensor(
                [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
                dtype=torch.float,
                device=self.cfg.device,
            )

            # 基础偏移位置
            shifts = torch.tensor(
                [
                    [0.0, 22.5, 0.0],
                    [0.0, -22.5, 0.0],
                    [22.5, 0.0, 0.0],
                    [-22.5, 0.0, 0.0],
                ],
                dtype=torch.float,
                device=self.cfg.device,
            )

            # 随机选择方向
            mask_indices = torch.randint(
                0, masks.size(0), (env_ids.size(0),), device=self.cfg.device
            )
            # 记录选择的方向掩码
            self._target_mask_idx[env_ids] = mask_indices.to(
                dtype=self._target_mask_idx.dtype
            )
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # 在围墙内部随机生成位置（范围：±22.5米）
            target_pos = 45.0 * torch.rand(
                env_ids.size(0), 1, 3, dtype=torch.float, device=self.cfg.device
            ) + (-22.5)

            # 高度固定为机器人质心高度（0.4米）
            target_pos[:, 0, 2] = 0.4

            # 应用掩码和偏移
            target_pos = target_pos * selected_masks + selected_shifts

            # 设置目标位置
            self.target_pos[env_ids] = target_pos
        else:
            # 评估模式：固定目标位置（所有环境在一条线上）
            # 先构建全局固定评估轨迹，再按 env_ids 取子集
            env_ids = env_ids.to(device=self.cfg.device, dtype=torch.long)
            eval_x = torch.linspace(-0.5, 0.5, self.num_envs, device=self.cfg.device) * 22.5  # x坐标：从-0.5到0.5均匀分布，缩放22.5倍
            self.target_pos[env_ids, 0, 0] = eval_x[env_ids]
            self.target_pos[env_ids, 0, 1] = -22.5  # y坐标：固定在-22.5米
            self.target_pos[env_ids, 0, 2] = 0.4    # z坐标：固定在机器人质心高度

    def _reset_idx(self, env_ids: torch.Tensor):
        """
        重置指定的 Go2 机器人状态
        在env.reset() 或 env.step()后调用

        重置内容包括：
        1. 机器人物理状态（位置、姿态、速度）
        2. 目标位置
        3. 统计信息

        参数：
        ----------
        env_ids : torch.Tensor
            需要重置的机器人 ID 张量
        """
        # 重置机器人速度
        self.go2._reset_idx_vel(env_ids, self.training)

        # 重置目标位置
        self.reset_target(env_ids)

        if self.training:
            # 训练模式：随机生成起始位置（与目标类似，限制在围墙内部）
            masks = torch.tensor(
                [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
                dtype=torch.float,
                device=self.cfg.device,
            )
            shifts = torch.tensor(
                [
                    [0.0, 22.5, 0.0],
                    [0.0, -22.5, 0.0],
                    [22.5, 0.0, 0.0],
                    [-22.5, 0.0, 0.0],
                ],
                dtype=torch.float,
                device=self.cfg.device,
            )

            # 确保起始位置和目标位置不在同一方向
            target_idx = self._target_mask_idx[env_ids]  
            rand_offset = torch.randint(1, 4, (env_ids.size(0),), device=self.cfg.device)  
            mask_indices = (target_idx + rand_offset) % 4
            
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # 随机生成起始位置（范围：±22.5米）
            pos = 45.0 * torch.rand(
                env_ids.size(0), 1, 3, dtype=torch.float, device=self.cfg.device
            ) + (-22.5)

            # 高度0.4米（地面机器人）
            pos[:, 0, 2] = 0.4

            # 应用掩码和偏移
            pos = pos * selected_masks + selected_shifts
            
        else:
            # 评估模式：固定起始位置
            pos = torch.zeros(len(env_ids), 1, 3, device=self.cfg.device)
            # x坐标：根据环境ID均匀分布
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 22.5
            # y坐标：固定在22.5米
            pos[:, 0, 1] = 22.5
            # z坐标：固定在0.4米高度
            pos[:, 0, 2] = 0.4

        # 坐标变换：计算目标方向（用于后续坐标变换）
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # 设置机器人朝向：面向目标
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.cfg.device)
        diff = self.target_pos[env_ids] - pos

        # 计算偏航角（水平面内朝向目标）
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        rpy[..., 2] = facing_yaw

        # 欧拉角转四元数
        rot = euler_to_quaternion(rpy)

        # 设置机器人位姿和速度
        self.go2.set_world_poses(pos, rot, env_ids)
        self.go2.set_velocities(self.init_vels[env_ids], env_ids)

        # 重置进度奖励的距离记录（避免新 episode 第一步使用旧值）
        if hasattr(self, "prev_distance"):
            # 计算初始距离并重置
            initial_distance = (self.target_pos[env_ids, :, :2] - pos[:, :, :2]).norm(dim=-1, keepdim=True)
            self.prev_distance[env_ids] = initial_distance

        # 重置统计信息
        self.stats[env_ids] = 0.0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        """
        仿真步前处理：应用动作到机器人
        参数：
        ----------
            tensordict[("agents", "action")]: 机器人动作 [num_envs, action_dim]
        """
        actions = tensordict[("agents", "action")]
        self.go2.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        """
        仿真步后处理：更新传感器和动态障碍物

        在物理引擎推进仿真之后调用，更新传感器状态和移动动态障碍物。
        """
        # 如果启用了动态障碍物，移动它们
        if self.cfg.env_dyn.num_obstacles != 0:
            self.move_dynamic_obstacle()

        # 更新 LiDAR 传感器状态
        self.lidar.update(self.dt)

    def _compute_reward(
        self,
        distance_2d: torch.Tensor,
        rpos: torch.Tensor,
        vel_w: torch.Tensor,
        target_angle_relative: torch.Tensor,
        angular_vel_yaw: torch.Tensor,
        collision: torch.Tensor,
        reach_goal: torch.Tensor,
        closest_dyn_obs_distance_reward: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算奖励函数（优化版本）

        奖励设计原则：
        1. 奖励塑形：提供密集的中间奖励，引导策略学习
        2. 尺度平衡：所有奖励分量在相似的数量级（±1到±10）
        3. 避免冲突：确保各奖励分量不相互矛盾

        参数：
        ----------
        distance_2d : torch.Tensor
            相对水平距离 [num_envs, 1]
        rpos : torch.Tensor
            相对位置向量 [num_envs, 3]
        vel_w : torch.Tensor
            机器人世界坐标速度 [num_envs, 3]
        target_angle_relative : torch.Tensor
            目标相对角度 [num_envs, 1]
        angular_vel_yaw : torch.Tensor
            yaw角速度 [num_envs, 1]
        collision : torch.Tensor
            碰撞状态 [num_envs, 1]
        reach_goal : torch.Tensor
            是否到达目标 [num_envs]
        closest_dyn_obs_distance_reward : torch.Tensor, optional
            动态障碍物距离 [num_envs, N]

        返回：
        ----------
        Tuple[torch.Tensor, dict]
            - reward: 总奖励 [num_envs, 1]
            - reward_dict: 各奖励分量的字典
        """
        # =========================================================================
        # 1. 目标导向奖励（核心） reward_distance
        # =========================================================================
        # a. 距离奖励
        # 奖励范围：[0.0, 2.0]，距离越近奖励越大
        b = 0.158  # 斜率常数
        reward_distance = torch.clamp(2.0 - b * distance_2d, min=0.0)

        # b. 进度奖励：距离减小的奖励（塑形奖励）
        # 奖励范围：[0.0, 13.5]
        if not hasattr(self, "prev_distance"):
            self.prev_distance = distance_2d.clone()

        distance_improved = (self.prev_distance - distance_2d).squeeze(-1)
        self.prev_distance = distance_2d.clone()

        # 奖励前进、惩罚后退
        reward_progress = torch.clamp(distance_improved, min=-0.4) * 8.0


        # =========================================================================
        # 2. 安全奖励（关键）
        # =========================================================================
        # a. 静态障碍物安全惩罚
        # self.lidar_scan 值越大表示障碍物越近
        max_lidar_value = self.lidar_scan.max(dim=-1)[0].max(dim=-1)[0]  # [num_envs]

        # 将 lidar 值转换为实际距离
        # lidar_value = lidar_range - actual_distance
        # actual_distance = lidar_range - lidar_value
        min_lidar_dist = self.lidar_range - max_lidar_value

        # 使用平滑分段函数：在 [0.5, 0.7] 区间对线性段与指数段做 smoothstep 混合，
        static_linear = -(1.5 - min_lidar_dist)
        # 让指数分支在 0.6m 与线性分支对齐: -A*exp(-0.6/0.3) = -0.9
        static_exp_scale = float(0.9 * np.exp(0.6 / 0.3))
        static_exp = -static_exp_scale * torch.exp(-min_lidar_dist / 0.3)

        static_blend_alpha = ((min_lidar_dist - 0.5) / 0.2).clamp(0.0, 1.0)
        static_blend_alpha = (
            static_blend_alpha * static_blend_alpha * (3.0 - 2.0 * static_blend_alpha)
        )
        static_mixed = (
            (1.0 - static_blend_alpha) * static_exp
            + static_blend_alpha * static_linear
        )
        # 范围 [-2.6, -0.9] [-0.9 0.0]
        safety_penalty_static = torch.where(
            min_lidar_dist > 1.5,
            torch.zeros_like(min_lidar_dist),
            torch.where(
                min_lidar_dist < 0.5,
                static_exp,
                torch.where(min_lidar_dist > 0.7, static_linear, static_mixed),
            ),
        )

        # b. 动态障碍物安全惩罚
        if self.cfg.env_dyn.num_obstacles != 0 and closest_dyn_obs_distance_reward is not None:
            # 使用与静态障碍物相同的分段函数
            min_dyn_obs_dist = closest_dyn_obs_distance_reward.min(dim=-1)[0]

            dynamic_linear = -(1.5 - min_dyn_obs_dist)
            dynamic_exp_scale = float(0.9 * np.exp(0.6 / 0.3))
            dynamic_exp = -dynamic_exp_scale * torch.exp(-min_dyn_obs_dist / 0.3)

            dynamic_blend_alpha = ((min_dyn_obs_dist - 0.5) / 0.2).clamp(0.0, 1.0)
            dynamic_blend_alpha = (
                dynamic_blend_alpha
                * dynamic_blend_alpha
                * (3.0 - 2.0 * dynamic_blend_alpha)
            )
            dynamic_mixed = (
                (1.0 - dynamic_blend_alpha) * dynamic_exp
                + dynamic_blend_alpha * dynamic_linear
            )

            safety_penalty_dynamic = torch.where(
                min_dyn_obs_dist > 1.5,
                torch.zeros_like(min_dyn_obs_dist),
                torch.where(
                    min_dyn_obs_dist < 0.5,
                    dynamic_exp,
                    torch.where(min_dyn_obs_dist > 0.7, dynamic_linear, dynamic_mixed),
                ),
            )
        else:
            # 未启用动态障碍物时，使用 LiDAR 量程作为“远离动态障碍物”的默认距离
            min_dyn_obs_dist = torch.full_like(min_lidar_dist, self.lidar_range)
            safety_penalty_dynamic = torch.zeros_like(safety_penalty_static)

        # =========================================================================
        # 3. 运动效率奖励
        # =========================================================================
        # a. 朝向目标移动的速度奖励
        # 奖励范围：[0.0, 2.25]
        # 归一化目标方向向量
        target_dir_norm = rpos[..., :2].norm(dim=-1, keepdim=True).clamp(1e-6)
        target_dir_normalized = rpos[..., :2] / target_dir_norm  # [num_envs, 1, 2]

        # 速度在目标方向的投影（世界坐标系）
        vel_toward_goal = (vel_w[..., :2] * target_dir_normalized).sum(-1).squeeze(-1)

        # 速度奖励：鼓励朝向目标移动，并在静/动态任一近障时抑制高速
        min_lidar_dist = min_lidar_dist.squeeze(-1) # [num_envs]
        min_obs_dist = torch.minimum(min_lidar_dist, min_dyn_obs_dist)
        near_obs_speed_gate = ((min_obs_dist - 0.50) / 1.0).clamp(0.1, 1.1)
        reward_velocity = torch.clamp(vel_toward_goal, min=0.0) * near_obs_speed_gate

        # b. 朝向奖励：鼓励朝向目标（与速度奖励协同）
        # 奖励范围：[0.0, 0.1]
        # 使用平滑的余弦函数，避免在目标附近震荡
        heading_cos = torch.cos(target_angle_relative.squeeze(-1))
        reward_heading = torch.clamp(heading_cos, min=0.0) * 0.1

        # =========================================================================
        # 4. 平滑性奖励
        # =========================================================================
        angular_vel_abs = torch.abs(angular_vel_yaw.squeeze(-1))
        angular_penalty = torch.where(
            angular_vel_abs > 0.7,
            - (angular_vel_abs - 0.7),  # 只惩罚超过阈值（>0.7 rad/s）的部分
            torch.zeros_like(angular_vel_abs),
        )

        # =========================================================================
        # 5. 终止奖励（稀疏奖励）
        # =========================================================================
        # a. 碰撞惩罚：大幅惩罚
        collision_penalty = collision.float().squeeze(-1) * (-150.0)

        # b. 到达目标奖励：大幅奖励
        goal_reward = reach_goal.float() * 120.0

        # =========================================================================
        # 6. 组合奖励
        # =========================================================================
        def _as_env_column(x: torch.Tensor, name: str) -> torch.Tensor:
            if x.dim() == 0:
                x = x.expand(self.num_envs).unsqueeze(-1)
            elif x.dim() == 1:
                x = x.unsqueeze(-1)
            else:
                x = x.reshape(self.num_envs, -1)
            if x.shape[1] != 1:
                raise RuntimeError(
                    f"{name} shape invalid after reshape: {x.shape}, expected {(self.num_envs, 1)}"
                )
            return x

        reward_distance_2d = _as_env_column(reward_distance, "reward_distance")
        reward_progress_2d = _as_env_column(reward_progress, "reward_progress")
        reward_velocity_2d = _as_env_column(reward_velocity, "reward_velocity")
        reward_heading_2d = _as_env_column(reward_heading, "reward_heading")
        safety_penalty_static_2d = _as_env_column(
            safety_penalty_static, "safety_penalty_static"
        )
        safety_penalty_dynamic_2d = _as_env_column(
            safety_penalty_dynamic, "safety_penalty_dynamic"
        )
        angular_penalty_2d = _as_env_column(angular_penalty, "angular_penalty")
        collision_penalty_2d = _as_env_column(collision_penalty, "collision_penalty")
        goal_reward_2d = _as_env_column(goal_reward, "goal_reward")

        # 组合奖励（固定权重）
        reward = (
            reward_distance_2d * 0.5  # 距离奖励
            + reward_progress_2d * 2.5  # 进度奖励
            + reward_velocity_2d * 2.0  # 速度奖励
            + reward_heading_2d * 0.5  # 朝向奖励
            + safety_penalty_static_2d * 3.0  # 安全惩罚
            + safety_penalty_dynamic_2d * 3.0  # 安全惩罚
            + angular_penalty_2d * 1.0 # 平滑性惩罚
            + collision_penalty_2d  * 1.0 # 碰撞惩罚
            + goal_reward_2d * 1.0  # 目标奖励
        )

        # 确保奖励的形状是 (num_envs, 1)
        assert reward.shape == (
            self.num_envs,
            1,
        ), f"Reward shape mismatch: {reward.shape} vs {(self.num_envs, 1)}"

        # 返回奖励和各分量
        reward_dict = {
            "reward_distance": reward_distance_2d,
            "reward_progress": reward_progress_2d,
            "reward_velocity": reward_velocity_2d,
            "reward_heading": reward_heading_2d,
            "reward_safety_static": safety_penalty_static_2d,
            "reward_safety_dynamic": safety_penalty_dynamic_2d,
            "reward_angular_penalty": angular_penalty_2d,
            "reward_collision": collision_penalty_2d,
            "reward_goal": goal_reward_2d,
        }

        return reward, reward_dict

    def _compute_state_and_obs(self):
        """
        计算观测和状态（核心函数）

        这是环境的核心函数，负责：
        1. 获取机器人状态
        2. 计算 LiDAR 观测
        3. 计算机器人内部状态观测
        4. 计算动态障碍物观测
        5. 计算奖励
        6. 判断终止条件
        7. 更新统计信息

        返回：
        ----------
        TensorDict
            包含以下内容的张量字典：
            - agents/observation: 策略网络输入
            - stats: 训练统计信息
            - info: 额外信息（用于控制器）

        观测空间详解：
        --------------
        1. LiDAR观测：
           - 格式：[num_envs, 1, 36, 3]
           - 值：探测距离（值越大表示障碍物越近）

        2. 机器人状态：
           - 格式：[num_envs, 7]
           - 内容：[相对于目标位置的水平单位向量(2), 水平距离(1), 水平速度(2), 目标相对角(1), yaw角速度(1)]
           - 坐标系：目标坐标系（旋转不变）

        3. 动态障碍物状态：
           - 格式：[num_envs, 1, N, 10]
           - 内容：[相对机器人的单位向量(3), 水平距离(1), z轴差(1), 速度(3), 宽度类别(1), 高度类别(1)]，前四项为目标坐标系下
           - N：最近的N个障碍物
        """
        # =========================================================================
        # 步骤1：获取机器人根状态
        # =========================================================================
        # 根状态：[世界位置(3), 姿态四元数(4), 世界速度(3), 角速度(3), ...]
        self.root_state = self.go2.get_state(env_frame=False)

        # 保存前13维用于额外信息（供控制器使用）
        self.info["robot_state_info"][:] = self.root_state[..., :13]

        # =========================================================================
        # 步骤2：计算 LiDAR 观测
        # =========================================================================
        # LiDAR扫描数据处理：
        # - lidar.data.ray_hits_w: 射线击中点世界坐标 [num_envs, num_rays, 3]
        # - lidar.data.pos_w: LiDAR传感器世界坐标 [num_envs, 3]
        # 计算每个射线击中点的距离
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1))
            .norm(dim=-1)  # 计算距离
            .clamp_max(self.lidar_range)  # 限制最大距离
            .reshape(
                self.num_envs, 1, *self.lidar_resolution
            )  # reshape为[num_envs, 1, 36, 3]
        )
        self.lidar_scan = torch.nan_to_num(
            self.lidar_scan, nan=0.0, posinf=0.0, neginf=0.0
        ) # 结果：值越大表示障碍物越近（探测范围 - 实际距离）

        # =========================================================================
        # 步骤3：计算机器人内部状态观测
        # =========================================================================
        # a. 距离信息
        rpos = self.target_pos - self.root_state[..., :3]  # 相对位置向量

        # 水平距离计算
        rpos_2d = rpos[..., :2]
        distance_2d = rpos_2d.norm(dim=-1, keepdim=True)

        # b. 目标方向（用于坐标变换）
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0  # 只考虑水平方向

        # c. 单位方向向量（世界坐标 -> 目标坐标系）
        rpos_g = vec_to_new_frame(rpos, target_dir_2d)
        rpos_g_2d = rpos_g[..., :2]
        rpos_clipped_g_2d = rpos_g_2d / rpos_g_2d.norm(dim=-1, keepdim=True).clamp(1e-6)

        # d. 速度（世界坐标 -> 目标坐标系）
        vel_w = self.root_state[..., 7:10]  # 世界坐标系下的速度
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)  # 转换到目标坐标系
        # 优化：只保留水平速度（2维）
        vel_g_2d = vel_g[..., :2]

        # e. 计算目标相对角度（目标方向 - 机器人朝向）
        # 计算机器人朝向（yaw角）
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        robot_quat = self.root_state[..., 3:7]  # [w, x, y, z]
        robot_yaw = torch.atan2(
            2.0
            * (
                robot_quat[..., 0] * robot_quat[..., 3]
                + robot_quat[..., 1] * robot_quat[..., 2]
            ),
            1.0 - 2.0 * (robot_quat[..., 2] ** 2 + robot_quat[..., 3] ** 2),
        )

        target_angle_world = torch.atan2(rpos_2d[..., 1], rpos_2d[..., 0])  # 目标方向角度（水平面内）
        
        target_angle_relative = target_angle_world - robot_yaw  # 目标相对角度（在机器人坐标系中）
        
        target_angle_relative = torch.atan2(    # 归一化到 [-pi, pi]
            torch.sin(target_angle_relative), torch.cos(target_angle_relative)
        )

        # f. yaw角速度
        angular_vel_yaw = self.root_state[..., 12]  # 只取yaw角速度 (wz)

        # 组合机器人状态（优化：7维）
        # [相对位置(2), 水平距离(1), 速度(2), 目标角度(1), yaw角速度(1)] = 7维
        robot_state = torch.cat(
            [
                rpos_clipped_g_2d,  # 相于目标位置的水平单位向量 (2)
                distance_2d,  # 水平距离 (1)
                vel_g_2d,  # 水平速度 (2)
                target_angle_relative.unsqueeze(-1),  # 目标相对角度 (1)
                angular_vel_yaw.unsqueeze(-1),  # yaw角速度 (1) 
            ],
            dim=-1,
        ).squeeze(1)
        robot_state = torch.nan_to_num(robot_state, nan=0.0, posinf=0.0, neginf=0.0)

        # =========================================================================
        # 步骤4：计算动态障碍物观测（如果启用）
        # =========================================================================
        if self.cfg.env_dyn.num_obstacles != 0:
            # 扩展障碍物位置维度以便广播计算
            dyn_obs_pos_expanded = (
                self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            )

            # 计算相对位置 （num_envs, N, 3）（世界坐标系）
            dyn_obs_rpos_expanded = (
                dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3]
            )
            # 2D圆柱体相对高度设为0（只考虑水平距离）
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0) / 2) :, 2] = 0.0

            # 计算相对距离 [num_envs, N]
            dyn_obs_distance = torch.norm(dyn_obs_rpos_expanded, dim=2)
            
            # 选择最近的N个动态障碍物
            _, closest_dyn_obs_idx = torch.topk(
                dyn_obs_distance,
                self.cfg.algo.feature_extractor.dyn_obs_num,
                dim=1,
                largest=False,
            )

            # 超出范围障碍物的标记 [num_envs, N] (布尔值)
            dyn_obs_range_mask = (
                dyn_obs_distance.gather(1, closest_dyn_obs_idx) > self.lidar_range
            )

            # 最近N个障碍物的相对位置 [num_envs, N, 3] (世界坐标系)
            closest_dyn_obs_rpos = torch.gather(
                dyn_obs_rpos_expanded,
                1,
                closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3),
            )

            # 最近N个障碍物的相对位置 [num_envs, N, 3] (目标坐标系)
            closest_dyn_obs_rpos_g = vec_to_new_frame(
                closest_dyn_obs_rpos, target_dir_2d
            )
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0.0  # 超出范围的置零

            # 计算动态障碍位置相关
            closest_dyn_obs_distance = closest_dyn_obs_rpos_g.norm(dim=-1, keepdim=True) # 3D相对距离（目标坐标系）
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm( dim=-1, keepdim=True) # 2D相对距离（目标坐标系）
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1) # z轴差（目标坐标系）
            closest_dyn_obs_rpos_gn = ( # 单位向量（目标坐标系）
                closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)
            )

            # b. 障碍物速度（目标坐标系）
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.0
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d)

            # c. 障碍物大小类别
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx]
            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)

            # 宽度类别：[1, 2, 3, 4]
            closest_dyn_obs_width_category = (closest_dyn_obs_width / self.dyn_obs_width_res)
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.0 # 对超出有效范围的障碍物宽度类别置零

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            # 高度类别：1表示2D圆柱体，2表示3D立方体
            closest_dyn_obs_height_category = torch.where(
                closest_dyn_obs_height > self.max_obs_3d_height,
                torch.tensor(1.0, device=self.cfg.device),  # 2D圆柱体
                torch.tensor(2.0, device=self.cfg.device),  # 3D立方体
            )
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.0 # 对超出有效范围的障碍物高度类别置零

            # 组合动态障碍物状态 [num_envs, 1, N, 10]
            dyn_obs_states = torch.cat(
                [
                    closest_dyn_obs_rpos_gn,  # 单位向量（目标坐标系） (3)
                    closest_dyn_obs_distance_2d,  # 2D距离（目标坐标系） (1)
                    closest_dyn_obs_distance_z,  # z轴差（目标坐标系） (1)
                    closest_dyn_obs_vel_g,  # 速度（目标坐标系） (3)
                    closest_dyn_obs_width_category,  # 宽度类别 (1)
                    closest_dyn_obs_height_category,  # 高度类别 (1)
                ],
                dim=-1,
            ).unsqueeze(1)
            dyn_obs_states = torch.nan_to_num(
                dyn_obs_states, nan=0.0, posinf=0.0, neginf=0.0
            )

            # d. 碰撞检测（世界坐标系）
            closest_dyn_obs_distance_2d_collision = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collision[dyn_obs_range_mask] = float("inf") # 对超出有效范围的障碍物2D距离置无穷大

            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].abs().unsqueeze(-1)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float("inf") # 对超出有效范围的障碍物垂直高度差置无穷大

            dynamic_collision_2d = closest_dyn_obs_distance_2d_collision <= ( # 2D碰撞检测
                closest_dyn_obs_width / 2.0 + 0.25
            )
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= ( # Z碰撞检测
                closest_dyn_obs_height / 2.0 + 0.15
            )
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z # 2D和Z方向同时满足才认为碰撞

            dynamic_collision = torch.any( # 检查是否存在任意一个障碍物导致碰撞
                dynamic_collision_each, dim=1
            )  # 形状: (num_envs, 1)

            # 用于奖励计算的动态障碍物距离（目标坐标系）
            closest_dyn_obs_distance_reward = (
                closest_dyn_obs_distance.squeeze(-1) - closest_dyn_obs_size[..., 0] / 2.0
            )
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = ( # 超出感知范围的障碍物，将其距离设为激光雷达最大探测距离
                self.cfg.sensor.lidar_range
            )

        else: # 未启用动态障碍物
            dyn_obs_states = torch.zeros( # 返回全零张量
                self.num_envs,
                1,
                self.cfg.algo.feature_extractor.dyn_obs_num,
                10,
                device=self.cfg.device,
            )
            dynamic_collision = torch.zeros( # 返回零碰撞
                self.num_envs, 1, dtype=torch.bool, device=self.cfg.device
            )

        # =========================================================================
        # 步骤5：组合最终观测
        # =========================================================================
        obs = {
            "state": robot_state,
            "lidar": self.lidar_scan,
            "direction": torch.nan_to_num(
                target_dir_2d, nan=0.0, posinf=0.0, neginf=0.0
            ),
            "dynamic_obstacle": dyn_obs_states,
        }

        # =========================================================================
        # 步骤6：碰撞检测和到达目标检测
        # =========================================================================
        # a. 静态障碍物碰撞检测
        # LiDAR读数接近最大值表示有障碍物非常近
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") > (
            self.lidar_range - 0.25
        )  # 0.25米碰撞半径

        # b. 组合碰撞检测结果
        # dynamic_collision 已在动态障碍物处理部分定义
        if dynamic_collision.dim() == 1:
            dynamic_collision = dynamic_collision.unsqueeze(
                -1
            )  # (num_envs,) -> (num_envs, 1)

        collision = static_collision | dynamic_collision

        # c. 到达目标检测
        # 只考虑水平距离（与高度无关）
        reach_goal = (distance_2d.squeeze(-1) < 0.5).squeeze(-1)  # 形状: (num_envs,)

        # =========================================================================
        # 步骤7：计算奖励
        # =========================================================================
        # 调用奖励计算函数
        if self.cfg.env_dyn.num_obstacles != 0:
            self.reward, reward_dict = self._compute_reward(
                distance_2d=distance_2d,
                rpos=rpos,
                vel_w=vel_w,
                target_angle_relative=target_angle_relative,
                angular_vel_yaw=angular_vel_yaw,
                collision=collision,
                reach_goal=reach_goal,
                closest_dyn_obs_distance_reward=closest_dyn_obs_distance_reward,

            )
        else:
            self.reward, reward_dict = self._compute_reward(
                distance_2d=distance_2d,
                rpos=rpos,
                vel_w=vel_w,
                target_angle_relative=target_angle_relative,
                angular_vel_yaw=angular_vel_yaw,
                collision=collision,
                reach_goal=reach_goal,
            )

        # 提取各奖励分量用于统计
        reward_distance_2d = reward_dict["reward_distance"]
        reward_progress_2d = reward_dict["reward_progress"]
        reward_vel_2d = reward_dict["reward_velocity"]
        reward_heading = reward_dict["reward_heading"]
        reward_safety_static_2d = reward_dict["reward_safety_static"]
        reward_safety_dynamic = reward_dict["reward_safety_dynamic"]
        angular_vel_penalty_2d = reward_dict["reward_angular_penalty"]
        collision_penalty = reward_dict["reward_collision"]
        reward_goal = reward_dict["reward_goal"]



        # =========================================================================
        # 步骤8：判断终止条件
        # =========================================================================
        # 终止条件：碰撞或到达目标
        self.terminated = collision | reach_goal.unsqueeze(-1)

        # 截断条件：达到最大回合长度
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # =========================================================================
        # 步骤9：更新统计信息
        # =========================================================================
        self.stats["return"] += self.reward  # 累积回报
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)  # 回合长度
        self.stats["reach_goal"] = reach_goal.float()  # 是否到达目标
        self.stats["collision"] = collision.float()  # 是否碰撞
        self.stats["truncated"] = self.truncated.float()  # 是否截断
        self.stats["reward_distance"] = reward_distance_2d
        self.stats["reward_progress"] = reward_progress_2d
        self.stats["reward_velocity"] = reward_vel_2d
        self.stats["reward_heading"] = reward_heading
        self.stats["reward_safety_static"] = reward_safety_static_2d
        self.stats["reward_safety_dynamic"] = reward_safety_dynamic
        self.stats["reward_angular_penalty"] = angular_vel_penalty_2d
        self.stats["reward_collision"] = collision_penalty
        self.stats["reward_goal"] = reward_goal


        # 返回观测张量字典
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": obs,
                    },
                    [self.num_envs],
                ),
                "stats": self.stats.clone(),
                "info": self.info,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        """
        计算奖励和终止条件

        将在 `_compute_state_and_obs` 中计算好的奖励和终止条件打包返回。
        这种设计允许在计算观测时同时计算奖励，避免重复计算。

        返回：
        ----------
        TensorDict
            包含以下内容的张量字典：
            - agents/reward: 奖励值 [num_envs, 1]
            - done: 是否结束 [num_envs, 1]
            - terminated: 是否终止（碰撞） [num_envs, 1]
            - truncated: 是否截断（超时） [num_envs, 1]
        """
        # 返回奖励和终止条件
        return TensorDict(
            {
                "agents": {"reward": self.reward},
                "done": self.terminated | self.truncated,  # 任一条件满足即结束
                "terminated": self.terminated,  # 终止（碰撞）
                "truncated": self.truncated,  # 截断（超时）
            },
            self.batch_size,
        )

