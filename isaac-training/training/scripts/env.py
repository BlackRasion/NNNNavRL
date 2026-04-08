"""
Navigation Environment for Drone Navigation with Reinforcement Learning
无人机导航强化学习环境

该文件实现了 NavigationEnv 类，是 NavRL 项目的核心环境模块。
基于 NVIDIA Isaac Sim 仿真平台，支持：
- 无人机物理仿真
- LiDAR 传感器模拟
- 静态和动态障碍物
- 多模态观测（LiDAR + 内部状态 + 动态障碍物）
- 目标坐标系下的旋转不变性
"""
import torch
import einops
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
import omni.isaac.orbit.sim as sim_utils
from omni_drones.robots.drone import MultirotorBase
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg
from omni_drones.utils.torch import euler_to_quaternion, quat_axis
from omni.isaac.orbit.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.core.utils.viewports import set_camera_view
from utils import vec_to_new_frame, vec_to_world, construct_input
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.assets import RigidObject, RigidObjectCfg
import time

class NavigationEnv(IsaacEnv):
    """
    导航环境类
    
    继承自 IsaacEnv，实现无人机在动态环境中的导航任务
    
    一个完整的仿真步骤流程：
    1. _pre_sim_step: 应用动作 -> 推进仿真
    2. _post_sim_step: 更新 LiDAR、移动动态障碍物
    3. 增加进度计数器
    4. _compute_state_and_obs: 计算观测和状态
    5. _compute_reward_and_done: 计算奖励和终止条件
    """

    def __init__(self, cfg):
        print("[Navigation Environment]: 环境初始化中...")
        
        # =========================================================================
        # LiDAR 参数配置
        # =========================================================================
        self.lidar_range = cfg.sensor.lidar_range                    # 探测范围 (4m)
        self.lidar_vfov = (max(-89., cfg.sensor.lidar_vfov[0]), min(89., cfg.sensor.lidar_vfov[1]))  # 垂直视场角
        self.lidar_vbeams = cfg.sensor.lidar_vbeams                  # 垂直光束数 (4)
        self.lidar_hres = cfg.sensor.lidar_hres                      # 水平分辨率 (10°)
        self.lidar_hbeams = int(360/self.lidar_hres)                # 水平光束数 (36)

        super().__init__(cfg, cfg.headless)
        
        # =========================================================================
        # 无人机初始化
        # =========================================================================        
        self.drone.initialize()
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        # =========================================================================
        # LiDAR 传感器初始化
        # =========================================================================
        # RayCaster 用于模拟 LiDAR 射线检测
        ray_caster_cfg = RayCasterCfg(
            prim_path="/World/envs/env_.*/Hummingbird_0/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.BpearlPatternCfg(
                horizontal_res=self.lidar_hres, # 水平分辨率
                vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams) # 垂直角度分布
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.lidar = RayCaster(ray_caster_cfg)
        self.lidar._initialize_impl()
        self.lidar_resolution = (self.lidar_hbeams, self.lidar_vbeams) 
        
        # =========================================================================
        # 状态变量初始化
        # =========================================================================
        with torch.device(self.device):
            self.target_pos = torch.zeros(self.num_envs, 1, 3)      # 目标位置 [num_envs, 1, 3]
            self.target_dir = torch.zeros(self.num_envs, 1, 3)      # 目标方向（用于坐标变换）
            self.height_range = torch.zeros(self.num_envs, 1, 2)    # 高度范围 [min, max]
            self.prev_drone_vel_w = torch.zeros(self.num_envs, 1, 3)  # 上一时刻速度


    def _design_scene(self):
        """
        设计仿真场景
        
        创建无人机、光照、地面、静态障碍物地形和动态障碍物
        
        返回:
            list: 无人机 prim 路径列表
        """
        # 1. 创建无人机 /World/envs/envs_0
        drone_model = MultirotorBase.REGISTRY[self.cfg.drone.model_name] 
        cfg = drone_model.cfg_cls(force_sensor=False)
        self.drone = drone_model(cfg=cfg)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0] # 在高度2米处生成.

        # 2. 添加光照
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(color=(0.2, 0.2, 0.3), intensity=2000.0),
        )
        light.spawn.func(light.prim_path, light.spawn, light.init_state.pos)
        sky_light.spawn.func(sky_light.prim_path, sky_light.spawn)
        
        # 3. 创建地平面
        cfg_ground = sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1), size=(300., 300.))
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground, translation=(0, 0, 0.01))

        # 4. 创建带静态障碍物的地形
        self.map_range = [20.0, 20.0, 4.5]  # 地图范围 [x, y, z]

        terrain_cfg = TerrainImporterCfg(
            num_envs=self.num_envs,
            env_spacing=0.0,
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=TerrainGeneratorCfg(
                seed=0,
                size=(self.map_range[0]*2, self.map_range[1]*2), 
                border_width=5.0,
                num_rows=1, 
                num_cols=1, 
                horizontal_scale=0.1,
                vertical_scale=0.1,
                slope_threshold=0.75,
                use_cache=False,
                color_scheme="height",
                sub_terrains={
                    "obstacles": HfDiscreteObstaclesTerrainCfg(
                        horizontal_scale=0.1,
                        vertical_scale=0.1,
                        border_width=0.0,
                        num_obstacles=self.cfg.env.num_obstacles,   # 静态障碍物数量
                        obstacle_height_mode="range",
                        obstacle_width_range=(0.4, 1.1),    # 宽度范围
                        obstacle_height_range=[1.0, 1.5, 2.0, 4.0, 6.0],    # 高度等级
                        obstacle_height_probability=[0.1, 0.15, 0.20, 0.55],     # 高度概率分布
                        platform_width=0.0,
                    ),
                },
            ),
            visual_material = None,
            max_init_terrain_level=None,
            collision_group=-1,
            debug_vis=True,
        )
        terrain_importer = TerrainImporter(terrain_cfg)

        # 5. 创建动态障碍物 （如果启用）
        if (self.cfg.env_dyn.num_obstacles == 0):
            return
        # 动态障碍物分类：
        # - 3D 障碍物：立方体，可在空中漂浮
        # - 2D 障碍物：圆柱体，只能水平移动
        # 宽度分为 N_w=4 个区间: [0, 0.25], [0.25, 0.50], [0.50, 0.75], [0.75, 1.0]
        # 高度分为 N_h=2 个区间: [0, 0.5], [0.5, inf]（区分3D和2D障碍物）
        N_w = 4  # 宽度区间数
        N_h = 2  # 高度区间数
        max_obs_width = 1.0
        self.max_obs_3d_height = 1.0
        self.max_obs_2d_height = 5.0
        self.dyn_obs_width_res = max_obs_width/float(N_w)
        dyn_obs_category_num = N_w * N_h
        self.dyn_obs_num_of_each_category = int(self.cfg.env_dyn.num_obstacles / dyn_obs_category_num)
        self.cfg.env_dyn.num_obstacles = self.dyn_obs_num_of_each_category * dyn_obs_category_num # 确保障碍物数量是整数


        # 动态障碍物状态变量
        self.dyn_obs_list = []
        self.dyn_obs_state = torch.zeros((self.cfg.env_dyn.num_obstacles, 13), dtype=torch.float, device=self.cfg.device) 
        self.dyn_obs_state[:, 3] = 1. # 四元数 w 分量初始化为1
        self.dyn_obs_goal = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_origin = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_vel = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.cfg.device)
        self.dyn_obs_step_count = 0 # 步数计数
        self.dyn_obs_size = torch.zeros((self.cfg.env_dyn.num_obstacles, 3), dtype=torch.float, device=self.device) 

        # 辅助函数：检查位置是否满足均匀分布条件
        def check_pos_validity(prev_pos_list, curr_pos, adjusted_obs_dist):
            for prev_pos in prev_pos_list:
                if (np.linalg.norm(curr_pos - prev_pos) <= adjusted_obs_dist):
                    return False
            return True            
        # 计算期望的障碍物间距
        obs_dist = 2 * np.sqrt(self.map_range[0] * self.map_range[1] / self.cfg.env_dyn.num_obstacles) 
        curr_obs_dist = obs_dist  # 初始化当前障碍物间距
        prev_pos_list = [] 
        cuboid_category_num = cylinder_category_num = int(dyn_obs_category_num/N_h) # 各4类

        # 为每个类别创建障碍物
        for category_idx in range(cuboid_category_num + cylinder_category_num):
            # 为该类别的每个障碍物生成原点位置
            for origin_idx in range(self.dyn_obs_num_of_each_category):
                # 随机采样位置
                start_time = time.time()
                while (True):
                    ox = np.random.uniform(low=-self.map_range[0], high=self.map_range[0])
                    oy = np.random.uniform(low=-self.map_range[1], high=self.map_range[1])
                    if (category_idx < cuboid_category_num):
                        oz = np.random.uniform(low=0.0, high=self.map_range[2]) 
                    else:
                        oz = self.max_obs_2d_height/2.
                    curr_pos = np.array([ox, oy])
                    valid = check_pos_validity(prev_pos_list, curr_pos, curr_obs_dist)
                    curr_time = time.time()
                    if (curr_time - start_time > 0.1):
                        curr_obs_dist *= 0.8
                        start_time = time.time()
                    if (valid):
                        prev_pos_list.append(curr_pos)
                        break
                curr_obs_dist = obs_dist
                origin = [ox, oy, oz]
                self.dyn_obs_origin[origin_idx+category_idx*self.dyn_obs_num_of_each_category] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)     
                self.dyn_obs_state[origin_idx+category_idx*self.dyn_obs_num_of_each_category, :3] = torch.tensor(origin, dtype=torch.float, device=self.cfg.device)                        
                prim_utils.create_prim(f"/World/Origin{origin_idx+category_idx*self.dyn_obs_num_of_each_category}", "Xform", translation=origin)

            # 根据类别生成不同形状的障碍物 
            if (category_idx < cuboid_category_num):
                # 3D 立方体障碍物
                obs_width = width = float(category_idx+1) * max_obs_width/float(N_w)
                obs_height = self.max_obs_3d_height
                cuboid_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cuboid",
                    spawn=sim_utils.CuboidCfg(
                        size=[width, width, self.max_obs_3d_height],
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cuboid_cfg)
            else:
                radius = float(category_idx-cuboid_category_num+1) * max_obs_width/float(N_w) / 2.
                obs_width = radius * 2
                obs_height = self.max_obs_2d_height
                # 2D 圆柱体障碍物
                cylinder_cfg = RigidObjectCfg(
                    prim_path=f"/World/Origin{construct_input(category_idx*self.dyn_obs_num_of_each_category, (category_idx+1)*self.dyn_obs_num_of_each_category)}/Cylinder",
                    spawn=sim_utils.CylinderCfg(
                        radius = radius,
                        height = self.max_obs_2d_height, 
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                dynamic_obstacle = RigidObject(cfg=cylinder_cfg)
            self.dyn_obs_list.append(dynamic_obstacle)
            self.dyn_obs_size[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category] \
                = torch.tensor([obs_width, obs_width, obs_height], dtype=torch.float, device=self.cfg.device)

    def move_dynamic_obstacle(self):
        """
        移动动态障碍物
        
        每个障碍物在局部范围内随机移动，模拟动态环境
        运动逻辑：
        1. 当接近当前目标时，随机采样新目标
        2. 每约2秒随机改变速度
        3. 更新位置并同步到仿真器
        """
        # 步骤1：为需要更新的障碍物随机采样新目标
        # 计算当前位置与目标的距离
        dyn_obs_goal_dist = torch.sqrt(torch.sum((self.dyn_obs_state[:, :3] - self.dyn_obs_goal)**2, dim=1)) if self.dyn_obs_step_count !=0 \
            else torch.zeros(self.dyn_obs_state.size(0), device=self.cfg.device)
        # 距离小于阈值则需要新目标
        dyn_obs_new_goal_mask = dyn_obs_goal_dist < 0.5 
        num_new_goal = torch.sum(dyn_obs_new_goal_mask)
        # 在局部范围内随机采样新目标
        sample_x_local = -self.cfg.env_dyn.local_range[0] + 2. * self.cfg.env_dyn.local_range[0] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_y_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[1] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_z_local = -self.cfg.env_dyn.local_range[1] + 2. * self.cfg.env_dyn.local_range[2] * torch.rand(num_new_goal, 1, dtype=torch.float, device=self.cfg.device)
        sample_goal_local = torch.cat([sample_x_local, sample_y_local, sample_z_local], dim=1)
    
        # 将局部目标转换到全局坐标
        self.dyn_obs_goal[dyn_obs_new_goal_mask] = self.dyn_obs_origin[dyn_obs_new_goal_mask] + sample_goal_local
        # 限制在地图范围内
        self.dyn_obs_goal[:, 0] = torch.clamp(self.dyn_obs_goal[:, 0], min=-self.map_range[0], max=self.map_range[0])
        self.dyn_obs_goal[:, 1] = torch.clamp(self.dyn_obs_goal[:, 1], min=-self.map_range[1], max=self.map_range[1])
        self.dyn_obs_goal[:, 2] = torch.clamp(self.dyn_obs_goal[:, 2], min=0., max=self.map_range[2])
        self.dyn_obs_goal[int(self.dyn_obs_goal.size(0)/2):, 2] = self.max_obs_2d_height/2. # for 2d obstacles

        # 步骤2：每约2秒随机改变速度
        if (self.dyn_obs_step_count % int(2.0/self.cfg.sim.dt) == 0):
            self.dyn_obs_vel_norm = self.cfg.env_dyn.vel_range[0] + (self.cfg.env_dyn.vel_range[1] \
              - self.cfg.env_dyn.vel_range[0]) * torch.rand(self.dyn_obs_vel.size(0), 1, dtype=torch.float, device=self.cfg.device)
            self.dyn_obs_vel = self.dyn_obs_vel_norm * \
                (self.dyn_obs_goal - self.dyn_obs_state[:, :3])/torch.norm((self.dyn_obs_goal - self.dyn_obs_state[:, :3]), dim=1, keepdim=True)

        # 步骤3：更新位置
        self.dyn_obs_state[:, :3] += self.dyn_obs_vel * self.cfg.sim.dt

        # 步骤4：同步到仿真器进行可视化
        for category_idx, dynamic_obstacle in enumerate(self.dyn_obs_list):
            dynamic_obstacle.write_root_state_to_sim(self.dyn_obs_state[category_idx*self.dyn_obs_num_of_each_category:(category_idx+1)*self.dyn_obs_num_of_each_category]) 
            dynamic_obstacle.write_data_to_sim()
            dynamic_obstacle.update(self.cfg.sim.dt)

        self.dyn_obs_step_count += 1

    def _set_specs(self):
        """
        定义观测、动作、奖励等空间规格
        
        TorchRL 使用这些规格来验证数据形状和类型
        """
        observation_dim = 8 # 无人机内部状态维度
        num_dim_each_dyn_obs_state = 10 # 每个动态障碍物的状态维度

        # =========================================================================
        # 观测空间格式定义
        # =========================================================================
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    # 无人机内部状态: [相对位置(3), 水平距离(1), 垂直距离(1), 速度(3)] = 8
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device),
                    # LiDAR 扫描数据: [1, 36, 4] (通道, 水平, 垂直) 
                    "lidar": UnboundedContinuousTensorSpec((1, self.lidar_hbeams, self.lidar_vbeams), device=self.device),
                    # 目标方向向量（用于坐标变换）
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    # 动态障碍物状态: [1, N, 10] (N=最近障碍物数)
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), device=self.device),
                }),
            }).expand(self.num_envs)
        }, shape=[self.num_envs], device=self.device)
        
        # =========================================================================
        # 动作空间格式定义
        # =========================================================================
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec, # 无人机动作空间
            })
        }).expand(self.num_envs).to(self.device)
        
        # =========================================================================
        # 奖励空间格式定义
        # =========================================================================
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1,))
            })
        }).expand(self.num_envs).to(self.device)


        # =========================================================================
        # 终止条件空间定义
        # =========================================================================
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),  # 是否结束
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),    # 是否终止（碰撞/到达/越界）
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool), # 是否截断（超时）
        }).expand(self.num_envs).to(self.device) 

        # =========================================================================
        # 统计信息空间定义
        # =========================================================================
        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),        # 回合回报
            "episode_len": UnboundedContinuousTensorSpec(1),   # 回合长度
            "reach_goal": UnboundedContinuousTensorSpec(1),    # 是否到达目标
            "collision": UnboundedContinuousTensorSpec(1),     # 是否碰撞
            "truncated": UnboundedContinuousTensorSpec(1),     # 是否截断
        }).expand(self.num_envs).to(self.device)

        # =========================================================================
        # 额外信息空间定义
        # =========================================================================
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
        }).expand(self.num_envs).to(self.device)
        # 将 stats 和 info 添加到观测规格中
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        # 初始化统计和信息
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    
    def reset_target(self, env_ids: torch.Tensor):
        """
        重置目标位置
        
        训练模式：随机生成目标位置（四个方向之一）
        评估模式：固定目标位置（用于对比）
        
        参数:
            env_ids: 需要重置的环境 ID
        """
        if (self.training):
            # 训练模式：随机选择四个方向之一
            # masks: 控制 x/y 哪个方向有偏移
            # [1,0,1]: x方向随机, y固定, z随机
            # [0,1,1]: x固定, y方向随机, z随机
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            # shifts: 基础偏移位置
            # [0, 24, 0]: 前方24米
            # [0, -24, 0]: 后方24米
            # [24, 0, 0]: 右方24米
            # [-24, 0, 0]: 左方24米
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            # 随机选择方向
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)
            # 在 [-24, 24] 范围内随机生成位置
            target_pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)    # 高度在 [0.5, 2.5] 米
            target_pos[:, 0, 2] = heights
            target_pos = target_pos * selected_masks + selected_shifts
            # 设置目标位置
            self.target_pos[env_ids] = target_pos
        else:  
            # 评估模式：固定目标位置（所有环境在一条线上）
            # 先构建全局固定评估轨迹，再按 env_ids 取子集
            env_ids = env_ids.to(device=self.cfg.device, dtype=torch.long)
            eval_x = torch.linspace(-0.5, 0.5, self.num_envs, device=self.cfg.device) * 32  # x坐标：从-0.5到0.5均匀分布，缩放22.5倍
            self.target_pos[env_ids, 0, 0] = eval_x[env_ids]
            self.target_pos[env_ids, 0, 1] = -24.  # y坐标：固定在-24米
            self.target_pos[env_ids, 0, 2] = 2.

    def _reset_idx(self, env_ids: torch.Tensor):
        """
        重置指定的无人机状态
        
        参数:
            env_ids: 需要重置的无人机 ID
        """
        self.drone._reset_idx(env_ids, self.training)    # 重置无人机物理状态
        self.reset_target(env_ids)  # 重置目标位置
        if (self.training):
            # 训练模式：随机生成起始位置（与目标类似）
            masks = torch.tensor([[1., 0., 1.], [1., 0., 1.], [0., 1., 1.], [0., 1., 1.]], dtype=torch.float, device=self.device)
            shifts = torch.tensor([[0., 24., 0.], [0., -24., 0.], [24., 0., 0.], [-24., 0., 0.]], dtype=torch.float, device=self.device)
            mask_indices = np.random.randint(0, masks.size(0), size=env_ids.size(0))
            selected_masks = masks[mask_indices].unsqueeze(1)
            selected_shifts = shifts[mask_indices].unsqueeze(1)

            # 随机生成起始位置
            pos = 48. * torch.rand(env_ids.size(0), 1, 3, dtype=torch.float, device=self.device) + (-24.)
            heights = 0.5 + torch.rand(env_ids.size(0), dtype=torch.float, device=self.device) * (2.5 - 0.5)
            pos[:, 0, 2] = heights# height
            pos = pos * selected_masks + selected_shifts
        else:
            # 评估模式：固定起始位置
            pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
            pos[:, 0, 0] = (env_ids / self.num_envs - 0.5) * 32.
            pos[:, 0, 1] = 24.
            pos[:, 0, 2] = 2.
        
        # 坐标变换：计算目标方向（用于后续坐标变换）
        self.target_dir[env_ids] = self.target_pos[env_ids] - pos

        # 设置无人机朝向：面向目标
        rpy = torch.zeros(len(env_ids), 1, 3, device=self.device)
        diff = self.target_pos[env_ids] - pos
        facing_yaw = torch.atan2(diff[..., 1], diff[..., 0])    # 计算偏航角（水平面内朝向目标）
        rpy[..., 2] = facing_yaw
        # 欧拉角转四元数
        rot = euler_to_quaternion(rpy)
        # 设置无人机位姿和速度
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        # 重置速度记录
        self.prev_drone_vel_w[env_ids] = 0.
        # 计算高度范围（起点和目标高度的最小/最大值）
        self.height_range[env_ids, 0, 0] = torch.min(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        self.height_range[env_ids, 0, 1] = torch.max(pos[:, 0, 2], self.target_pos[env_ids, 0, 2])
        # 重置统计信息
        self.stats[env_ids] = 0.  
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        """
        仿真步前处理：应用动作
        """
        actions = tensordict[("agents", "action")] 
        self.drone.apply_action(actions) 

    def _post_sim_step(self, tensordict: TensorDictBase):
        """
        仿真步后处理：更新传感器和动态障碍物
        """
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.move_dynamic_obstacle()
        self.lidar.update(self.dt)
    
    def _compute_state_and_obs(self):
        """
        计算观测和状态（核心函数）
        
        返回包含以下内容的 TensorDict:
        - agents/observation: 策略网络输入
        - stats: 训练统计信息
        - info: 额外信息（用于控制器）
        """
        # =========================================================================
        # 获取机器人根状态
        # =========================================================================
        self.root_state = self.drone.get_state(env_frame=False) # [世界位置(3), 姿态四元数(4), 世界速度(3), 角速度(3), ...]
        self.info["drone_state"][:] = self.root_state[..., :13] # 保存前 13 维用于额外信息

        # =========================================================================
        # 观测 I: LiDAR 扫描数据
        # =========================================================================
        # 计算每个射线击中点的距离
        self.lidar_scan = self.lidar_range - (
            (self.lidar.data.ray_hits_w - self.lidar.data.pos_w.unsqueeze(1)) # lidar.data.ray_hits_w: 射线击中点世界坐标 [num_envs, num_rays, 3] lidar.data.pos_w: LiDAR 传感器世界坐标 [num_envs, 3]
            .norm(dim=-1)   
            .clamp_max(self.lidar_range)       # 限制最大距离并 reshape
            .reshape(self.num_envs, 1, *self.lidar_resolution)
        ) 
        # 结果: [num_envs, 1, 36, 4]，值越大表示障碍物越近

        # =========================================================================
        # 观测 II: 无人机内部状态
        # =========================================================================
        # a. 距离信息
        rpos = self.target_pos - self.root_state[..., :3]       # 相对位置向量  
        distance = rpos.norm(dim=-1, keepdim=True)              # 总距离
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)  # 水平距离
        distance_z = rpos[..., 2].unsqueeze(-1)                 # 垂直距离
        
        # b. 目标方向（用于坐标变换）
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[..., 2] = 0   # 只考虑水平方向

        # c. 单位方向向量（世界坐标 -> 目标坐标）
        rpos_clipped = rpos / distance.clamp(1e-6) # 归一化
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d)

        # d. 速度（世界坐标 -> 目标坐标）
        vel_w = self.root_state[..., 7:10] # 世界速度
        vel_g = vec_to_new_frame(vel_w, target_dir_2d)

        # 组合无人机状态: [相对位置(3), 水平距离(1), 垂直距离(1), 速度(3)] = 8
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).squeeze(1)

        # =========================================================================
        # 观测 III: 动态障碍物状态（如果启用）
        # =========================================================================
        if (self.cfg.env_dyn.num_obstacles != 0):
            # a. 找到最近的 N 个障碍物
            dyn_obs_pos_expanded = self.dyn_obs_state[..., :3].unsqueeze(0).repeat(self.num_envs, 1, 1)
            dyn_obs_rpos_expanded = dyn_obs_pos_expanded[..., :3] - self.root_state[..., :3]
            # 2D 障碍物高度设为 0（只考虑水平距离） 
            dyn_obs_rpos_expanded[:, int(self.dyn_obs_state.size(0)/2):, 2] = 0.
            # 计算水平距离
            dyn_obs_distance_2d = torch.norm(dyn_obs_rpos_expanded[..., :2], dim=2)  # Shape: (1000, 40). calculate 2d distance to each obstacle for all drones
            # 选择最近的 N 个
            _, closest_dyn_obs_idx = torch.topk(dyn_obs_distance_2d, self.cfg.algo.feature_extractor.dyn_obs_num, dim=1, largest=False) # pick top N closest obstacle index
            # 标记超出范围的障碍物
            dyn_obs_range_mask = dyn_obs_distance_2d.gather(1, closest_dyn_obs_idx) > self.lidar_range

            # 获取最近障碍物的相对位置
            closest_dyn_obs_rpos = torch.gather(dyn_obs_rpos_expanded, 1, closest_dyn_obs_idx.unsqueeze(-1).expand(-1, -1, 3))
            # 转换到目标坐标系
            closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos, target_dir_2d) 
            closest_dyn_obs_rpos_g[dyn_obs_range_mask] = 0. # 超出范围的置零
            # 计算距离
            closest_dyn_obs_distance = closest_dyn_obs_rpos_g.norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
            closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

            # b. 障碍物速度（目标坐标系）
            closest_dyn_obs_vel = self.dyn_obs_vel[closest_dyn_obs_idx]
            closest_dyn_obs_vel[dyn_obs_range_mask] = 0.
            closest_dyn_obs_vel_g = vec_to_new_frame(closest_dyn_obs_vel, target_dir_2d) 

            # c. 障碍物大小类别
            closest_dyn_obs_size = self.dyn_obs_size[closest_dyn_obs_idx] 
            closest_dyn_obs_width = closest_dyn_obs_size[..., 0].unsqueeze(-1)

            closest_dyn_obs_width_category = closest_dyn_obs_width / self.dyn_obs_width_res - 1. # 宽度类别: [0, 1, 2, 3]
            closest_dyn_obs_width_category[dyn_obs_range_mask] = 0.

            closest_dyn_obs_height = closest_dyn_obs_size[..., 2].unsqueeze(-1)
            # 高度类别: 0 表示 2D 障碍物，其他表示 3D
            closest_dyn_obs_height_category = torch.where(closest_dyn_obs_height > self.max_obs_3d_height, torch.tensor(0.0), closest_dyn_obs_height)
            closest_dyn_obs_height_category[dyn_obs_range_mask] = 0.

            # 组合动态障碍物状态
            dyn_obs_states = torch.cat([
                closest_dyn_obs_rpos_gn,          # 归一化相对位置 (3)
                closest_dyn_obs_distance_2d,       # 水平距离 (1)
                closest_dyn_obs_distance_z,        # 垂直距离 (1)
                closest_dyn_obs_vel_g,             # 速度 (3)
                closest_dyn_obs_width_category,    # 宽度类别 (1)
                closest_dyn_obs_height_category    # 高度类别 (1)
            ], dim=-1).unsqueeze(1)  # [num_envs, 1, N, 10]

            # 碰撞检测
            closest_dyn_obs_distance_2d_collsion = closest_dyn_obs_rpos[..., :2].norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_2d_collsion[dyn_obs_range_mask] = float('inf')
            closest_dyn_obs_distance_zn_collision = closest_dyn_obs_rpos[..., 2].unsqueeze(-1).norm(dim=-1, keepdim=True)
            closest_dyn_obs_distance_zn_collision[dyn_obs_range_mask] = float('inf')
            # 2D 和 Z 方向同时满足才认为碰撞
            dynamic_collision_2d = closest_dyn_obs_distance_2d_collsion <= (closest_dyn_obs_width/2. + 0.3)
            dynamic_collision_z = closest_dyn_obs_distance_zn_collision <= (closest_dyn_obs_height/2. + 0.3)
            dynamic_collision_each = dynamic_collision_2d & dynamic_collision_z
            dynamic_collision = torch.any(dynamic_collision_each, dim=1)

            # 用于奖励计算的动态障碍物距离
            closest_dyn_obs_distance_reward = closest_dyn_obs_rpos.norm(dim=-1) - closest_dyn_obs_size[..., 0]/2. # for those 2D obstacle, z distance will not be considered
            closest_dyn_obs_distance_reward[dyn_obs_range_mask] = self.cfg.sensor.lidar_range
            
        else:
            dyn_obs_states = torch.zeros(self.num_envs, 1, self.cfg.algo.feature_extractor.dyn_obs_num, 10, device=self.cfg.device)
            dynamic_collision = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.cfg.device)

        # =========================================================================
        # 组合最终观测
        # =========================================================================
        obs = {
            "state": drone_state,
            "lidar": self.lidar_scan,
            "direction": target_dir_2d,
            "dynamic_obstacle": dyn_obs_states
        }

        # =========================================================================
        # 奖励计算
        # =========================================================================
        # a. 静态障碍物安全奖励: 基于 LiDAR 距离的对数，距离越近奖励越低（负值越大），鼓励保持安全距离
        reward_safety_static = torch.log((self.lidar_range-self.lidar_scan).clamp(min=1e-6, max=self.lidar_range)).mean(dim=(2, 3))

        # b. 动态障碍物安全奖励
        if (self.cfg.env_dyn.num_obstacles != 0):
            reward_safety_dynamic = torch.log((closest_dyn_obs_distance_reward).clamp(min=1e-6, max=self.lidar_range)).mean(dim=-1, keepdim=True)

        # c. 速度奖励: 朝向目标方向的速度分量
        vel_direction = rpos / distance.clamp_min(1e-6)
        reward_vel = (self.drone.vel_w[..., :3] * vel_direction).sum(-1)#.clip(max=2.0)
        
        # d. 平滑性惩罚: 速度变化的大小
        penalty_smooth = (self.drone.vel_w[..., :3] - self.prev_drone_vel_w).norm(dim=-1)
        
        # e. 高度惩罚: 飞出起点-目标高度范围时惩罚
        penalty_height = torch.zeros(self.num_envs, 1, device=self.cfg.device)
        penalty_height[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)] = ( (self.drone.pos[..., 2] - self.height_range[..., 1] - 0.2)**2 )[self.drone.pos[..., 2] > (self.height_range[..., 1] + 0.2)]
        penalty_height[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)] = ( (self.height_range[..., 0] - 0.2 - self.drone.pos[..., 2])**2 )[self.drone.pos[..., 2] < (self.height_range[..., 0] - 0.2)]


        # f. 碰撞检测，LiDAR 读数接近最大值表示有障碍物非常近
        static_collision = einops.reduce(self.lidar_scan, "n 1 w h -> n 1", "max") >  (self.lidar_range - 0.3) # 0.3 collision radius
        collision = static_collision | dynamic_collision
        
        # 最终奖励组合
        if (self.cfg.env_dyn.num_obstacles != 0):
            self.reward = (
                reward_vel +                    # 朝向目标的速度
                1. +                            # 生存奖励
                reward_safety_static * 1.0 +    # 静态障碍物安全
                reward_safety_dynamic * 1.0 -   # 动态障碍物安全
                penalty_smooth * 0.1 -          # 平滑性惩罚
                penalty_height * 8.0            # 高度惩罚（权重较大）
            )
        else:
            self.reward = (
                reward_vel + 
                1. + 
                reward_safety_static * 1.0 - 
                penalty_smooth * 0.1 - 
                penalty_height * 8.0
            )

        # =========================================================================
        # 终止条件
        # =========================================================================
        # 到达目标（距离 < 0.5m）
        reach_goal = (distance.squeeze(-1) < 0.5)
        # 高度越界
        below_bound = self.drone.pos[..., 2] < 0.2   # 低于 0.2m
        above_bound = self.drone.pos[..., 2] > 4.    # 高于 4m

        # 终止条件：越界或碰撞
        self.terminated = below_bound | above_bound | collision

        # 截断条件：达到最大回合长度
        self.truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)

        # 更新上一时刻速度（用于下次平滑性计算）
        self.prev_drone_vel_w = self.drone.vel_w[..., :3].clone()

        # =========================================================================
        # 更新统计信息
        # =========================================================================        
        self.stats["return"] += self.reward           # 累积回报
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)  # 回合长度
        self.stats["reach_goal"] = reach_goal.float() # 是否到达目标
        self.stats["collision"] = collision.float()   # 是否碰撞
        self.stats["truncated"] = self.truncated.float()  # 是否截断

        return TensorDict({
            "agents": TensorDict(
                {
                    "observation": obs,
                }, 
                [self.num_envs]
            ),
            "stats": self.stats.clone(),
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        """
        计算奖励和终止条件
        """
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": terminated | truncated,     # 任一条件满足即结束
                "terminated": terminated,           # 终止（碰撞/到达/越界）
                "truncated": truncated,             # 截断（超时）
            },
            self.batch_size,
        )
