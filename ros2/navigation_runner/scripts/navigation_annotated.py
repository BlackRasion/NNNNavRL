#!/usr/bin/env python3
# =============================================================================
# Navigation Node 导航主节点
# =============================================================================
# 该文件实现导航系统的核心控制逻辑
# 主要功能:
#   1. 加载 PPO 策略模型并进行推理
#   2. 订阅里程计数据获取当前位姿
#   3. 调用感知服务获取环境信息 (LiDAR, 动态障碍物)
#   4. 构建观测向量并输入神经网络
#   5. 发布速度控制指令驱动无人机
# =============================================================================

from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from builtin_interfaces.msg import Duration
from map_manager.srv import RayCast
from onboard_detector.srv import GetDynamicObstacles
from navigation_runner.srv import GetSafeAction
import torch
import numpy as np
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from ppo import PPO
from utils import vec_to_new_frame, vec_to_world
from pid_controller import AnglePIDController
import os


class Navigation(Node):
    """
    导航主节点类
    
    该类是 ROS2 节点，负责:
    - 加载和执行 PPO 策略模型
    - 处理传感器数据
    - 生成控制指令
    """
    
    def __init__(self, cfg):
        """
        初始化导航节点
        
        参数:
            cfg: 配置对象，包含传感器参数、算法参数等
        """
        super().__init__('navigation_node')
        self.cfg = cfg
        
        # =========================================================================
        # LiDAR 参数计算
        # =========================================================================
        # 水平扫描线数 = 360度 / 水平分辨率
        self.lidar_hbeams = int(360/self.cfg.sensor.lidar_hres)
        
        # 射线击中点列表 (用于可视化)
        self.raypoints = []
        
        # 动态障碍物信息 (位置, 速度, 尺寸)
        self.dynamic_obstacles = []
        
        # 机器人半径 (用于安全距离计算)
        self.robot_size = 0.3
        
        # 射线垂直分辨率 (弧度)
        self.raycast_vres = ((self.cfg.sensor.lidar_vfov[1] - self.cfg.sensor.lidar_vfov[0]))/(self.cfg.sensor.lidar_vbeams - 1) * np.pi/180.0
        
        # 射线水平分辨率 (弧度)
        self.raycast_hres = self.cfg.sensor.lidar_hres * np.pi/180.0

        # =========================================================================
        # 状态变量初始化
        # =========================================================================
        self.goal = None                    # 目标点
        self.goal_received = False          # 是否收到目标点
        self.target_dir = None              # 目标方向向量
        self.stable_times = 0               # 稳定次数计数器
        self.has_action = False             # 是否有有效动作
        self.laser_points_msg = None        # LiDAR点云消息

        # 功能开关
        self.height_control = False         # 是否控制高度
        self.use_policy_server = False      # 是否使用策略服务器
        self.odom_received = False          # 是否收到里程计
        self.safety_stop = False            # 安全停止标志
    
        # =========================================================================
        # 参数声明与加载
        # =========================================================================
        # 速度限制参数
        self.declare_parameter('vel_limit', 1.0)
        self.vel_limit = self.get_parameter('vel_limit').get_parameter_value().double_value
        self.get_logger().info(f"[导航节点]: 速度限制: {self.vel_limit} m/s")

        # 射线可视化参数
        self.declare_parameter('visualize_raycast', False)
        self.vis_raycast = self.get_parameter('visualize_raycast').get_parameter_value().bool_value
        self.get_logger().info(f"[导航节点]: 射线可视化 {'开启' if self.vis_raycast else '关闭'}")

        # =========================================================================
        # 订阅者 (Subscriber) 配置
        # =========================================================================
        # 里程计话题
        self.declare_parameter('odom_topic', '/unitree_go2/odom')
        odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.get_logger().info(f"[导航节点]: 里程计话题: {odom_topic}")
        self.odom_sub = self.create_subscription(
            Odometry, 
            odom_topic, 
            self.odom_callback, 
            10
        )
        
        # 目标点话题
        self.goal_sub = self.create_subscription(
            PoseStamped, 
            '/goal_pose', 
            self.goal_callback, 
            10
        )
        
        # 紧急停止话题
        self.emergency_stop_sub = self.create_subscription(
            Bool, 
            '/navigation_emergency_stop', 
            self.safety_check_callback, 
            10
        )
        
        # =========================================================================
        # 发布者 (Publisher) 配置
        # =========================================================================
        # 控制指令话题
        self.declare_parameter('cmd_topic', '/unitree_go2/cmd_vel')
        cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.get_logger().info(f"[导航节点]: 控制指令话题: {cmd_topic}")
        self.action_pub = self.create_publisher(Twist, cmd_topic, 10)
        
        # 目标点可视化话题
        self.goal_vis_pub = self.create_publisher(MarkerArray, 'navigation_runner/goal', 10)

        # =========================================================================
        # 服务客户端 (Service Client) 配置
        # =========================================================================
        # RayCast 服务客户端 (获取LiDAR数据)
        raycast_client_group = MutuallyExclusiveCallbackGroup()
        self.raycast_client = self.create_client(
            RayCast, 
            '/occupancy_map/raycast', 
            callback_group=raycast_client_group
        )
        
        # 动态障碍物服务客户端
        dynamic_obstacle_client_group = MutuallyExclusiveCallbackGroup()
        self.get_dyn_obs_client = self.create_client(
            GetDynamicObstacles, 
            '/onboard_detector/get_dynamic_obstacles', 
            callback_group=dynamic_obstacle_client_group
        )
        
        # 安全动作服务客户端
        safe_action_client_group = MutuallyExclusiveCallbackGroup()
        self.get_safe_action_client = self.create_client(
            GetSafeAction, 
            '/safe_action/get_safe_action', 
            callback_group=safe_action_client_group
        )
        
        # 等待 RayCast 服务可用
        while not self.raycast_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('[导航节点]: 等待地图服务 (/occupancy_map/raycast) 可用...')
        self.get_logger().info('[导航节点]: 地图服务已连接')

        # =========================================================================
        # 控制器初始化
        # =========================================================================
        # 角度 PID 控制器 (用于朝向控制)
        self.angle_controller = AnglePIDController(
            kp=1.0, 
            ki=0.0, 
            kd=0.1, 
            dt=0.05, 
            max_angular_velocity=1.0
        )

        # =========================================================================
        # 模型加载
        # =========================================================================
        self.declare_parameter('checkpoint_file', 'navrl_checkpoint.pt')
        ckpt_file = self.get_parameter('checkpoint_file').get_parameter_value().string_value
        self.get_logger().info(f"[导航节点]: 加载模型: {ckpt_file}")
        self.policy = self.init_model(ckpt_file)
        self.policy.eval()
        self.get_logger().info('[导航节点]: 模型加载完成，导航节点初始化成功')


    def init_model(self, ckpt_file):
        """
        初始化 PPO 策略模型
        
        参数:
            ckpt_file: 模型检查点文件名
            
        返回:
            policy: 加载好的 PPO 策略网络
        """
        # 观测维度定义
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        
        # 构建观测空间规格
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    # 状态观测: [目标方向(3), 距离2D, 距离Z, 速度(3)]
                    "state": UnboundedContinuousTensorSpec(
                        (observation_dim,), 
                        device=self.cfg.device
                    ), 
                    # LiDAR观测: [1, 水平线数, 垂直线数]
                    "lidar": UnboundedContinuousTensorSpec(
                        (1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams), 
                        device=self.cfg.device
                    ),
                    # 目标方向
                    "direction": UnboundedContinuousTensorSpec(
                        (1, 3), 
                        device=self.cfg.device
                    ),
                    # 动态障碍物观测
                    "dynamic_obstacle": UnboundedContinuousTensorSpec(
                        (1, self.cfg.algo.feature_extractor.dyn_obs_num, num_dim_each_dyn_obs_state), 
                        device=self.cfg.device
                    ),
                }),
            }).expand(1)
        }, shape=[1], device=self.cfg.device)

        # 动作维度 (x, y, z 三轴速度)
        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec(
                    (action_dim,), 
                    device=self.cfg.device
                ), 
            })
        }).expand(1, action_dim).to(self.cfg.device)

        # 创建 PPO 策略网络
        policy = PPO(self.cfg.algo, observation_spec, action_spec, self.cfg.device)
        
        # 加载模型权重
        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")
        checkpoint = os.path.join(file_dir, ckpt_file)
        policy.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=self.cfg.device))
        
        self.get_logger().info(f"[导航节点]: 模型从 {checkpoint} 加载成功")
        return policy


    def safety_check_callback(self, msg):
        """
        安全检查回调函数
        
        参数:
            msg: Bool 消息，True 表示触发紧急停止
        """
        if msg.data == True:
            self.safety_stop = True
            self.get_logger().warn('[导航节点]: 收到紧急停止信号')
        else:
            self.safety_stop = False
            self.get_logger().info('[导航节点]: 紧急停止解除')


    def odom_callback(self, odom):
        """
        里程计回调函数
        
        参数:
            odom: Odometry 消息，包含位置和速度信息
        """
        self.odom = odom
        self.odom_received = True


    def goal_callback(self, goal):
        """
        目标点回调函数
        
        参数:
            goal: PoseStamped 消息，包含目标位置
        """
        # 如果还没有收到里程计，不处理目标点
        if not self.odom_received:
            self.get_logger().warn('[导航节点]: 尚未收到里程计数据，无法设置目标点')
            return

        self.goal = goal
        # 保持当前高度
        self.goal.pose.position.z = self.odom.pose.pose.position.z
        
        # 计算目标方向向量
        dir_x = self.goal.pose.position.x - self.odom.pose.pose.position.x
        dir_y = self.goal.pose.position.y - self.odom.pose.pose.position.y
        dir_z = self.goal.pose.position.z - self.odom.pose.pose.position.z
        self.target_dir = torch.tensor([dir_x, dir_y, dir_z], device=self.cfg.device)

        self.goal_received = True
        distance = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        self.get_logger().info(f'[导航节点]: 收到目标点: ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f}, {goal.pose.position.z:.2f}), 距离: {distance:.2f}m')


    def get_raycast(self, pos, start_angle):
        """
        调用 RayCast 服务获取 LiDAR 扫描数据
        
        参数:
            pos: 当前位置 [x, y, z]
            start_angle: 起始角度 (弧度)
            
        返回:
            raypoints: 射线击中点列表 [[x1,y1,z1], [x2,y2,z2], ...]
        """
        raypoints = []
        
        # 构建位置消息
        pos_msg = Point()
        pos_msg.x = pos[0]
        pos_msg.y = pos[1]
        pos_msg.z = pos[2]

        # 构建服务请求
        request = RayCast.Request()
        request.position = pos_msg
        request.start_angle = float(start_angle)
        request.range = float(self.cfg.sensor.lidar_range)
        request.vfov_min = float(self.cfg.sensor.lidar_vfov[0])
        request.vfov_max = float(self.cfg.sensor.lidar_vfov[1])
        request.vbeams = int(self.cfg.sensor.lidar_vbeams)
        request.hres = float(self.cfg.sensor.lidar_hres)
        request.visualize = self.vis_raycast

        # 调用服务
        response = self.raycast_client.call(request)
        num_points = int(len(response.points) / 3)
        self.laser_points_msg = response.points
        
        # 解析响应数据
        for i in range(num_points):
            p = [
                response.points[3 * i + 0],
                response.points[3 * i + 1],
                response.points[3 * i + 2]
            ]
            raypoints.append(p)

        return raypoints


    def get_dynamic_obstacles(self, pos):
        """
        调用服务获取动态障碍物信息
        
        参数:
            pos: 当前位置 [x, y, z]
            
        返回:
            dynamic_obstacle_pos: 障碍物位置 [N, 3]
            dynamic_obstacle_vel: 障碍物速度 [N, 3]
            dynamic_obstacle_size: 障碍物尺寸 [N, 3]
        """
        # 初始化障碍物张量
        max_obs_num = self.cfg.algo.feature_extractor.dyn_obs_num
        dynamic_obstacle_pos = torch.zeros(max_obs_num, 3, dtype=torch.float, device=self.cfg.device)
        dynamic_obstacle_vel = torch.zeros(max_obs_num, 3, dtype=torch.float, device=self.cfg.device)
        dynamic_obstacle_size = torch.zeros(max_obs_num, 3, dtype=torch.float, device=self.cfg.device)

        # 查询范围
        distance_range = 4.0
        
        # 构建位置消息
        pos_msg = Point()
        pos_msg.x = pos[0]
        pos_msg.y = pos[1]
        pos_msg.z = pos[2]
        
        # 构建服务请求
        request = GetDynamicObstacles.Request()
        request.current_position = pos_msg
        request.range = float(distance_range)

        # 调用服务
        response = self.get_dyn_obs_client.call(request)
        total_obs_num = len(response.position)

        # 填充障碍物数据
        for i in range(min(max_obs_num, total_obs_num)):
            pos_vec = response.position[i]
            vel_vec = response.velocity[i]
            size_vec = response.size[i]

            dynamic_obstacle_pos[i] = torch.tensor(
                [pos_vec.x, pos_vec.y, pos_vec.z],
                dtype=torch.float, device=self.cfg.device
            )
            dynamic_obstacle_vel[i] = torch.tensor(
                [vel_vec.x, vel_vec.y, vel_vec.z],
                dtype=torch.float, device=self.cfg.device
            )
            dynamic_obstacle_size[i] = torch.tensor(
                [size_vec.x, size_vec.y, size_vec.z],
                dtype=torch.float, device=self.cfg.device
            )

        return dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size


    def check_obstacle(self, lidar_scan, dyn_obs_states):
        """
        检查范围内是否有障碍物
        
        参数:
            lidar_scan: LiDAR扫描数据 [1, 1, 36, 4]
            dyn_obs_states: 动态障碍物状态 [1, 1, 5, 10]
            
        返回:
            bool: 是否有障碍物
        """
        quarter_size = lidar_scan.shape[2] // 4
        
        # 检查前方区域 (前1/4和后1/4)
        first_quarter_check = torch.all(lidar_scan[:, :, :quarter_size, 1:] < 0.2)
        last_quarter_check = torch.all(lidar_scan[:, :, -quarter_size:, 1:] < 0.2)
        
        has_static = (not first_quarter_check) or (not last_quarter_check)
        has_dynamic = not torch.all(dyn_obs_states == 0.)
        
        return has_static or has_dynamic


    def get_safe_action(self, vel_world, action_vel_world):
        """
        调用安全动作服务获取安全修正后的速度
        
        参数:
            vel_world: 当前世界坐标系速度
            action_vel_world: 策略输出的原始速度
            
        返回:
            safe_action: 安全修正后的速度
        """
        if self.get_safe_action_client.service_is_ready():
            safe_action = np.zeros(3)
            
            # 构建位置消息
            pos_msg = Point(
                x=self.odom.pose.pose.position.x, 
                y=self.odom.pose.pose.position.y, 
                z=self.odom.pose.pose.position.z
            )
            
            # 构建速度消息
            vel_msg = Vector3(
                x=vel_world[0].item(), 
                y=vel_world[1].item(), 
                z=vel_world[2].item()
            )
            action_vel_msg = Vector3(
                x=action_vel_world[0].item(), 
                y=action_vel_world[1].item(), 
                z=action_vel_world[2].item()
            )
            
            max_vel = np.sqrt(2. * self.vel_limit**2)
            
            # 构建障碍物列表
            obstacle_pos_list = []
            obstacle_vel_list = []
            obstacle_size_list = []
            
            for i in range(len(self.dynamic_obstacles[0])):
                if self.dynamic_obstacles[2][i][0] != 0:
                    obs_pos = Vector3(
                        x=self.dynamic_obstacles[0][i][0].item(),
                        y=self.dynamic_obstacles[0][i][1].item(),
                        z=self.dynamic_obstacles[0][i][2].item()
                    )
                    obs_vel = Vector3(
                        x=self.dynamic_obstacles[1][i][0].item(),
                        y=self.dynamic_obstacles[1][i][1].item(),
                        z=self.dynamic_obstacles[1][i][2].item()
                    )
                    obs_size = Vector3(
                        x=self.dynamic_obstacles[2][i][0].item(),
                        y=self.dynamic_obstacles[2][i][1].item(),
                        z=self.dynamic_obstacles[2][i][2].item()
                    )
                    obstacle_pos_list.append(obs_pos)
                    obstacle_vel_list.append(obs_vel)
                    obstacle_size_list.append(obs_size)

            # 构建服务请求
            request = GetSafeAction.Request()
            request.agent_position = pos_msg
            request.agent_velocity = vel_msg
            request.agent_size = self.robot_size
            request.obs_position = obstacle_pos_list
            request.obs_velocity = obstacle_vel_list
            request.obs_size = obstacle_size_list
            request.laser_points = self.laser_points_msg
            request.laser_range = float(self.cfg.sensor.lidar_range)
            request.laser_res = float(max(self.raycast_vres, self.raycast_hres))
            request.max_velocity = float(max_vel)
            request.rl_velocity = action_vel_msg

            # 调用服务
            response = self.get_safe_action_client.call(request)
            safe_action = np.array([
                response.safe_action.x,
                response.safe_action.y,
                response.safe_action.z
            ])
            return safe_action
        else:
            # 如果服务不可用，返回原始动作
            return action_vel_world


    def raycast_callback(self):
        """
        RayCast 定时回调函数 (20Hz)
        
        定期调用 RayCast 服务更新 LiDAR 数据
        """
        if not self.odom_received or not self.goal_received:
            return
            
        pos = np.array([
            self.odom.pose.pose.position.x, 
            self.odom.pose.pose.position.y, 
            self.odom.pose.pose.position.z
        ])
        
        # 计算起始角度 (朝向目标方向)
        start_angle = np.arctan2(
            self.target_dir[1].cpu().numpy(), 
            self.target_dir[0].cpu().numpy()
        )
        
        self.raypoints = self.get_raycast(pos, start_angle)


    def dynamic_obstacle_callback(self):
        """
        动态障碍物定时回调函数 (20Hz)
        
        定期更新动态障碍物信息
        """
        if not self.odom_received:
            return
            
        pos = np.array([
            self.odom.pose.pose.position.x, 
            self.odom.pose.pose.position.y, 
            self.odom.pose.pose.position.z
        ])
        
        dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size = \
            self.get_dynamic_obstacles(pos)
        self.dynamic_obstacles = (dynamic_obstacle_pos, dynamic_obstacle_vel, dynamic_obstacle_size)


    def get_action(self, pos: torch.Tensor, vel: torch.Tensor, goal: torch.Tensor):
        """
        使用 PPO 策略网络计算动作
        
        参数:
            pos: 当前位置 [3]
            vel: 当前速度 [3]
            goal: 目标位置 [3]
            
        返回:
            vel_world: 世界坐标系下的速度指令 [3]
        """
        # 计算相对目标位置和距离
        rpos = goal - pos
        distance = rpos.norm(dim=-1, keepdim=True)
        distance_2d = rpos[..., :2].norm(dim=-1, keepdim=True)
        distance_z = rpos[..., 2].unsqueeze(-1)

        # 目标方向 (水平面)
        target_dir_2d = self.target_dir.clone()
        target_dir_2d[2] = 0.

        # 归一化相对位置方向
        rpos_clipped = rpos / distance.clamp(1e-6)
        rpos_clipped_g = vec_to_new_frame(rpos_clipped, target_dir_2d).squeeze(0).squeeze(0)

        # 相对速度 (转换到目标坐标系)
        vel_g = vec_to_new_frame(vel, target_dir_2d).squeeze(0).squeeze(0)

        # 构建状态观测
        drone_state = torch.cat([rpos_clipped_g, distance_2d, distance_z, vel_g], dim=-1).unsqueeze(0)

        # =========================================================================
        # LiDAR 观测处理
        # =========================================================================
        lidar_scan = torch.tensor(self.raypoints, device=self.cfg.device)
        lidar_scan = (lidar_scan - pos).norm(dim=-1).clamp_max(self.cfg.sensor.lidar_range)
        lidar_scan = lidar_scan.reshape(1, 1, self.lidar_hbeams, self.cfg.sensor.lidar_vbeams)
        lidar_scan = self.cfg.sensor.lidar_range - lidar_scan

        # =========================================================================
        # 动态障碍物观测处理
        # =========================================================================
        dynamic_obstacle_pos = self.dynamic_obstacles[0].clone()
        dynamic_obstacle_vel = self.dynamic_obstacles[1].clone()
        dynamic_obstacle_size = self.dynamic_obstacles[2].clone()
        
        closest_dyn_obs_rpos = dynamic_obstacle_pos - pos
        closest_dyn_obs_rpos[dynamic_obstacle_size[:, 2] == 0] = 0.
        closest_dyn_obs_rpos[:, 2][dynamic_obstacle_size[:, 2] > 1] = 0.
        closest_dyn_obs_rpos_g = vec_to_new_frame(closest_dyn_obs_rpos.unsqueeze(0), target_dir_2d).squeeze(0)
        
        closest_dyn_obs_distance = closest_dyn_obs_rpos.norm(dim=-1, keepdim=True)
        closest_dyn_obs_distance_2d = closest_dyn_obs_rpos_g[..., :2].norm(dim=-1, keepdim=True)
        closest_dyn_obs_distance_z = closest_dyn_obs_rpos_g[..., 2].unsqueeze(-1)
        closest_dyn_obs_rpos_gn = closest_dyn_obs_rpos_g / closest_dyn_obs_distance.clamp(1e-6)

        closest_dyn_obs_vel_g = vec_to_new_frame(dynamic_obstacle_vel.unsqueeze(0), target_dir_2d).squeeze(0)
        
        # 处理障碍物尺寸
        obs_res = 0.25
        closest_dyn_obs_width = torch.max(dynamic_obstacle_size[:, 0], dynamic_obstacle_size[:, 1])
        closest_dyn_obs_width += self.robot_size * 2.
        closest_dyn_obs_width = torch.clamp(
            torch.ceil(closest_dyn_obs_width / 0.25) - 1, 
            min=0, 
            max=1./obs_res - 1
        )
        closest_dyn_obs_width[dynamic_obstacle_size[:, 2] == 0] = 0.
        
        closest_dyn_obs_height = dynamic_obstacle_size[:, 2]
        closest_dyn_obs_height[(closest_dyn_obs_height <= 1) & (closest_dyn_obs_height != 0)] = 1.
        closest_dyn_obs_height[closest_dyn_obs_height > 1] = 0.
        
        # 组合动态障碍物观测
        dyn_obs_states = torch.cat([
            closest_dyn_obs_rpos_gn, 
            closest_dyn_obs_distance_2d, 
            closest_dyn_obs_distance_z, 
            closest_dyn_obs_vel_g,
            closest_dyn_obs_width.unsqueeze(1), 
            closest_dyn_obs_height.unsqueeze(1)
        ], dim=-1).unsqueeze(0).unsqueeze(0)

        # =========================================================================
        # 构建完整观测
        # =========================================================================
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": drone_state,
                    "lidar": lidar_scan,
                    "direction": target_dir_2d,
                    "dynamic_obstacle": dyn_obs_states
                })
            })
        })

        # 检查是否有障碍物
        has_obstacle_in_range = self.check_obstacle(lidar_scan, dyn_obs_states)
        
        if has_obstacle_in_range:
            # 使用 PPO 策略推理
            with set_exploration_type(ExplorationType.MEAN):
                output = self.policy(obs)
            
            # 将归一化动作转换为实际速度
            vel_local_normalized = output["agents", "action_normalized"]
            vel_local_world = 2.0 * vel_local_normalized * self.vel_limit - self.vel_limit
            vel_world = vec_to_world(vel_local_world, output["agents", "observation", "direction"])
        else:
            # 无障碍物时，直接向目标飞行
            vel_world = (goal - pos) / torch.norm(goal - pos) * self.vel_limit
            
        return vel_world


    def control_callback(self):
        """
        控制定时回调函数 (20Hz)
        
        主控制循环，执行策略推理并发布控制指令
        """
        # 检查必要条件
        if not self.odom_received:
            return

        if not self.goal_received or len(self.raypoints) == 0 or len(self.dynamic_obstacles) == 0:
            return

        # 紧急停止检查
        if self.safety_stop:
            final_cmd_vel = Twist()
            final_cmd_vel.linear.x = 0.
            final_cmd_vel.linear.y = 0.
            final_cmd_vel.angular.x = 0.
            self.action_pub.publish(final_cmd_vel)
            self.get_logger().warn('[导航节点]: 紧急停止已触发，无人机悬停')
            return

        # =========================================================================
        # 朝向控制
        # =========================================================================
        goal_angle = np.arctan2(self.target_dir[1].cpu().numpy(), self.target_dir[0].cpu().numpy())
        _, _, curr_angle = self.quaternion_to_euler(
            self.odom.pose.pose.orientation.w, 
            self.odom.pose.pose.orientation.x, 
            self.odom.pose.pose.orientation.y, 
            self.odom.pose.pose.orientation.z
        )
        angle_diff = np.abs(goal_angle - curr_angle)
        angular_velocity = self.angle_controller.compute_angular_velocity(goal_angle, curr_angle)

        # 如果朝向偏差较大，先调整朝向
        if angle_diff >= 0.3:
            final_cmd_vel = Twist()
            final_cmd_vel.angular.z = angular_velocity
            self.action_pub.publish(final_cmd_vel)
            return

        # =========================================================================
        # 位置控制 (PPO 策略)
        # =========================================================================
        pos = torch.tensor([
            self.odom.pose.pose.position.x, 
            self.odom.pose.pose.position.y, 
            self.odom.pose.pose.position.z
        ], device=self.cfg.device)
        
        goal = torch.tensor([
            self.goal.pose.position.x, 
            self.goal.pose.position.y, 
            self.goal.pose.position.z
        ], device=self.cfg.device)
        
        # 计算世界坐标系速度
        rot = self.quaternion_to_rotation_matrix(self.odom.pose.pose.orientation)
        vel_body = np.array([
            self.odom.twist.twist.linear.x, 
            self.odom.twist.twist.linear.y, 
            self.odom.twist.twist.linear.z
        ])
        vel_world = torch.tensor(rot @ vel_body, device=self.cfg.device, dtype=torch.float)

        # 获取 PPO 策略动作
        cmd_vel_world = self.get_action(pos, vel_world, goal).squeeze(0).squeeze(0).detach().cpu().numpy()
        self.cmd_vel_world = cmd_vel_world.copy()

        # 获取安全修正后的速度
        safe_cmd_vel_world = self.get_safe_action(vel_world, cmd_vel_world)
        self.safe_cmd_vel_world = safe_cmd_vel_world.copy()

        # 转换到局部坐标系
        quat_no_tilt = self.euler_to_quaternion(0, 0, curr_angle)
        quat_msg = Quaternion()
        quat_msg.w = quat_no_tilt[0]
        quat_msg.x = quat_no_tilt[1]
        quat_msg.y = quat_no_tilt[2]
        quat_msg.z = quat_no_tilt[3]
        rot_no_tilt = self.quaternion_to_rotation_matrix(quat_msg)
        safe_cmd_vel_local = np.linalg.inv(rot_no_tilt) @ safe_cmd_vel_world

        # =========================================================================
        # 目标接近处理
        # =========================================================================
        distance = (pos - goal).norm()
        
        if distance <= 3.0 and distance > 1.0:
            # 接近目标，减速
            if np.linalg.norm(safe_cmd_vel_local) != 0:
                safe_cmd_vel_local = 1.0 * safe_cmd_vel_local / np.linalg.norm(safe_cmd_vel_local)
                safe_cmd_vel_world = 1.0 * safe_cmd_vel_world / np.linalg.norm(safe_cmd_vel_world)
        elif distance <= 1.0:
            # 到达目标，停止
            safe_cmd_vel_local *= 0.
            safe_cmd_vel_world *= 0.
            if not hasattr(self, 'goal_reached_logged'):
                self.get_logger().info(f'[导航节点]: 已到达目标点，距离: {distance:.2f}m')
                self.goal_reached_logged = True
        else:
            self.goal_reached_logged = False

        # =========================================================================
        # 发布控制指令
        # =========================================================================
        final_cmd_vel = Twist()
        final_cmd_vel.linear.x = safe_cmd_vel_local[0].item()
        final_cmd_vel.linear.y = safe_cmd_vel_local[1].item()
        final_cmd_vel.angular.z = angular_velocity
        
        if self.height_control:
            final_cmd_vel.linear.z = safe_cmd_vel_world[2].item()
        else:
            final_cmd_vel.linear.z = 0.0
            
        self.action_pub.publish(final_cmd_vel)


    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        欧拉角转四元数
        
        参数:
            roll: 横滚角 (弧度)
            pitch: 俯仰角 (弧度)
            yaw: 偏航角 (弧度)
            
        返回:
            (w, x, y, z): 四元数
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (w, x, y, z)


    def quaternion_to_euler(self, w, x, y, z):
        """
        四元数转欧拉角
        
        参数:
            w, x, y, z: 四元数分量
            
        返回:
            (roll, pitch, yaw): 欧拉角 (弧度)
        """
        # 归一化
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w /= norm
        x /= norm
        y /= norm
        z /= norm

        # 横滚角 (X轴旋转)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # 俯仰角 (Y轴旋转)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # 偏航角 (Z轴旋转)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return (roll, pitch, yaw)


    def quaternion_to_rotation_matrix(self, quaternion):
        """
        四元数转旋转矩阵
        
        参数:
            quaternion: Quaternion 消息
            
        返回:
            3x3 旋转矩阵
        """
        w = quaternion.w
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        xx, xy, xz = x**2, x*y, x*z
        yy, yz = y**2, y*z
        zz = z**2
        wx, wy, wz = w*x, w*y, w*z

        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
        ])


    def run(self):
        """
        运行导航节点
        
        创建定时器启动各个回调函数
        """
        # 创建互斥回调组 (防止并发访问)
        raycast_cb_group = MutuallyExclusiveCallbackGroup()
        dynamic_obstacle_cb_group = MutuallyExclusiveCallbackGroup()
        control_cb_group = MutuallyExclusiveCallbackGroup()
        vis_cb_group = MutuallyExclusiveCallbackGroup()
        
        # 创建定时器 (20Hz)
        self.create_timer(0.05, self.raycast_callback, callback_group=raycast_cb_group)
        self.create_timer(0.05, self.dynamic_obstacle_callback, callback_group=dynamic_obstacle_cb_group)
        self.create_timer(0.05, self.control_callback, callback_group=control_cb_group)
        self.create_timer(0.05, self.goal_vis_callback, callback_group=vis_cb_group)
        
        self.get_logger().info('[导航节点]: 导航节点开始运行')


    def goal_vis_callback(self):
        """
        目标点可视化回调函数
        
        发布目标点标记用于 RViz 显示
        """
        if not self.goal_received:
            return
            
        msg = MarkerArray()
        goal_point = Marker()
        goal_point.header.frame_id = "map"
        goal_point.header.stamp = self.get_clock().now().to_msg()
        goal_point.ns = "goal_point"
        goal_point.id = 1
        goal_point.type = goal_point.SPHERE
        goal_point.action = goal_point.ADD
        goal_point.pose.position.x = self.goal.pose.position.x
        goal_point.pose.position.y = self.goal.pose.position.y
        goal_point.pose.position.z = self.goal.pose.position.z
        goal_point.lifetime = Duration(sec=0, nanosec=int(0.1 * 1e9))
        goal_point.scale.x = 0.3
        goal_point.scale.y = 0.3
        goal_point.scale.z = 0.3
        goal_point.color.r = 1.0
        goal_point.color.b = 1.0
        goal_point.color.a = 1.0
        msg.markers.append(goal_point)
        self.goal_vis_pub.publish(msg)
