# =============================================================================
# Navigation Launch File 导航主模块启动文件
# =============================================================================
# 该文件用于启动导航系统的核心控制节点
# 启动命令: ros2 launch navigation_runner navigation.launch.py
# =============================================================================
# 作用流程:
# 1. 加载 navigation_param.yaml 参数文件
# 2. 启动 navigation_node (导航主节点)
# 3. 该节点是导航系统的核心，负责:
#    - 加载 PPO 策略模型
#    - 订阅传感器数据 (里程计)
#    - 调用感知服务 (RayCast, GetDynamicObstacles)
#    - 执行 PPO 策略推理
#    - 发布速度控制指令
# =============================================================================

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # =========================================================================
    # 参数文件路径配置
    # =========================================================================
    # 获取 navigation_runner 功能包的共享目录路径
    # 参数文件位于: <package_share_directory>/cfg/
    
    # 导航参数文件路径
    navigation_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'navigation_param.yaml'
    )

    # =========================================================================
    # 导航主节点定义
    # =========================================================================
    # -------------------------------------------------------------------------
    # 导航主节点 (Navigation Node)
    # -------------------------------------------------------------------------
    # 主要功能:
    #   1. 加载 PPO 策略模型 (navrl_checkpoint.pt)
    #      - 从 Isaac Sim 训练生成的神经网络权重
    #      - 包含 Actor (策略网络) 和 Critic (价值网络)
    #
    #   2. 订阅传感器数据
    #      - 里程计话题 (/unitree_go2/odom)
    #      - 获取当前位置、速度、朝向
    #
    #   3. 调用感知服务获取环境信息
    #      - RayCast 服务 (/occupancy_map/raycast)
    #        获取 LiDAR 扫描数据 [1, 1, 36, 4]
    #      - GetDynamicObstacles 服务 (/onboard_detector/get_dynamic_obstacles)
    #        获取动态障碍物信息 [1, 1, 5, 10]
    #
    #   4. 构建观测向量
    #      - 状态观测: 位置、速度、目标方向
    #      - LiDAR 观测: 射线扫描数据
    #      - 动态障碍物观测: 相对位置、速度、尺寸
    #
    #   5. PPO 策略推理
    #      - 输入观测向量到神经网络
    #      - Actor 网络输出动作 (速度指令)
    #      - 动作范围通过 Beta 分布映射到 [0, 1]
    #
    #   6. 发布控制指令
    #      - 发布到 /unitree_go2/cmd_vel (geometry_msgs/Twist)
    #      - 控制无人机飞行速度
    #
    # 输入话题 (订阅):
    #   - /unitree_go2/odom (nav_msgs/Odometry)
    #     包含: 位置 pose.pose.position (x, y, z)
    #           朝向 pose.pose.orientation (qx, qy, qz, qw)
    #           线速度 twist.twist.linear (vx, vy, vz)
    #           角速度 twist.twist.angular (wx, wy, wz)
    #
    # 输出话题 (发布):
    #   - /unitree_go2/cmd_vel (geometry_msgs/Twist)
    #     包含: 线速度 linear (x, y, z)
    #           角速度 angular (x, y, z)
    #
    # 调用的服务:
    #   - /occupancy_map/raycast (RayCast)
    #     请求: 当前位置、查询范围、射线参数
    #     响应: 射线击中点坐标
    #
    #   - /onboard_detector/get_dynamic_obstacles (GetDynamicObstacles)
    #     请求: 当前位置、查询范围
    #     响应: 动态障碍物列表 (位置、速度、尺寸)
    #
    # 控制循环频率:
    #   - 通常以 10-30 Hz 运行
    #   - 每周期: 获取观测 -> 策略推理 -> 发布指令
    #
    # 依赖关系:
    #   - 必须先启动 perception.launch.py (提供地图和障碍物服务)
    #   - 可选: 启动 safe_action.launch.py (提供安全保护)
    #   - 可选: 启动 rviz.launch.py (提供可视化)
    #
    # 使用场景:
    #   1. 部署训练好的 PPO 策略进行真实飞行
    #   2. 测试 Sim-to-Real 迁移效果
    #   3. 收集真实环境数据进行策略改进
    #
    # 注意事项:
    #   • 确保 PPO 模型文件 (navrl_checkpoint.pt) 存在于 scripts/ckpts/ 目录
    #   • 确保感知模块已启动，否则服务调用会失败
    #   • 首次部署建议先在仿真环境验证
    #   • 注意速度限制，避免过快飞行
    # -------------------------------------------------------------------------
    safe_navigation_node = Node(
        package='navigation_runner',        # 功能包名称
        executable='navigation_node.py',    # Python可执行文件
        # name='navigation_node',           # 节点名称 (使用默认)
        output='screen',                    # 输出到屏幕
        parameters=[navigation_param_path]  # 加载参数文件
    )

    # =========================================================================
    # 启动描述
    # =========================================================================
    # 返回 LaunchDescription，包含 navigation_node
    #
    # 启动后执行流程:
    #   1. 加载参数文件 navigation_param.yaml
    #   2. 启动 navigation_node.py 脚本
    #   3. 脚本初始化:
    #      - 加载 PPO 模型
    #      - 创建 ROS2 订阅者和发布者
    #      - 创建服务客户端
    #   4. 进入主循环，等待里程计数据
    #   5. 开始控制循环:
    #      - 获取当前位姿
    #      - 调用 RayCast 服务
    #      - 调用 GetDynamicObstacles 服务
    #      - 构建观测
    #      - PPO 策略推理
    #      - 发布速度指令
    #
    # 典型启动序列:
    #   Terminal 1: ros2 launch navigation_runner perception.launch.py
    #   Terminal 2: ros2 launch navigation_runner navigation.launch.py
    #   Terminal 3: ros2 launch navigation_runner rviz.launch.py (可选)
    #
    # 停止导航:
    #   - 按 Ctrl+C 停止 launch 文件
    #   - 或发送零速度指令使无人机悬停
    # =========================================================================
    return LaunchDescription([
        safe_navigation_node,   # 启动导航主节点
    ])
