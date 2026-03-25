# =============================================================================
# RViz Launch File 可视化启动文件
# =============================================================================
# 该文件用于启动 RViz2 可视化工具，显示导航系统的状态信息
# 启动命令: ros2 launch navigation_runner rviz.launch.py
# =============================================================================
# 作用流程:
# 1. 加载 RViz 配置文件 (navigation.rviz)
# 2. 启动 rviz2 节点
# 3. 显示导航系统的可视化信息，包括:
#    - 占据地图 (Occupancy Map)
#    - 无人机位姿和轨迹
#    - LiDAR 射线投射结果
#    - 动态障碍物标记
#    - 目标点标记
#    - ESDF 地图 (可选)
# =============================================================================

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # =========================================================================
    # 获取功能包目录
    # =========================================================================
    # 获取 navigation_runner 功能包的共享目录路径
    # RViz 配置文件位于: <package_share_directory>/rviz/
    package_name = 'navigation_runner'
    
    # 构建 RViz 配置文件的完整路径
    # 该配置文件预先定义了要显示的话题、颜色、视角等
    rviz_config_path = os.path.join(
        get_package_share_directory(package_name), 
        'rviz', 
        'navigation.rviz'  # RViz 配置文件
    )

    # =========================================================================
    # 启动参数声明
    # =========================================================================
    # 声明 rviz_config 启动参数
    # 允许用户在命令行指定自定义的 RViz 配置文件
    # 默认使用功能包自带的 navigation.rviz
    #
    # 使用示例:
    #   ros2 launch navigation_runner rviz.launch.py
    #   ros2 launch navigation_runner rviz.launch.py rviz_config:=/path/to/custom.rviz
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',                              # 参数名称
        default_value=rviz_config_path,             # 默认值
        description='Full path to the RViz config file'  # 参数说明
    )

    # =========================================================================
    # RViz 节点定义
    # =========================================================================
    # -------------------------------------------------------------------------
    # RViz2 可视化节点
    # -------------------------------------------------------------------------
    # 主要功能:
    #   1. 3D 可视化 ROS2 话题数据
    #   2. 显示导航系统的实时状态
    #   3. 提供交互式界面查看传感器数据
    #
    # 可视化内容 (根据 navigation.rviz 配置):
    #   • 占据地图 (/occupancy_map)
    #     - 3D 体素网格显示环境障碍物
    #     - 颜色编码: 红色=占据, 绿色=空闲
    #
    #   • 无人机模型 (/robot_description)
    #     - 显示无人机在 3D 空间中的位姿
    #     - 实时更新位置和朝向
    #
    #   • LiDAR 扫描 (/lidar_scan_visualization)
    #     - 显示射线投射结果
    #     - 射线击中点用点云显示
    #
    #   • 动态障碍物 (/dynamic_obstacles)
    #     - 用立方体标记显示检测到的障碍物
    #     - 颜色区分动态(红色)和静态(蓝色)
    #
    #   • 目标点 (/goal_marker)
    #     - 显示导航目标位置
    #     - 通常为绿色球体或箭头
    #
    #   • 轨迹 (/trajectory)
    #     - 显示无人机飞行轨迹
    #     - 历史路径用线条显示
    #
    # 输入话题 (订阅):
    #   - /occupancy_map (visualization_msgs/MarkerArray)
    #   - /robot_description (sensor_msgs/JointState + urdf)
    #   - /lidar_scan_visualization (sensor_msgs/PointCloud2)
    #   - /dynamic_obstacles (visualization_msgs/MarkerArray)
    #   - /goal_marker (visualization_msgs/Marker)
    #   - /trajectory (nav_msgs/Path)
    #   - /tf (tf2_msgs/TFMessage)
    #
    # 启动参数:
    #   -d <config_file>: 指定 RViz 配置文件路径
    #
    # 使用场景:
    #   1. 调试导航系统时查看传感器数据
    #   2. 监控无人机飞行状态和轨迹
    #   3. 验证地图构建和障碍物检测效果
    #   4. 演示和教学用途
    #
    # 注意事项:
    #   • 需要先启动感知模块 (perception.launch.py) 才能看到地图和障碍物
    #   • 需要先启动导航模块 (navigation.launch.py) 才能看到轨迹和目标
    #   • RViz 是可视化工具，不参与实际控制
    # -------------------------------------------------------------------------
    rviz_node = Node(
        package='rviz2',                    # 功能包名称
        executable='rviz2',                 # 可执行文件名称
        name='rviz2',                       # 节点名称
        output='screen',                    # 输出到屏幕
        arguments=['-d', LaunchConfiguration('rviz_config')]  # 加载配置文件
    )

    # =========================================================================
    # 启动描述
    # =========================================================================
    # 返回 LaunchDescription，包含:
    #   1. rviz_config_arg: 启动参数声明
    #   2. rviz_node: RViz2 可视化节点
    #
    # 启动后效果:
    #   • RViz2 窗口打开
    #   • 加载预设的可视化配置
    #   • 开始订阅和显示 ROS2 话题数据
    #   • 用户可以通过界面交互查看不同视角
    #
    # 典型使用流程:
    #   Terminal 1: ros2 launch navigation_runner perception.launch.py
    #   Terminal 2: ros2 launch navigation_runner navigation.launch.py
    #   Terminal 3: ros2 launch navigation_runner rviz.launch.py
    # =========================================================================
    return LaunchDescription([
        rviz_config_arg,    # 声明启动参数 (必须先声明)
        rviz_node,          # 启动 RViz2 节点
    ])
