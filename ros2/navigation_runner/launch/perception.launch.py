# =============================================================================
# Perception Launch File 感知模块启动文件
# =============================================================================
# 该文件用于启动导航系统的感知模块，包括地图管理和障碍物检测
# 启动命令: ros2 launch navigation_runner perception.launch.py
# =============================================================================
# 作用流程:
# 1. 加载三个 YAML 参数文件 (map_param.yaml, dynamic_detector_param.yaml, yolo_detector_param.yaml)
# 2. 启动 occupancy_map_node (占据地图节点)
# 3. 启动 dynamic_detector_node (动态障碍物检测节点)
# 4. 启动 yolo_detector_node (YOLO视觉检测节点)
# 5. 这三个节点协同工作，为导航提供环境感知信息
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
    
    # 地图管理参数文件路径
    # 包含: 深度相机参数、占据地图参数、射线投射参数、可视化参数
    map_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'map_param.yaml'
    )

    # 动态障碍物检测参数文件路径
    # 包含: 传感器配置、DBSCAN聚类参数、卡尔曼滤波参数、动态分类参数
    dynamic_detector_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'dynamic_detector_param.yaml'
    )

    # YOLO视觉检测参数文件路径
    # 包含: RGB相机话题、时间步长、调试可视化开关
    yolo_detector_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'yolo_detector_param.yaml'
    )

    # =========================================================================
    # 节点定义
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 占据地图节点 (Occupancy Map Node)
    # -------------------------------------------------------------------------
    # 主要功能:
    #   1. 订阅深度图像或LiDAR点云，构建3D占据地图
    #   2. 提供 RayCast 服务，用于获取LiDAR扫描数据
    #   3. 提供 CheckPosCollision 服务，检查位置是否碰撞
    #   4. 发布占据地图用于可视化
    #
    # 输入话题:
    #   - /unitree_go2/front_cam/depth_image (sensor_msgs/Image)
    #   - /unitree_go2/lidar/point_cloud (sensor_msgs/PointCloud2)
    #   - /unitree_go2/pose (geometry_msgs/PoseStamped)
    #
    # 输出话题:
    #   - /occupancy_map (visualization_msgs/MarkerArray)
    #
    # 提供服务:
    #   - /occupancy_map/raycast (RayCast)
    #   - /occupancy_map/check_pos_collision (CheckPosCollision)
    # -------------------------------------------------------------------------
    occupancy_map_node = Node(
        package='map_manager',              # 功能包名称
        executable='occupancy_map_node',    # 可执行文件名称
        name='map_manager_node',            # 节点名称
        output='screen',                    # 输出到屏幕
        parameters=[map_param_path]         # 加载参数文件
    )

    # -------------------------------------------------------------------------
    # 动态障碍物检测节点 (Dynamic Detector Node)
    # -------------------------------------------------------------------------
    # 主要功能:
    #   1. 融合深度图像和RGB图像进行障碍物检测
    #   2. 使用 UV-Detector 进行快速2D检测
    #   3. 使用 DBSCAN 进行3D点云聚类
    #   4. 使用卡尔曼滤波进行多目标跟踪
    #   5. 动态/静态障碍物分类
    #
    # 输入话题:
    #   - /unitree_go2/front_cam/depth_image (sensor_msgs/Image)
    #   - /unitree_go2/front_cam/color_image (sensor_msgs/Image)
    #   - /unitree_go2/pose (geometry_msgs/PoseStamped)
    #
    # 输出话题:
    #   - /onboard_detector/dynamic_obstacles (visualization_msgs/MarkerArray)
    #
    # 提供服务:
    #   - /onboard_detector/get_dynamic_obstacles (GetDynamicObstacles)
    # -------------------------------------------------------------------------
    dynamic_detector_node = Node(
        package='onboard_detector',         # 功能包名称
        executable='dynamic_detector_node', # 可执行文件名称
        name='dynamic_detector_node',       # 节点名称
        output='screen',                    # 输出到屏幕
        parameters=[dynamic_detector_param_path]  # 加载参数文件
    )

    # -------------------------------------------------------------------------
    # YOLO视觉检测节点 (YOLO Detector Node)
    # -------------------------------------------------------------------------
    # 主要功能:
    #   1. 使用 YOLO 神经网络从 RGB 图像检测目标
    #   2. 检测类别: 行人、车辆等
    #   3. 输出检测框和类别信息
    #   4. 可选: 与深度图像融合获取3D位置
    #
    # 输入话题:
    #   - /unitree_go2/front_cam/color_image (sensor_msgs/Image)
    #
    # 输出话题:
    #   - /yolo_detector/detections (vision_msgs/Detection2DArray)
    #
    # 注意: 该节点是可选的，用于增强检测能力
    # -------------------------------------------------------------------------
    yolo_detector_node = Node(
        package='onboard_detector',         # 功能包名称
        executable='yolo_detector_node.py', # Python可执行文件
        name='yolo_detector_node',          # 节点名称
        output='screen',                    # 输出到屏幕
        parameters=[yolo_detector_param_path]  # 加载参数文件
    )

    # =========================================================================
    # 启动描述
    # =========================================================================
    # 返回 LaunchDescription，包含所有要启动的节点
    # 启动顺序:
    #   1. occupancy_map_node (地图管理)
    #   2. dynamic_detector_node (动态检测)
    #   3. yolo_detector_node (视觉检测)
    #
    # 节点间依赖关系:
    #   - dynamic_detector_node 依赖 occupancy_map_node 的地图服务
    #   - yolo_detector_node 独立运行，提供额外的视觉检测结果
    # =========================================================================
    return LaunchDescription([
        occupancy_map_node,      # 首先启动地图管理节点
        dynamic_detector_node,   # 然后启动动态检测节点
        yolo_detector_node,      # 最后启动YOLO视觉检测节点
    ])
