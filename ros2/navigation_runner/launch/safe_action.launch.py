# =============================================================================
# Safe Action Launch File 安全动作模块启动文件
# =============================================================================
# 该文件用于启动导航系统的安全动作模块 (ORCA安全控制器)
# 启动命令: ros2 launch navigation_runner safe_action.launch.py
# =============================================================================
# 作用流程:
# 1. 加载 safe_action_param.yaml 参数文件
# 2. 启动 safe_action_node (安全动作节点)
# 3. 该节点提供 ORCA (Optimal Reciprocal Collision Avoidance) 安全控制服务
# 4. 确保导航过程中的碰撞避免，作为策略输出的安全保护层
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
    
    # 安全动作参数文件路径
    # 包含: ORCA算法参数 (时间窗口、时间步长、安全距离、高度约束)
    safe_action_param_path = os.path.join(
        get_package_share_directory('navigation_runner'),
        'cfg',
        'safe_action_param.yaml'
    )

    # =========================================================================
    # 安全动作节点定义
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 安全动作节点 (Safe Action Node)
    # -------------------------------------------------------------------------
    # 主要功能:
    #   1. 实现 ORCA (Optimal Reciprocal Collision Avoidance) 算法
    #   2. 接收原始速度指令，计算安全修正后的速度
    #   3. 确保无人机与障碍物保持安全距离
    #   4. 限制飞行高度在安全范围内
    #   5. 作为导航系统的安全保护层
    #
    # 算法原理:
    #   ORCA 是一种分布式避障算法，通过计算速度障碍 (Velocity Obstacle)
    #   来确保多个动态智能体之间不会发生碰撞。每个智能体根据其他智能体的状态独立计算安全的速度选择。
    # 输入:
    #   - 原始速度指令 (来自 PPO 策略网络)
    #   - 当前位置和速度
    #   - 障碍物信息 (静态和动态)
    #   - 目标位置
    #
    # 输出:
    #   - 安全修正后的速度指令
    #
    # 提供服务:
    #   - /safe_action/get_safe_action
    #     请求: 原始速度、当前状态、障碍物列表
    #     响应: 安全速度
    #
    # 使用场景:
    #   1. 单独测试 ORCA 避障算法
    #   2. 作为导航系统的安全保护层
    #   3. 与其他避障算法进行对比测试
    # -------------------------------------------------------------------------
    safe_action_node = Node(
        package='navigation_runner',        # 功能包名称
        executable='safe_action_node',      # 可执行文件名称
        name='safe_action_node',            # 节点名称
        output='screen',                    # 输出到屏幕
        parameters=[safe_action_param_path] # 加载参数文件
    )

    # =========================================================================
    # 启动描述
    # =========================================================================
    # 返回 LaunchDescription，包含 safe_action_node
    # 该节点通常与 navigation_node 配合使用
    # 启动顺序建议:
    #   1. 先启动感知模块 (perception.launch.py)
    #   2. 然后启动安全动作模块 (safe_action.launch.py)
    #   3. 最后启动导航主模块 (navigation.launch.py)
    # =========================================================================
    return LaunchDescription([
        safe_action_node,      # 启动 ORCA 安全动作节点
    ])
