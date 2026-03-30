# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unitree 四足机器人配置文件

本文件定义了 Unitree 机器人在 Isaac Sim 仿真环境中的完整配置。

可用配置:
* :obj:`UNITREE_A1_CFG`:  Unitree A1 机器人，使用直流电机模型 (DC Motor)
* :obj:`UNITREE_GO1_CFG`: Unitree Go1 机器人，使用神经网络执行器模型 (Actuator Net MLP)
* :obj:`UNITREE_GO2_CFG`: Unitree Go2 机器人，使用直流电机模型 (DC Motor)

参考文档: https://github.com/unitreerobotics/unitree_ros
"""

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.actuators import ActuatorNetMLPCfg, DCMotorCfg
from omni.isaac.orbit.assets.articulation import ArticulationCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

UNITREE_GO2_CFG = ArticulationCfg(
    # =========================================================================
    # 机器人模型加载配置
    # =========================================================================
    spawn=sim_utils.UsdFileCfg(
        usd_path=r"isaac-training\third_party\Go2\go2.usd",
        activate_contact_sensors=True,  # 启用接触传感器
        rigid_props=sim_utils.RigidBodyPropertiesCfg(   # 刚体属性
            disable_gravity=False,  
            retain_accelerations=False, 
            linear_damping=0.0, 
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg( # 关节属性
            enabled_self_collisions=False,  
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    
    # =========================================================================
    # 初始状态配置
    # =========================================================================
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),  # 初始高度 0.4m
        joint_pos={ # 初始关节角度
            ".*L_hip_joint": 0.1,   
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    
    # =========================================================================
    # 执行器配置 - 使用直流电机模型
    # =========================================================================
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            
            # Go2 电机规格
            effort_limit=23.5,       # 最大扭矩
            saturation_effort=23.5,  # 饱和扭矩
            velocity_limit=30.0,     # 最大角速度
            
            # PD 控制器参数
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""
Unitree Go2 机器人配置 - 使用直流电机模型

Go2 是 Unitree 的最新款四足机器人，具有更强的电机性能。
"""
