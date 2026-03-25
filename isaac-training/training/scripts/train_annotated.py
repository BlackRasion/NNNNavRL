"""
NavRL 训练主程序

该脚本实现了 NavRL 框架的完整训练流程，包括：
1. Isaac Sim 仿真环境初始化
2. WandB 训练监控
3. 导航环境构建
4. PPO 策略训练
5. 周期性评估和模型保存
"""

import argparse
import os
import hydra
import datetime
import wandb
import torch
from omegaconf import DictConfig, OmegaConf
from omni.isaac.kit import SimulationApp
from ppo import PPO
from omni_drones.controllers import LeePositionController
from omni_drones.utils.torchrl.transforms import VelController, ravel_composite
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType


# =============================================================================
# 配置文件路径设置
# =============================================================================
# 获取配置文件目录路径（相对于当前脚本的父目录的 cfg 文件夹）
FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


# =============================================================================
# Hydra 配置装饰器
# =============================================================================
# @hydra.main 装饰器自动加载配置文件
# config_name: 主配置文件名（train.yaml）
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    """
    主训练函数
    """
    
    # =========================================================================
    # 步骤 1: 初始化
    # =========================================================================
    # SimulationApp 是 Isaac Sim 的入口点，负责启动仿真引擎
    # headless: 是否无头模式运行（不显示 GUI，用于服务器训练）
    # anti_aliasing: 抗锯齿级别，设置为 1 提高渲染质量
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    
    # WandB (Weights & Biases) 用于实时监控训练过程、记录指标和可视化
    # 支持断点续训：如果提供了 run_id，则恢复之前的训练
    if (cfg.wandb.run_id is None):
        run = wandb.init( # 情况 A: 开始新的训练运行
            project=cfg.wandb.project,           # WandB 项目名称
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",  # 运行名称（带时间戳）
            entity=cfg.wandb.entity,             # WandB 团队/组织名称
            config=cfg,                          # 记录完整配置
            mode=cfg.wandb.mode,                 # 模式: online/offline/disabled
            id=wandb.util.generate_id(),         # 生成唯一运行 ID
        )
    else:
        run = wandb.init( # 情况 B: 恢复之前的训练（断点续训）
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,                 # 使用已有的运行 ID
            resume="must"                        # 强制恢复模式
        )

    # =========================================================================
    # 步骤 2: 构建导航训练环境
    # =========================================================================
    # 负责管理仿真场景、障碍物、无人机状态等
    from env import NavigationEnv # NavigationEnv 是自定义的强化学习环境，继承自 IsaacEnv
    env = NavigationEnv(cfg)

    # 构建环境变换 TransformedEnv 允许在原始环境上叠加多个变换层
    # 这里主要添加速度控制器，将策略输出的速度指令转换为电机控制信号
    
    transforms = []
    
    # LeePositionController: 基于 Lee 等人论文的位置控制器
    controller = LeePositionController(9.81, env.drone.params).to(cfg.device) # 9.81: 重力加速度 env.drone.params: 无人机物理参数（质量、惯性等）
    
    # VelController: 速度控制器变换
    vel_transform = VelController(controller, yaw_control=False) # yaw_control=False: 不控制偏航角，只控制位置速度
    transforms.append(vel_transform)
    
    # 应用所有变换，创建训练环境
    transformed_env = TransformedEnv(env, Compose(*transforms)).train() # .train() 设置环境为训练模式
    transformed_env.set_seed(cfg.seed)  # 设置随机种子保证可复现性
    
    # =========================================================================
    # 步骤 3: 初始化 PPO 策略网络
    # =========================================================================    
    policy = PPO(
        cfg.algo,                               # 算法配置（学习率、网络结构等）
        transformed_env.observation_spec,       # 观测空间规格
        transformed_env.action_spec,            # 动作空间规格
        cfg.device                              # 计算设备（cuda/cpu）
    )

    # -------------------------------------------------------------------------
    # 可选：从检查点加载预训练模型（用于微调或继续训练）
    # -------------------------------------------------------------------------
    # checkpoint = "/path/to/checkpoint_2500.pt"
    # policy.load_state_dict(torch.load(checkpoint))
    
    # =========================================================================
    # 步骤 4: 初始化回合统计收集器 和 数据收集器
    # =========================================================================
    # EpisodeStats 用于收集和统计每个训练回合的指标，如：回合长度、总回报、是否到达目标等
    episode_stats_keys = [     # 从观测规格中提取所有以 "stats" 开头的键
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # SyncDataCollector 是 TorchRL 提供的数据收集器，负责并行收集环境交互数据，供策略训练使用
    collector = SyncDataCollector(
        transformed_env,                        # 训练环境
        policy=policy,                          # 策略网络
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,  # 每批数据量
        total_frames=cfg.max_frame_num,         # 总训练帧数
        device=cfg.device,                      # 计算设备
        return_same_td=True,                    # 原地更新 TensorDict（节省内存）
        exploration_type=ExplorationType.RANDOM, # 探索策略：随机采样
    )

    # =========================================================================
    # 步骤 5: 主训练循环
    # =========================================================================
    # 不断重复：1. 收集数据 → 2. 更新策略 → 3. 评估 → 4. 记录日志 → 5. 保存模型
    
    for i, data in enumerate(collector):
        # ---------------------------------------------------------------------
        # 5.1 记录基础信息
        # ---------------------------------------------------------------------
        # env_frames: 已收集的总帧数
        # rollout_fps: 数据收集速度（帧/秒）
        info = {
            "env_frames": collector._frames, 
            "rollout_fps": collector._fps
        }

        # ---------------------------------------------------------------------
        # 5.2 训练策略
        # ---------------------------------------------------------------------
        # policy.train() 执行 PPO 更新：
        # - 计算 GAE（广义优势估计）
        # - 更新 Actor 网络（策略）
        # - 更新 Critic 网络（价值函数）
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats)  # 合并训练损失信息

        # ---------------------------------------------------------------------
        # 5.3 计算和记录训练回合统计
        # ---------------------------------------------------------------------
        episode_stats.add(data)
        
        # 当所有并行环境都完成至少一个回合时，计算统计信息
        if len(episode_stats) >= transformed_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # ---------------------------------------------------------------------
        # 5.4 周期性评估策略
        # ---------------------------------------------------------------------
        # 每隔 eval_interval 个训练步，运行一次完整评估
        # 评估时使用确定性策略（MEAN 探索模式），不添加随机噪声
        
        if i % cfg.eval_interval == 0:
            print("[NavRL]: start evaluating policy at training step: ", i)
            
            # 启用渲染以记录评估视频
            env.enable_render(True)
            env.eval()  # 切换到评估模式
            
            # 运行评估
            eval_info = evaluate(
                env=transformed_env, 
                policy=policy,
                seed=cfg.seed, 
                cfg=cfg,
                exploration_type=ExplorationType.MEAN  # 使用均值策略（确定性）
            )
            
            # 恢复训练设置
            env.enable_render(not cfg.headless)
            env.train()  # 切换回训练模式
            env.reset()  # 重置环境
            
            # 合并评估信息
            info.update(eval_info)
            print("\n[NavRL]: evaluation done.")
        
        # ---------------------------------------------------------------------
        # 5.5 更新 WandB 日志
        # ---------------------------------------------------------------------
        # 将所有统计信息发送到 WandB 服务器进行可视化
        run.log(info)

        # ---------------------------------------------------------------------
        # 5.6 周期性保存模型检查点
        # ---------------------------------------------------------------------
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: model saved at training step: ", i)

    # =========================================================================
    # 步骤 6: 训练结束，保存最终模型
    # =========================================================================
    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    
    # 关闭 WandB 和仿真应用
    wandb.finish()
    sim_app.close()


# =============================================================================
# 程序入口
# =============================================================================
if __name__ == "__main__":
    main()
