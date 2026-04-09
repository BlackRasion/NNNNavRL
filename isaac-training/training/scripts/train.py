import os
import hydra
import datetime
import wandb
import torch
from omni.isaac.kit import SimulationApp
from ppo import PPO
from go2_velocity_controller import Go2VelocityController, Go2VelController
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType

# =============================================================================
# 配置文件路径设置
# =============================================================================
# 获取配置文件目录路径（相对于当前脚本的父目录的 cfg 文件夹）
FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


# @hydra.main 自动加载配置文件
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # =========================================================================
    # 步骤 1: 初始化
    # =========================================================================
    # Simulation App，启动仿真引擎
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # Wandb 实时监控训练过程、记录指标和可视化
    if cfg.wandb.run_id is None:
        run = wandb.init(  # 情况 A: 开始新的训练运行
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(  # 情况 B: 恢复之前的训练（断点续训
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,  # 使用已有的运行 ID
            resume="must",
        )
    # =========================================================================
    # 步骤 2: 构建导航训练环境
    # =========================================================================
    from env import NavigationEnv

    env = NavigationEnv(cfg)
    # 构建环境变换 TransformedEnv 允许在原始环境上叠加多个变换层,这里主要添加速度控制器
    transforms = []
    controller = Go2VelocityController(dt=cfg.sim.dt).to(cfg.device)
    vel_transform = Go2VelController(controller)
    transforms.append(vel_transform)
    # 应用所有变换，创建训练环境
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)

    # =========================================================================
    # 步骤 3: 初始化 PPO 策略网络
    # =========================================================================
    policy = PPO(
        cfg.algo,  # 算法配置（学习率、网络结构等）
        transformed_env.observation_spec,  # 观测空间规格
        transformed_env.action_spec,  # 动作空间规格
        cfg.device,  # 计算设备（cuda/cpu）
    )

    # -------------------------------------------------------------------------
    # 可选：从检查点加载预训练模型（用于微调或继续训练）
    # -------------------------------------------------------------------------
    # checkpoint = "/home/sia/whn_NavRL/NNNNavRL/isaac-training/wandb/offline-run-20260407_183000-gjm4m08p/files/checkpoint_final.pt"
    # policy.load_state_dict(torch.load(checkpoint))

    # =========================================================================
    # 步骤 4: 初始化回合统计收集器 和 数据收集器
    # =========================================================================
    # EpisodeStats 用于收集和统计每个训练回合的指标，如：回合长度、总回报、是否到达目标等
    episode_stats_keys = [
        k
        for k in transformed_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # SyncDataCollector负责并行收集环境交互数据，供策略训练使用
    collector = SyncDataCollector(
        transformed_env,  # 训练环境
        policy=policy,  # 策略网络
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,  # 每批数据量
        total_frames=cfg.max_frame_num,  # 总训练帧数
        device=cfg.device,  # 计算设备
        return_same_td=True,  # 原地更新 TensorDict（节省内存）
        exploration_type=ExplorationType.RANDOM,  # 探索策略：随机采样
    )
    # =========================================================================
    # 步骤 5: 主训练循环
    # =========================================================================
    for i, data in enumerate(collector):
        info = {
            "env_frames": collector._frames,  # env_frames: 已收集的总帧数
            "rollout_fps": collector._fps,  # rollout_fps: 数据收集速度（帧/秒）
        }

        # 训练策略
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats)  # 合并训练损失信息

        # 计算和记录训练回合统计信息
        episode_stats.add(data)
        if (
            len(episode_stats) >= transformed_env.num_envs
        ):  # 当所有并行无人机都完成至少一个回合时，计算统计信息
            stats = {
                "train/"
                + (".".join(k) if isinstance(k, tuple) else k): torch.mean(
                    v.float()
                ).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        # 周期性评估策略
        if i % cfg.eval_interval == 0:  # 每 cfg.eval_interval 步评估一次
            print("[NavRL]: 开始评估策略，训练步数: ", i)
            env.enable_render(True)
            env.eval()
            eval_info = evaluate(
                env=transformed_env,
                policy=policy,
                seed=cfg.seed,
                cfg=cfg,
                exploration_type=ExplorationType.MEAN,  # 评估时使用确定性策略MEAN，不添加随机噪声
            )
            env.enable_render(not cfg.headless)
            env.train()
            env.reset()
            info.update(eval_info)
            print("\n[NavRL]: 评估策略完成，训练步数: ", i)

        # 更新 WandB 日志，发送所有统计信息
        run.log(info)

        # 周期性保存模型检查点
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[NavRL]: 模型已保存，训练步数: ", i)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    # 关闭 WandB 和仿真应用
    wandb.finish()
    sim_app.close()


if __name__ == "__main__":
    main()
