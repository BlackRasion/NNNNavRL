"""
NavRL 策略评估脚本

该脚本用于加载训练好的模型并进行评估，主要功能：
1. 加载预训练模型检查点
2. 运行策略评估（确定性策略）
3. 记录评估视频和性能指标到 WandB

与 train.py 的区别：
- 不执行训练更新（注释掉 policy.train）
- 每轮都进行评估（而非周期性）
- 从指定检查点加载模型
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
from go2_velocity_controller import Go2VelocityController, Go2VelController
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from torchrl.envs.transforms import TransformedEnv, Compose
from utils import evaluate
from torchrl.envs.utils import ExplorationType

# 配置文件路径
FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # =========================================================================
    # 步骤 1: 初始化 Isaac Sim 仿真应用
    # =========================================================================
    sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})

    # =========================================================================
    # 步骤 2: 初始化 WandB（用于记录评估结果）
    # =========================================================================
    if (cfg.wandb.run_id is None):
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=wandb.util.generate_id(),
        )
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
            entity=cfg.wandb.entity,
            config=cfg,
            mode=cfg.wandb.mode,
            id=cfg.wandb.run_id,
            resume="must"
        )

    # =========================================================================
    # 步骤 3: 创建导航环境
    # =========================================================================
    from env import NavigationEnv
    env = NavigationEnv(cfg)

    # =========================================================================
    # 步骤 4: 构建变换环境（与训练时相同）
    # =========================================================================
    transforms = []
    controller = Go2VelocityController(dt=cfg.sim.dt).to(cfg.device)
    vel_transform = Go2VelController(controller)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed)    

    # =========================================================================
    # 步骤 5: 初始化 PPO 策略并加载预训练模型
    # =========================================================================
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    # 注意：使用新的 Go2VelocityController 后，维度不兼容，需要重新训练
    # checkpoint = "/home/sia/whn_NavRL/NNNNavRL/isaac-training/wandb/offline-run-20260329_170703-38sxzjis/files/checkpoint_29000.pt"
    # python training/scripts/eval.py headless=True env.num_envs=1024 max_frame_num=1e6
    # policy.load_state_dict(torch.load(checkpoint))
    
    # =========================================================================
    # 步骤 6: 初始化统计收集器
    # =========================================================================
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)

    # =========================================================================
    # 步骤 7: 创建数据收集器
    # =========================================================================
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=cfg.device,
        return_same_td=True, # update the return tensordict inplace (should set to false if we need to use replace buffer)
        exploration_type=ExplorationType.RANDOM, # sample from normal distribution
    )
    # =========================================================================
    # 步骤 8: 评估循环（每轮都评估，不训练）
    # =========================================================================
    for i, data in enumerate(collector):
        # 记录基础信息
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # 注意：评估时不执行训练更新
        # train_loss_stats = policy.train(data)
        # info.update(train_loss_stats) # log training loss info

        # =========================================================================
        # 每轮都进行评估（使用确定性策略 MEAN）
        # =========================================================================
        print("[NavRL]: 开始评估策略，步骤: ", i)
        env.eval()
        eval_info = evaluate(
            env=transformed_env, 
            policy=policy,
            seed=cfg.seed, 
            cfg=cfg,
            exploration_type=ExplorationType.MEAN   # 确定性策略
        )
        env.train()
        env.reset()
        info.update(eval_info)
        print("\n[NavRL]: 评估完成")
        
        # 记录到 WandB
        run.log(info)

    # 结束评估
    wandb.finish()
    sim_app.close()

if __name__ == "__main__":
    main()
    