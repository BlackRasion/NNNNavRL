#!/usr/bin/env python3
# =============================================================================
# NavRL 多GPU训练脚本
# =============================================================================
# 该脚本支持使用多块 GPU 进行并行训练
# 使用方法:
#   python train_multi_gpu.py  # 使用默认配置
#   python train_multi_gpu.py env.num_envs=4096  # 命令行覆盖
# =============================================================================
# 多GPU训练原理:
#   - 环境并行: 每个 GPU 运行独立的仿真环境
#   - 数据收集: 每个 GPU 独立收集训练数据
#   - 梯度同步: 可选的梯度同步 (DDP)
#   - 日志记录: 仅主进程记录 WandB 日志
# =============================================================================

import argparse
import os
import sys
import hydra
import datetime
import wandb
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")


def find_free_port():
    """
    查找可用端口，避免多训练任务端口冲突
    
    返回:
        int: 可用的端口号
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed(rank, world_size, port, device_id):
    """
    初始化分布式训练环境
    
    参数:
        rank: 当前进程的排名 (0, 1, ..., world_size-1)
        world_size: 总进程数 (等于GPU数量)
        port: 通信端口
        device_id: 实际使用的 GPU 设备 ID
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # 设置可见设备，确保每个进程只看到自己的 GPU
    dist.init_process_group(
        backend='nccl',  # 使用 NCCL 后端 (NVIDIA GPU 优化)
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(device_id)


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def sync_model_parameters(model, device):
    """
    同步模型参数到所有进程 (用于禁用 DDP 时的参数同步)
    
    参数:
        model: PPO 模型
        device: 计算设备
    """
    for param in model.feature_extractor.parameters():
        dist.broadcast(param.data, src=0)
    for param in model.actor.parameters():
        dist.broadcast(param.data, src=0)
    for param in model.critic.parameters():
        dist.broadcast(param.data, src=0)
    # 同步 ValueNorm 状态
    dist.broadcast(model.value_norm.mean.data, src=0)
    dist.broadcast(model.value_norm.var.data, src=0)
    dist.broadcast(model.value_norm.count.data, src=0)


def sync_value_norm(policy, device):
    """
    同步 ValueNorm 的统计信息 (running mean/var)
    
    参数:
        policy: PPO 策略
        device: 计算设备
    """
    # 聚合所有 GPU 的 ValueNorm 统计
    mean_tensor = policy.value_norm.mean.clone()
    var_tensor = policy.value_norm.var.clone()
    count_tensor = policy.value_norm.count.clone()
    
    dist.all_reduce(mean_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(var_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    
    # 计算平均值
    world_size = dist.get_world_size()
    policy.value_norm.mean.copy_(mean_tensor / world_size)
    policy.value_norm.var.copy_(var_tensor / world_size)
    policy.value_norm.count.copy_(count_tensor / world_size)


def train_worker(rank, world_size, cfg, port):
    """
    单个 GPU 上的训练工作进程
    
    参数:
        rank: 当前进程的排名
        world_size: 总进程数
        cfg: 配置对象
        port: 分布式通信端口
    """
    # =========================================================================
    # GPU 设备映射
    # =========================================================================
    # 从配置中获取实际 GPU 设备列表
    devices = cfg.multi_gpu.devices if hasattr(cfg, 'multi_gpu') else list(range(world_size))
    device_id = devices[rank]  # 获取当前进程对应的实际 GPU ID
    device = f"cuda:{device_id}"
    
    is_main_process = (rank == 0)
    
    # =========================================================================
    # 初始化分布式环境
    # =========================================================================
    setup_distributed(rank, world_size, port, device_id)
    
    cfg.device = device
    
    # =========================================================================
    # 初始化 Simulation App (每个进程独立)
    # =========================================================================
    try:
        sim_app = SimulationApp({"headless": cfg.headless, "anti_aliasing": 1})
    except Exception as e:
        print(f"[Rank {rank}] 初始化 SimulationApp 失败: {e}")
        cleanup_distributed()
        sys.exit(1)
    
    # =========================================================================
    # WandB 初始化 (仅主进程)
    # =========================================================================
    run = None
    if is_main_process:
        try:
            if cfg.wandb.run_id is None:
                run = wandb.init( 
                    project=cfg.wandb.project,
                    name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
                    entity=cfg.wandb.entity,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    mode=cfg.wandb.mode,
                    id=wandb.util.generate_id(),
                )
            else:
                run = wandb.init(
                    project=cfg.wandb.project,
                    name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
                    entity=cfg.wandb.entity,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    mode=cfg.wandb.mode,
                    id=cfg.wandb.run_id,
                    resume="must"
                )
        except Exception as e:
            print(f"[Rank {rank}] WandB 初始化失败: {e}")
    
    # =========================================================================
    # 计算每个 GPU 的环境数量
    # =========================================================================
    total_envs = cfg.env.num_envs
    envs_per_gpu = total_envs // world_size
    cfg.env.num_envs = envs_per_gpu
    
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"[NavRL] 多GPU训练配置:")
        print(f"{'='*60}")
        print(f"  - GPU 数量: {world_size}")
        print(f"  - GPU 设备映射: rank -> device")
        for r in range(world_size):
            print(f"      rank {r} -> cuda:{devices[r]}")
        print(f"  - 总环境数: {total_envs}")
        print(f"  - 每GPU环境数: {envs_per_gpu}")
        print(f"  - 通信端口: {port}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # 创建训练环境
    # =========================================================================
    try:
        from env import NavigationEnv
        env = NavigationEnv(cfg)
    except Exception as e:
        print(f"[Rank {rank}] 创建环境失败: {e}")
        cleanup_distributed()
        sim_app.close()
        sys.exit(1)
    
    # =========================================================================
    # 环境变换
    # =========================================================================
    transforms = []
    controller = LeePositionController(9.81, env.drone.params).to(device)
    vel_transform = VelController(controller, yaw_control=False)
    transforms.append(vel_transform)
    transformed_env = TransformedEnv(env, Compose(*transforms)).train()
    transformed_env.set_seed(cfg.seed + rank)
    
    # =========================================================================
    # 创建 PPO 策略
    # =========================================================================
    use_ddp = cfg.multi_gpu.get('use_ddp', True) if hasattr(cfg, 'multi_gpu') else False
    
    try:
        policy = PPO(
            cfg.algo,
            transformed_env.observation_spec,
            transformed_env.action_spec,
            device,
            enable_ddp=use_ddp
        )
    except Exception as e:
        print(f"[Rank {rank}] 创建 PPO 策略失败: {e}")
        cleanup_distributed()
        sim_app.close()
        sys.exit(1)
    
    if is_main_process:
        print(f"[NavRL] 策略配置:")
        print(f"  - DDP 梯度同步: {'启用' if use_ddp else '禁用'}")
        if not use_ddp:
            print(f"  - 参数同步间隔: 每 {cfg.eval_interval} 步")

    # -------------------------------------------------------------------------
    # 可选：从检查点加载预训练模型
    # -------------------------------------------------------------------------
    # checkpoint = "/path/to/checkpoint_2500.pt"
    # policy.load_state_dict(torch.load(checkpoint, map_location=device))
    
    # =========================================================================
    # Episode Stats 收集器
    # =========================================================================
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(episode_stats_keys)
    
    # =========================================================================
    # 数据收集器
    # =========================================================================
    collector = SyncDataCollector(
        transformed_env,
        policy=policy, 
        frames_per_batch=envs_per_gpu * cfg.algo.training_frame_num, 
        total_frames=cfg.max_frame_num,
        device=device,
        return_same_td=True,
        exploration_type=ExplorationType.RANDOM,
    )
    
    # =========================================================================
    # 训练循环
    # =========================================================================
    try:
        for i, data in enumerate(collector):
            # 训练策略
            train_loss_stats = policy.train(data)
            
            # 收集 episode 统计信息
            episode_stats.add(data)
            
            # =====================================================================
            # 同步训练损失统计 (所有GPU)
            # =====================================================================
            loss_keys = list(train_loss_stats.keys())
            loss_stats_tensor = torch.tensor(
                [train_loss_stats.get(k, 0.0) for k in loss_keys],
                device=device
            )
            dist.all_reduce(loss_stats_tensor, op=dist.ReduceOp.SUM)
            loss_stats_tensor /= world_size
            
            # =====================================================================
            # 同步 ValueNorm 统计信息
            # =====================================================================
            # ValueNorm 的 running mean/var 需要保持一致
            sync_value_norm(policy, device)
            
            # =====================================================================
            # Episode 统计处理
            # =====================================================================
            has_episode_stats = torch.tensor(
                [1.0 if len(episode_stats) >= transformed_env.num_envs else 0.0],
                device=device
            )
            dist.all_reduce(has_episode_stats, op=dist.ReduceOp.SUM)
            
            episode_stats_dict = {}
            if len(episode_stats) >= transformed_env.num_envs:
                episode_stats_dict = {
                    k: v for k, v in episode_stats.pop().items(True, True)
                }
            
            dist.barrier()
            
            # =====================================================================
            # 主进程日志记录
            # =====================================================================
            if is_main_process:
                info = {
                    "env_frames": collector._frames * world_size,
                    "rollout_fps": collector._fps * world_size,
                }
                
                for idx, k in enumerate(loss_keys):
                    info[k] = loss_stats_tensor[idx].item()
                
                if len(episode_stats) >= transformed_env.num_envs:
                    stats = {
                        "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                        for k, v in episode_stats_dict.items()
                    }
                    info.update(stats)
            
            # =====================================================================
            # 评估阶段 (所有GPU同步)
            # =====================================================================
            if i % cfg.eval_interval == 0:
                dist.barrier()
                
                # 禁用 DDP 时，定期同步模型参数
                if not use_ddp:
                    sync_model_parameters(policy, device)
                
                if is_main_process:
                    print(f"[NavRL]: 开始评估策略，训练步数: {i}")
                    env.enable_render(True)
                    env.eval()
                    eval_info = evaluate(
                        env=transformed_env, 
                        policy=policy,
                        seed=cfg.seed, 
                        cfg=cfg,
                        exploration_type=ExplorationType.MEAN
                    )
                    env.enable_render(not cfg.headless)
                    env.train()
                    env.reset()
                    if 'info' not in dir():
                        info = {}
                    info.update(eval_info)
                    print("\n[NavRL]: 评估完成")
                
                dist.barrier()
            
            # =====================================================================
            # 日志记录和模型保存 (仅主进程)
            # =====================================================================
            if is_main_process:
                run.log(info)
                
                if i % cfg.save_interval == 0:
                    ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
                    torch.save(policy.state_dict(), ckpt_path)
                    print(f"[NavRL]: 模型已保存，训练步数: {i}")
    
    except Exception as e:
        print(f"[Rank {rank}] 训练过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # =====================================================================
        # 清理资源
        # =====================================================================
        if is_main_process and run is not None:
            ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
            torch.save(policy.state_dict(), ckpt_path)
            wandb.finish()
        
        cleanup_distributed()
        sim_app.close()


# =============================================================================
# Hydra 配置装饰器
# =============================================================================
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # =========================================================================
    # 检查是否启用多GPU训练
    # =========================================================================
    if hasattr(cfg, 'multi_gpu') and cfg.multi_gpu.enabled:
        world_size = cfg.multi_gpu.num_gpus
        devices = cfg.multi_gpu.devices
        
        # 验证 GPU 可用性
        available_gpus = torch.cuda.device_count()
        if available_gpus < world_size:
            print(f"[错误] 请求 {world_size} 个 GPU，但只有 {available_gpus} 个可用")
            return
        
        # 验证设备列表
        for dev_id in devices:
            if dev_id >= available_gpus:
                print(f"[错误] GPU {dev_id} 不存在，可用 GPU: 0-{available_gpus-1}")
                return
        
        # 查找可用端口
        port = find_free_port()
        
        print(f"\n{'='*60}")
        print(f"[NavRL] 启动多GPU训练")
        print(f"{'='*60}")
        print(f"  - GPU 设备: {devices}")
        print(f"  - 进程数: {world_size}")
        print(f"  - 训练策略: {cfg.multi_gpu.strategy}")
        print(f"  - DDP 梯度同步: {cfg.multi_gpu.get('use_ddp', True)}")
        print(f"{'='*60}\n")
        
        # 启动多进程训练
        try:
            mp.spawn(
                train_worker,
                args=(world_size, cfg, port),
                nprocs=world_size,
                join=True
            )
        except KeyboardInterrupt:
            print("\n[NavRL] 训练被用户中断")
        except Exception as e:
            print(f"\n[NavRL] 训练出错: {e}")
    else:
        print(f"[NavRL] 未启用多GPU训练，请检查配置文件中的 multi_gpu.enabled 设置")


if __name__ == "__main__":
    main()
