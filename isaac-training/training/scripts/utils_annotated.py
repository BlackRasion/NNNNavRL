"""
NavRL 工具函数模块

该文件包含 PPO 训练和评估所需的各种辅助类和函数：
1. ValueNorm: 价值函数归一化（PopArt）
2. 分布类: IndependentNormal, IndependentBeta
3. Actor 网络: Actor, BetaActor
4. GAE: 广义优势估计
5. 评估函数: evaluate
6. 坐标变换: vec_to_new_frame, vec_to_world
"""

import torch
import torch.nn as nn
import wandb
import numpy as np
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
from omni_drones.utils.torchrl import RenderCallback
from torchrl.envs.utils import ExplorationType, set_exploration_type


# =============================================================================
# ValueNorm: 价值函数归一化 (PopArt)
# =============================================================================
# PopArt (Preserving Outputs Precisely while Adaptively Rescaling Targets)
# 用于稳定价值函数学习，通过维护运行均值和方差来归一化目标值
# 参考: https://arxiv.org/abs/1602.07714

class ValueNorm(nn.Module):
    """
    价值归一化模块
    
    使用指数移动平均维护回报的均值和方差，用于：
    1. 归一化 Critic 的学习目标（回报）
    2. 反归一化 Critic 的输出以获取真实价值估计
    
    这有助于处理不同训练阶段回报尺度变化的问题
    """
    
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,           # 指数移动平均衰减系数，接近1表示更平滑的更新
        epsilon=1e-5,         # 数值稳定性常数，防止除零
    ) -> None:
        super().__init__()

        # 确定输入形状
        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        # 注册缓冲区（不参与梯度计算，但会随模型保存）
        # 运行均值: 维护回报的指数移动平均
        self.running_mean: torch.Tensor
        # 运行均方: 用于计算方差
        self.running_mean_sq: torch.Tensor
        # 去偏项: 用于修正指数移动平均的偏差
        self.debiasing_term: torch.Tensor
        
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        """重置所有统计量为零"""
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        """
        计算去偏后的均值和方差
        
        由于使用指数移动平均，早期估计会有偏差
        通过除以 debiasing_term 来修正
        
        返回:
            debiased_mean: 去偏均值
            debiased_var: 去偏方差（最小值为 1e-2）
        """
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        # 方差 = E[X^2] - E[X]^2，并限制最小值防止数值问题
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        """
        使用新数据更新运行统计量
        
        参数:
            input_vector: 新的回报数据，形状应与 input_shape 匹配
        """
        # 验证输入形状
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        
        # 计算需要归约的维度（除最后 input_shape 维度外的所有维度）
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        
        # 计算当前批次的均值和均方
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        # 指数移动平均更新
        # new_mean = beta * old_mean + (1-beta) * batch_mean
        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        """
        归一化输入向量
        
        公式: (x - mean) / sqrt(var)
        
        参数:
            input_vector: 待归一化的数据
            
        返回:
            归一化后的数据，均值为0，标准差为1
        """
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        """
        反归一化输入向量
        
        公式: x * sqrt(var) + mean
        
        参数:
            input_vector: 归一化的数据
            
        返回:
            反归一化后的原始尺度数据
        """
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out


# =============================================================================
# MLP 构建函数
# =============================================================================

def make_mlp(num_units):
    """
    构建多层感知机 (MLP)
    
    每层包含: Linear -> LeakyReLU -> LayerNorm
    
    参数:
        num_units: 每层神经元数量的列表，如 [256, 128, 64]
        
    返回:
        nn.Sequential: 构建好的 MLP 网络
    """
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))   # 延迟初始化线性层
        layers.append(nn.LeakyReLU())      # LeakyReLU 激活（允许负值的小梯度）
        layers.append(nn.LayerNorm(n))     # 层归一化，稳定训练
    return nn.Sequential(*layers)


# =============================================================================
# 概率分布类
# =============================================================================
class IndependentBeta(torch.distributions.Independent):
    """
    独立 Beta 分布，适合有界动作空间 (0, 1)
    Beta 分布通过 alpha 和 beta 两个形状参数控制分布形态
    
    优点:
    - 天然支持有界输出，无需额外裁剪
    - 可以表示各种分布形状（对称、偏斜、U型等）
    """
    # 参数约束: alpha 和 beta 都必须为正数
    arg_constraints = {
        "alpha": torch.distributions.constraints.positive, 
        "beta": torch.distributions.constraints.positive
    }

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)


# =============================================================================
# Actor 网络类
# =============================================================================

class Actor(nn.Module):
    """
    高斯策略 Actor（备用实现）
    
    输出正态分布的均值 (loc) 和标准差 (scale)
    注意: NavRL 实际使用 BetaActor，此类可能用于对比实验
    """
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        # 均值网络: 特征 -> 动作维度
        self.actor_mean = nn.LazyLinear(action_dim)
        # 对数标准差: 可学习参数，所有状态共享
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        # 预测均值
        loc = self.actor_mean(features)
        # 将对数标准差转换为标准差，并扩展到与均值相同形状
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale


class BetaActor(nn.Module):
    """
    Beta 分布策略 Actor（NavRL 使用）
    
    输出 Beta 分布的 alpha 和 beta 参数
    通过 Softplus 激活确保参数为正
    
    Beta 分布特性:
    - alpha > 1, beta > 1: 单峰分布（类似高斯）
    - alpha < 1, beta < 1: U型分布（倾向于边界）
    - alpha = beta: 对称分布
    - alpha > beta: 右偏（倾向于1）
    - alpha < beta: 左偏（倾向于0）
    """
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        # alpha 参数网络
        self.alpha_layer = nn.LazyLinear(action_dim)
        # beta 参数网络
        self.beta_layer = nn.LazyLinear(action_dim)
        # Softplus 激活: smooth ReLU，输出始终为正
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        """
        前向传播计算 Beta 分布参数
        
        参数:
            features: 输入特征 [batch_size, feature_dim]
            
        返回:
            alpha: Beta 分布 alpha 参数 [batch_size, action_dim]
            beta: Beta 分布 beta 参数 [batch_size, action_dim]
        """
        # 计算 alpha: 1 + softplus(output) + epsilon
        # 加 1 确保 alpha > 1，使分布单峰
        # 加 epsilon 防止数值问题
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta


# =============================================================================
# GAE: 广义优势估计
# =============================================================================

class GAE(nn.Module):
    """
    Generalized Advantage Estimation (GAE)
    
    用于计算优势函数的估计，平衡偏差和方差
    
    参数:
        gamma: 折扣因子，控制未来奖励的重要性
        lmbda: GAE 参数，控制偏差-方差权衡
    
    参考: https://arxiv.org/abs/1506.02438
    """
    def __init__(self, gamma, lmbda):
        super().__init__()
        # 注册为 buffer，使其随模型保存但不参与梯度
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor,         # 奖励 [batch, num_steps]
        terminated: torch.Tensor,     # 终止标志 [batch, num_steps]
        value: torch.Tensor,          # 当前状态价值 [batch, num_steps]
        next_value: torch.Tensor      # 下一状态价值 [batch, num_steps]
    ):
        """
        计算 GAE 优势估计和回报
        
        算法:
        1. 从后向前遍历时间步
        2. 计算 TD 残差: delta = r + γ*V(s')*(1-done) - V(s)
        3. 累积优势: A = delta + γ*λ*(1-done)*A_next
        4. 回报 = 优势 + 价值
        
        返回:
            advantages: 优势估计
            returns: 回报估计（用于 Critic 训练）
        """
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        # 未终止标志 (1 = 未终止，0 = 终止)
        not_done = 1 - terminated.float()
        gae = 0  # 累积优势
        
        # 反向遍历时间步（从最后一步到第一步）
        for step in reversed(range(num_steps)):
            # TD 残差: r_t + γ*V(s_{t+1})*(1-done) - V(s_t)
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            # GAE 累积: A_t = delta_t + γ*λ*(1-done)*A_{t+1}
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        
        # 回报 = 优势 + 价值（用于 Critic 的目标）
        returns = advantages + value
        return advantages, returns


# =============================================================================
# 批次生成函数
# =============================================================================

def make_batch(tensordict: TensorDict, num_minibatches: int):
    """
    将数据分割为多个 minibatch
    
    用于 PPO 的多 epoch 训练，每个 epoch 将数据打乱并分割
    
    参数:
        tensordict: 训练数据，形状 [batch_size, ...]
        num_minibatches: minibatch 数量
        
    返回:
        生成器，每次 yield 一个 minibatch
    """
    # 展平前两个维度 [num_envs, num_frames, ...] -> [num_envs*num_frames, ...]
    tensordict = tensordict.reshape(-1) 
    
    # 计算可整除的总样本数
    total_samples = (tensordict.shape[0] // num_minibatches) * num_minibatches
    
    # 生成随机排列并 reshape 为 [num_minibatches, samples_per_batch]
    perm = torch.randperm(
        total_samples,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    
    # 按索引提取 minibatch
    for indices in perm:
        yield tensordict[indices]


# =============================================================================
# 评估函数
# =============================================================================

@torch.no_grad()
def evaluate(
    env,
    policy,
    cfg,
    seed: int=0, 
    exploration_type: ExplorationType=ExplorationType.MEAN
):
    """
    评估训练好的策略
    
    在评估模式下运行策略，记录性能指标和渲染视频
    
    参数:
        env: 评估环境
        policy: 策略网络
        cfg: 配置对象
        seed: 随机种子
        exploration_type: 探索类型
            - MEAN: 使用均值策略（确定性，用于评估）
            - RANDOM: 随机采样（用于训练探索）
    
    返回:
        info: 包含评估统计和视频的字典
    """
    # 启用渲染以记录视频
    env.enable_render(True)
    # 设置评估模式（禁用 dropout 等）
    env.eval()
    # 设置随机种子保证可复现
    env.set_seed(seed)

    # 渲染回调: 定期捕获帧用于视频生成
    render_callback = RenderCallback(interval=2)
    
    # 设置探索类型并执行 rollout
    with set_exploration_type(exploration_type):
        trajs = env.rollout(
            max_steps=env.max_episode_length,  # 最大回合长度
            policy=policy,                      # 评估策略
            callback=render_callback,           # 渲染回调
            auto_reset=True,                    # 回合结束自动重置
            break_when_any_done=False,          # 等待所有环境完成
            return_contiguous=False,            # 允许非连续内存
        )
    
    # 恢复渲染设置
    env.enable_render(not cfg.headless)
    # 重置环境
    env.reset()
    
    # 提取每个轨迹的首次完成索引
    done = trajs.get(("next", "done")) 
    # argmax 找到第一个 done=True 的位置
    first_done = torch.argmax(done.long(), dim=1).cpu()

    def take_first_episode(tensor: torch.Tensor):
        """提取每个轨迹第一次完成时的数据"""
        # 调整索引形状以匹配张量维度
        indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
        return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

    # 提取统计信息
    traj_stats = {
        k: take_first_episode(v)
        for k, v in trajs[("next", "stats")].cpu().items()
    }

    # 构建评估信息字典
    info = {
        "eval/stats." + k: torch.mean(v.float()).item() 
        for k, v in traj_stats.items()
    }

    # 添加视频到日志
    info["recording"] = wandb.Video(
        render_callback.get_video_array(axes="t c h w"),  # 视频数组
        fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),        # 帧率计算
        format="mp4"
    )
    
    # 恢复训练模式
    env.train()
    
    return info


# =============================================================================
# 坐标变换函数
# =============================================================================

def vec_to_new_frame(vec, goal_direction):
    """
    将向量从世界坐标系转换到目标坐标系
    
    目标坐标系定义:
    - x轴: 目标方向（归一化）
    - y轴: z轴 × x轴（水平面内垂直于目标方向）
    - z轴: 垂直于 x-y 平面
    
    这种坐标变换使得策略可以学习旋转不变的行为
    
    参数:
        vec: 待转换的向量 [N, 3] 或 [N, M, 3]
        goal_direction: 目标方向向量（定义新坐标系的 x 轴）[N, 3]
        
    返回:
        vec_new: 在新坐标系下的向量 [N, 3] 或 [N, M, 3]
    """
    # 处理单样本输入
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)

    # 新坐标系 x 轴: 目标方向归一化
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    
    # 世界坐标系 z 轴
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # 新坐标系 y 轴: z × x（叉积，确保右手坐标系）
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # 新坐标系 z 轴: x × y（确保正交）
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    
    # 根据输入维度选择计算方式
    if len(vec.size()) == 3:
        # 输入形状 [N, M, 3]，批量处理
        # 通过 batch matrix multiplication 计算投影
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        # 输入形状 [N, 3]
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    # 拼接三个分量
    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    """
    将向量从局部目标坐标系转换回世界坐标系
    
    这是 vec_to_new_frame 的逆操作
    首先计算世界坐标系在目标坐标系中的表示，然后进行转换
    
    参数:
        vec: 目标坐标系中的向量 [N, 3]
        goal_direction: 目标方向（定义局部坐标系）[N, 3]
        
    返回:
        world_frame_vel: 世界坐标系中的向量 [N, 3]
    """
    # 世界坐标系的 x 轴方向
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # 计算世界坐标系在目标坐标系中的表示
    # 这相当于目标坐标系的基向量在世界坐标系中的表示的转置
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # 使用这个变换将局部速度转换回世界坐标
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel


def construct_input(start, end):
    """
    构建正则表达式输入模式
    
    用于生成匹配数字范围的正则表达式，如 (0|1|2|3|4)
    
    参数:
        start: 起始数字
        end: 结束数字（不包含）
        
    返回:
        正则表达式字符串
    """
    input = []
    for n in range(start, end):
        input.append(f"{n}")
    return "(" + "|".join(input) + ")"
