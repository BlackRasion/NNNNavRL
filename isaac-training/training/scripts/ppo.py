"""
PPO (Proximal Policy Optimization) 算法实现 - Go2 四足机器人导航

该文件实现了 Go2 四足机器人的 PPO 算法，包括：
1. 特征提取器（CNN + MLP）
   - LiDAR 静态障碍物特征: [batch, 1, 36, 3] -> 128 维
   - 机器人状态特征: 8 维 (位置、速度、方向等)
   - 动态障碍物特征: [batch, 1, 5, 10] -> 64 维
2. Actor 网络（Beta 分布策略）
   - 输出动作维度: 3 (Vx, Vy, Vyaw)
3. Critic 网络（状态价值估计）
4. GAE 估计
5. PPO 损失计算和参数更新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, GAE, make_batch, IndependentBeta, BetaActor


class PPO(TensorDictModuleBase):
    """
    PPO 算法实现
    网络结构:
    1. Feature Extractor: 提取 LiDAR、状态、动态障碍物特征
    2. Actor: 输出 Beta 分布参数 (alpha, beta)，采样动作
    3. Critic: 估计状态价值 V(s)
    """

    def __init__(self, cfg, observation_spec, action_spec, device):
        """
        初始化 PPO 策略网络

        参数:
            cfg: 配置对象
            observation_spec: 观测空间规格
            action_spec: 动作空间规格
            device: 计算设备
        """
        super().__init__()
        self.cfg = cfg
        self.device = device

        # =========================================================================
        # 步骤 1: 构建特征提取网络
        # =========================================================================
        # 1.1 LiDAR 静态障碍物特征提取器 (CNN)
        # 输入: LiDAR 数据 [batch, channels=1, 水平=36, 垂直=3]
        # 输出: 128 维特征向量
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]),
            nn.ELU(),
            nn.LazyConv2d(
                out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]
            ),
            nn.ELU(),
            nn.LazyConv2d(
                out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]
            ),
            nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128),
            nn.LayerNorm(128),
        ).to(self.device)

        # 1.2 动态障碍物特征提取器 (MLP)
        # -------------------------------------------------------------------------
        # 输入: 动态障碍物信息 [batch, dyn_obs_num=5, features=10]
        # 输出: 64 维特征向量
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"), make_mlp([128, 64])
        ).to(self.device)

        # 1.3 组合特征提取器
        # - 静态障碍物特征 (LiDAR): 128 维
        # - 机器人内部状态特征 (位置、速度、方向等): 来自 observation.state 8 维
        # - 动态障碍物特征: 64 维
        # - 总特征维度: 128 + 8 + 64 = 200 维
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(
                feature_extractor_network,
                [("agents", "observation", "lidar")],
                ["_cnn_feature"],
            ),
            TensorDictModule(
                dynamic_obstacle_network,
                [("agents", "observation", "dynamic_obstacle")],
                ["_dynamic_obstacle_feature"],
            ),
            CatTensors(
                [
                    "_cnn_feature",
                    ("agents", "observation", "state"),
                    "_dynamic_obstacle_feature",
                ],
                "_feature",
                del_keys=False,
            ),
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        # =========================================================================
        # 步骤 2: 构建 Actor 网络（策略网络）
        # =========================================================================
        # 动作维度: 3 (Vx, Vy, Vyaw) - Go2 四足机器人的线速度和角速度
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            TensorDictModule(
                BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]
            ),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")],
            distribution_class=IndependentBeta,
            return_log_prob=True,
        ).to(self.device)

        # =========================================================================
        # 步骤 3: 构建 Critic 网络（价值网络）
        # =========================================================================
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"]
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # =========================================================================
        # 步骤 4: 损失函数和相关组件
        # =========================================================================
        # GAE (Generalized Advantage Estimation): 广义优势估计
        # gamma: 折扣因子 (0.99)，控制未来奖励的重要性
        # lambda: GAE 参数 (0.95)，控制偏差-方差权衡
        self.gae = GAE(0.99, 0.95)
        self.critic_loss_fn = nn.HuberLoss(
            delta=10
        )  # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # =========================================================================
        # 步骤 5: 优化器设置
        # =========================================================================
        self.feature_extractor_optim = torch.optim.Adam(
            self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate
        )
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor.learning_rate
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic.learning_rate
        )

        # =========================================================================
        # 步骤 6: 网络初始化
        # =========================================================================
        dummy_input = observation_spec.zero()
        self.__call__(dummy_input)

        # 使用正交初始化网络权重
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        """
        前向传播: 根据观测生成动作（修复版 - 使用机器人坐标系）

        参数:
            tensordict: 包含观测数据的 TensorDict
        返回:
            tensordict: 添加了动作和价值估计的 TensorDict

        关键改进：
        - 策略输出机器人坐标系下的速度命令（Vx, Vy, Vyaw）
        - 不再转换到世界坐标系，避免坐标系混乱
        - 状态空间已包含机器人朝向信息，策略可以学习正确的运动方向
        """
        self.feature_extractor(tensordict)  # 提取特征
        self.actor(tensordict)  # Actor 前向: 采样动作 (归一化到 0-1)
        self.critic(tensordict)  # Critic 前向: 估计状态价值

        # =========================================================================
        # 动作处理：将归一化动作映射到实际速度范围
        # =========================================================================
        # 策略输出机器人坐标系下的速度命令：
        # - Vx: 前进速度（正值前进，负值后退）
        # - Vy: 横向速度（正值左移，负值右移）
        # - Vyaw: 旋转速度（正值逆时针旋转，负值顺时针旋转）

        # 将归一化动作 (0,1) 映射到实际速度范围 [-action_limit, action_limit]
        actions = (
            2 * tensordict["agents", "action_normalized"] - 1.0
        ) * self.cfg.actor.action_limit

        # ❌ 删除世界坐标系转换（避免坐标系混乱）
        # 现在策略直接输出机器人坐标系下的速度命令
        # actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])

        tensordict["agents", "action"] = actions
        return tensordict

    def train(self, tensordict):
        """
        PPO 训练主函数
        参数:
            tensordict: 包含一批环境交互数据的 TensorDict
                       形状: (num_envs, num_frames, dim), batchsize = num_env * num_frames(帧数)
        返回:
            dict: 训练统计信息（损失、梯度范数等）
        """
        next_tensordict = tensordict["next"]
        # =====================================================================
        # 步骤 1: 计算下一状态的价值估计（无梯度）
        # =====================================================================
        with torch.no_grad():
            # 使用 vmap 对批次中的每个样本提取特征
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict)
            # Critic 估计下一状态价值
            next_values = self.critic(next_tensordict)["state_value"]

        # =====================================================================
        # 步骤 2: 提取训练数据
        # =====================================================================
        rewards = tensordict["next", "agents", "reward"]  # 奖励: 状态转移获得的即时奖励

        dones = tensordict["next", "terminated"]  # 终止标志: 下一状态是否为终止状态

        values = tensordict["state_value"]  # 当前状态价值: 前向传播时已计算并存储

        values = self.value_norm.denormalize(values)  # 反归一化当前价值
        next_values = self.value_norm.denormalize(next_values)

        # =====================================================================
        # 步骤 3: 计算 GAE 优势估计和回报
        # =====================================================================
        adv, ret = self.gae(
            rewards, dones, values, next_values
        )  # GAE 提供低方差的优势估计，同时控制偏差

        # 标准化优势: 均值为 0，标准差为 1
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)  # 防止除零

        # 更新价值归一化的运行统计量
        self.value_norm.update(ret)

        # 归一化回报（用于 Critic 训练）
        ret = self.value_norm.normalize(ret)

        # 存储优势和回报到 tensordict
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        # =====================================================================
        # 步骤 4: PPO 训练循环
        # =====================================================================
        # PPO 特点: 对同一批数据进行多次 epoch 训练，每次 epoch 将数据分为多个 minibatch

        infos = []  # 存储每个 minibatch 的训练信息

        for epoch in range(self.cfg.training_epoch_num):
            # 将数据分割为 minibatches
            batch = make_batch(tensordict, self.cfg.num_minibatches)

            # 遍历每个 minibatch 进行更新
            for minibatch in batch:
                infos.append(self._update(minibatch))

        # 将所有 minibatch 的信息堆叠为 TensorDict
        infos = torch.stack(infos).to_tensordict()

        # 计算平均统计量
        infos = infos.apply(torch.mean, batch_size=[])

        # 转换为普通字典返回
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict):
        """
        单次 PPO 更新（处理一个 minibatch）
        参数:
            tensordict: 一个 minibatch 的数据
        返回:
            TensorDict: 包含各种损失和统计信息
        """
        # =====================================================================
        # 步骤 1: 提取特征
        # =====================================================================
        self.feature_extractor(tensordict)

        # =====================================================================
        # 步骤 2: 计算新策略的动作概率
        # =====================================================================
        # 获取当前策略的分布
        action_dist = self.actor.get_dist(tensordict)

        # 计算旧动作在新策略下的对数概率，用于计算重要性采样比率 (ratio)
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")])

        # =====================================================================
        # 步骤 3: 计算熵损失（鼓励探索）
        # =====================================================================
        # 熵衡量策略的随机性，高熵 = 更多探索
        # 损失函数中减去熵，鼓励策略保持一定随机性
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # =====================================================================
        # 步骤 4: 计算 Actor 损失（PPO 核心）
        # =====================================================================
        # 重要性采样比率: pi_new / pi_old
        # 通过对数概率差计算: exp(log_prob_new - log_prob_old)
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)

        # 获取 GAE 优势估计
        advantage = tensordict["adv"]

        # PPO 裁剪目标函数
        surr1 = advantage * ratio  # surr1: 未裁剪的目标
        surr2 = advantage * ratio.clamp(
            1.0 - self.cfg.actor.clip_ratio, 1.0 + self.cfg.actor.clip_ratio
        )  # surr2: 裁剪后的目标，限制 ratio 在 [1-ε, 1+ε] 范围内

        # 取最小值，防止策略更新过大
        # 负号因为我们要最大化目标，但 PyTorch 默认最小化损失
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim

        # =====================================================================
        # 步骤 5: 计算 Critic 损失
        # =====================================================================
        # 旧价值估计（用于裁剪）
        b_value = tensordict["state_value"]

        # 目标回报（来自 GAE）
        ret = tensordict["ret"]

        # 新价值估计
        value = self.critic(tensordict)["state_value"]

        # 裁剪价值估计，防止 Critic 更新过大
        value_clipped = b_value + (value - b_value).clamp(
            -self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio
        )

        # 计算两种损失的 Huber Loss
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)

        # 取最大值，确保使用更保守（更大）的损失
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # =====================================================================
        # 步骤 6: 组合总损失并反向传播
        # =====================================================================
        # 总损失 = 熵损失 + Actor 损失 + Critic 损失
        loss = entropy_loss + actor_loss + critic_loss

        # 清零梯度
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        # 反向传播
        loss.backward()

        # =====================================================================
        # 步骤 7: 梯度裁剪和参数更新
        # =====================================================================
        # 梯度裁剪: 防止梯度爆炸，限制梯度范数
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.actor.parameters(), max_norm=5.0
        )
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), max_norm=5.0
        )

        # 执行参数更新
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()

        # =====================================================================
        # 步骤 8: 计算解释方差（训练监控指标）
        # =====================================================================
        # 解释方差衡量 Critic 对回报的拟合程度
        # 值越接近 1，说明 Critic 预测越准确
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()

        # 返回训练统计信息
        return TensorDict(
            {
                "actor_loss": actor_loss,  # Actor 损失
                "critic_loss": critic_loss,  # Critic 损失
                "entropy": entropy_loss,  # 熵损失
                "actor_grad_norm": actor_grad_norm,  # Actor 梯度范数
                "critic_grad_norm": critic_grad_norm,  # Critic 梯度范数
                "explained_var": explained_var,  # 解释方差
            },
            [],
        )
