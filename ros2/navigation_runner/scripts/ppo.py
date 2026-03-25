import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from utils import ValueNorm, make_mlp, GAE, IndependentBeta, BetaActor, vec_to_world

class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        
        # =========================================================================
        # 步骤 1: 构建特征提取网络
        # =========================================================================
        # 1.1 LiDAR 静态障碍物特征提取器 (CNN)
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        # 1.2 动态障碍物特征提取器 (MLP)
        # -------------------------------------------------------------------------
        # 输入: 动态障碍物信息 [batch, dyn_obs_num=5, features]
        # 输出: 64 维特征向量
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        # 1.3 组合特征提取器
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        # =========================================================================
        # 步骤 2: 构建 Actor 网络（策略网络）
        # =========================================================================
        self.n_agents, self.action_dim = action_spec.shape
        self.actor = ProbabilisticActor(
            # BetaActor: 输入特征，输出 alpha 和 beta 参数
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # =========================================================================
        # 步骤 3: 构建 Critic 网络（价值网络）
        # =========================================================================
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # GAE (Generalized Advantage Estimation): 广义优势估计
        # gamma: 折扣因子 (0.99)，控制未来奖励的重要性
        # lambda: GAE 参数 (0.95)，控制偏差-方差权衡        
        self.gae = GAE(0.99, 0.95)
        self.critic_loss_fn = nn.HuberLoss(delta=10) # Huber Loss: 结合 L1 和 L2 损失的优点
 
        # =========================================================================
        # 步骤 5: 优化器设置
        # =========================================================================
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        # =========================================================================
        # 步骤 6: 网络初始化
        # =========================================================================
        dummy_input = observation_spec.zero() # 创建零观测作为虚拟输入
        # print("dummy_input: ", dummy_input)
        self.__call__(dummy_input)

        # 使用正交初始化网络权重
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        """
        前向传播: 根据观测生成动作
        
        参数:
            tensordict: 包含观测数据的 TensorDict
            
        返回:
            tensordict: 添加了动作和价值估计的 TensorDict
        """
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # 使用方向向量将局部动作转换为世界坐标系
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"]) # direction:无人机的朝向向量
        tensordict["agents", "action"] = actions_world
        return tensordict