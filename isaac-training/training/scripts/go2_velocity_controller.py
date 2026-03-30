import torch
import torch.nn as nn
from tensordict.tensordict import TensorDictBase
from torchrl.envs.transforms import Transform
from torchrl.data import UnboundedContinuousTensorSpec


class Go2VelocityController(nn.Module):
    """
    Go2 机器人速度控制器
    
    将策略网络输出的归一化动作转换为速度命令
    """
    
    def __init__(self, dt: float = 0.016):
        super().__init__()
        self.dt = dt
        self.max_linear_vel = nn.Parameter(torch.tensor(2.0), requires_grad=False)
        self.max_angular_vel = nn.Parameter(torch.tensor(3.14), requires_grad=False)
    
    def forward(self, actions: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            actions: 归一化动作 [batch, 3]，范围 [0, 1]
            robot_state: 机器人状态 [batch, 13] (pos, quat, lin_vel, ang_vel)
        
        返回:
            velocity_commands: 速度命令 [batch, 3]，(Vx, Vy, Vyaw)
        """
        actions = 2.0 * actions - 1.0
        vx = actions[..., 0] * self.max_linear_vel
        vy = actions[..., 1] * self.max_linear_vel
        vyaw = actions[..., 2] * self.max_angular_vel
        velocity_commands = torch.stack([vx, vy, vyaw], dim=-1)
        return velocity_commands


class Go2VelController(Transform):
    """
    TorchRL 变换：将策略输出转换为速度命令
    """
    
    def __init__(
        self,
        controller: Go2VelocityController,
        action_key: tuple = ("agents", "action"),
        robot_state_key: tuple = ("info", "robot_state"),
    ):
        super().__init__([], in_keys_inv=[robot_state_key])
        self.controller = controller
        self.action_key = action_key
        self.robot_state_key = robot_state_key
    
    def transform_input_spec(self, input_spec) -> UnboundedContinuousTensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(
            action_spec.shape[:-1] + (3,),
            device=action_spec.device
        )
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec
    
    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        robot_state = tensordict.get(self.robot_state_key)
        if robot_state.shape[-1] >= 13:
            robot_state = robot_state[..., :13]
        actions = tensordict.get(self.action_key)
        velocity_commands = self.controller(actions, robot_state)
        torch.nan_to_num_(velocity_commands, 0.0)
        tensordict.set(self.action_key, velocity_commands)
        return tensordict
    
    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict
