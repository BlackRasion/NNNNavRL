import torch
import torch.nn as nn
from tensordict.tensordict import TensorDictBase
from torchrl.envs.transforms import Transform


class Go2VelocityController(nn.Module):
    """
    Go2速度控制器：将动作转换为速度命令
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0) # 处理 NaN 值，确保速度命令有效值


class Go2VelController(Transform):
    """
    Go2速度控制器变换：在环境中应用速度控制器
    """

    def __init__(
        self,
        controller: Go2VelocityController = None,
        action_key: tuple = ("agents", "action"),
    ) -> None:
        super().__init__([], in_keys_inv=[action_key])
        self.controller = controller if controller is not None else Go2VelocityController()
        self.action_key = action_key

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        actions = tensordict.get(self.action_key)
        tensordict.set(self.action_key, self.controller(actions))
        return tensordict

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict
