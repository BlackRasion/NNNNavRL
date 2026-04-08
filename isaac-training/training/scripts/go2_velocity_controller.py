import torch
import torch.nn as nn

try:
    from tensordict.tensordict import TensorDictBase
    from torchrl.envs.transforms import Transform
    from torchrl.data import UnboundedContinuousTensorSpec
except ModuleNotFoundError:
    TensorDictBase = object
    UnboundedContinuousTensorSpec = object

    class Transform:
        def __init__(self, *args, **kwargs):
            pass


class Go2VelocityController(nn.Module):
    def __init__(
        self,
        dt: float = 0.016,
    ):
        super().__init__()
        self.dt = dt

    def forward(
        self, actions: torch.Tensor, emergency_stop: torch.Tensor | None = None
    ) -> torch.Tensor:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        velocity_commands = actions
        if emergency_stop is not None:
            mask = emergency_stop.to(dtype=torch.bool, device=velocity_commands.device)
            while mask.dim() < velocity_commands.dim():
                mask = mask.unsqueeze(-1)
            velocity_commands = torch.where(
                mask, torch.zeros_like(velocity_commands), velocity_commands
            )
        return velocity_commands


class Go2VelController(Transform):
    def __init__(
        self,
        controller: Go2VelocityController,
        action_key: tuple = ("agents", "action"),
        emergency_stop_key: tuple = ("agents", "emergency_stop"),
    ):
        super().__init__([], in_keys_inv=[])
        self.controller = controller
        self.action_key = action_key
        self.emergency_stop_key = emergency_stop_key

    def transform_input_spec(self, input_spec) -> UnboundedContinuousTensorSpec:
        action_spec = input_spec[("full_action_spec", *self.action_key)]
        spec = UnboundedContinuousTensorSpec(
            action_spec.shape[:-1] + (3,), device=action_spec.device
        )
        input_spec[("full_action_spec", *self.action_key)] = spec
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        actions = tensordict.get(self.action_key)
        emergency_stop = tensordict.get(self.emergency_stop_key, None)
        velocity_commands = self.controller(actions, emergency_stop=emergency_stop)
        torch.nan_to_num_(velocity_commands, 0.0)
        tensordict.set(self.action_key, velocity_commands)
        return tensordict

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict
