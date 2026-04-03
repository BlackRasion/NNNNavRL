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
        action_limit: float = 2.0,
        max_linear_vel: float = 2.5,
        max_angular_vel: float = 1.57,
        smoothing_alpha: float = 0.3,
    ):
        super().__init__()
        self.dt = dt
        self.max_linear_vel = nn.Parameter(
            torch.tensor(max_linear_vel), requires_grad=False
        )
        self.max_angular_vel = nn.Parameter(
            torch.tensor(max_angular_vel), requires_grad=False
        )
        self.action_limit = nn.Parameter(
            torch.tensor(action_limit), requires_grad=False
        )
        self.smoothing_alpha = nn.Parameter(
            torch.tensor(smoothing_alpha), requires_grad=False
        )
        self._prev_velocity_commands = None

    def _map_to_velocity(self, actions: torch.Tensor) -> torch.Tensor:
        scale_linear = self.max_linear_vel / self.action_limit
        scale_angular = self.max_angular_vel / self.action_limit
        vx = actions[..., 0] * scale_linear
        vy = actions[..., 1] * scale_linear
        vyaw = actions[..., 2] * scale_angular
        return torch.stack([vx, vy, vyaw], dim=-1)

    def _smooth_velocity(self, velocity_commands: torch.Tensor) -> torch.Tensor:
        if (
            self._prev_velocity_commands is None
            or self._prev_velocity_commands.shape != velocity_commands.shape
            or self._prev_velocity_commands.device != velocity_commands.device
        ):
            self._prev_velocity_commands = velocity_commands.clone()
            return velocity_commands
        alpha = torch.clamp(self.smoothing_alpha, 0.0, 1.0)
        smoothed = (
            alpha * velocity_commands + (1.0 - alpha) * self._prev_velocity_commands
        )
        self._prev_velocity_commands = smoothed.clone()
        return smoothed

    def forward(
        self, actions: torch.Tensor, emergency_stop: torch.Tensor | None = None
    ) -> torch.Tensor:
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        actions = torch.clamp(actions, -self.action_limit, self.action_limit)
        velocity_commands = self._map_to_velocity(actions)
        velocity_commands = self._smooth_velocity(velocity_commands)
        velocity_commands[..., 0] = torch.clamp(
            velocity_commands[..., 0], -self.max_linear_vel, self.max_linear_vel
        )
        velocity_commands[..., 1] = torch.clamp(
            velocity_commands[..., 1], -self.max_linear_vel, self.max_linear_vel
        )
        velocity_commands[..., 2] = torch.clamp(
            velocity_commands[..., 2], -self.max_angular_vel, self.max_angular_vel
        )
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
