import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def get_controller_cls():
    from go2_velocity_controller import Go2VelocityController

    return Go2VelocityController


def test_velocity_controller_respects_physical_limits():
    controller = get_controller_cls()(
        dt=0.016,
        action_limit=2.0,
        max_linear_vel=2.5,
        max_angular_vel=1.57,
        smoothing_alpha=1.0,
    )
    actions = (torch.rand(512, 3) - 0.5) * 20.0
    commands = controller(actions)
    assert commands.shape == (512, 3)
    assert torch.all(commands[:, 0].abs() <= 2.5 + 1e-6)
    assert torch.all(commands[:, 1].abs() <= 2.5 + 1e-6)
    assert torch.all(commands[:, 2].abs() <= 1.57 + 1e-6)


def test_velocity_controller_emergency_stop_zeroes_output():
    controller = get_controller_cls()(smoothing_alpha=1.0)
    actions = torch.ones(16, 3) * 2.0
    stop = torch.ones(16, 1, dtype=torch.bool)
    commands = controller(actions, emergency_stop=stop)
    assert torch.allclose(commands, torch.zeros_like(commands))
