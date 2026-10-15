"""
Shared action-space definition and Q-network for the DQN experiments --
train_dqn.py / dqn_diagnose.py / watch_dqn.py / q_sweep.py all import
from here instead of each keeping their own copy.

Design v2 (turning fully decoupled from forward motion, matching the
embodied-AI navigation convention -- e.g. Habitat/AI2-THOR's
MoveForward + TurnLeft + TurnRight):
  action 0: forward     vx=forward_speed, yaw_rate=0,        no turning
  action 1: turn left   vx=0,             yaw_rate=+turn_rate, no forward motion
  action 2: turn right  vx=0,             yaw_rate=-turn_rate, no forward motion

Compared to the deprecated v1 (4-action) design: v1's "turn toward
target" action set its raw vx input to 0, intending vx=0 -- but because
vx passes through a ((tanh(x)+1)/2) squash (used to force vx
non-negative), an input of exactly 0 maps to the MIDPOINT of the output
range (1.25 m/s), not the minimum. This made that one action the only
one that both moved forward AND adapted its turn direction to the
target, causing the network's learned policy to depend almost entirely
on that single accidental "smart" action -- fragile, and prone to
collapsing under any training disturbance (e.g. resuming from a
checkpoint). v1's other 3 actions had fixed turn directions blind to
where the target actually was.

v2 fixes this by making vx an explicit, correctly-computed value for
every action (no more "leave it at the default and hope it's 0"), and
by giving turning its own dedicated actions (1 and 2) instead of
overloading a forward-motion action with adaptive steering.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dqn_config import DQNConfig

_cfg = DQNConfig()

NUM_ACTIONS = _cfg.num_actions
STATE_DIM = _cfg.state_dim
# Action index = speed * NUM_TURNS + turn, speed in (stop, forward), turn in (none, left, right).
NUM_SPEEDS, NUM_TURNS = 2, 3
ACTION_NAMES = ["hover", "turn_left", "turn_right", "forward", "fwd_left", "fwd_right"]


def _arctanh(y):
    return 0.5 * math.log((1 + y) / (1 - y))


# vx's squash formula is ((tanh(x)+1)/2)*max_xy_vel (forces non-negative,
# see _apply_action in the env). Inverting it:
#   to get vx=forward_speed, need x = arctanh(2*forward_speed/max_xy_vel - 1)
#   to get vx=0, need x = arctanh(-1), which is undefined (division by
#   zero) -- approximated with a large negative number so tanh(x) is
#   close enough to -1 that the squashed vx is negligible (~1e-4 m/s).
_VX_FORWARD_RAW = _arctanh(2 * _cfg.forward_speed / _cfg.max_xy_vel - 1)
_VX_ZERO_RAW = -10.0
_YAW_TURN_RAW = _arctanh(_cfg.turn_rate / _cfg.max_yaw_rate)


def discrete_action_to_continuous(action_idx: torch.Tensor, device) -> torch.Tensor:
    """action_idx: (N,) values 0-5. Returns (N,3) continuous action
    tensor, ready to pass to env.step(). No dy parameter needed --
    unlike v1, turn direction is a discrete choice the network makes
    (left vs right), not something computed dynamically from dy's sign."""
    n = action_idx.shape[0]
    out = torch.zeros(n, 3, device=device)
    speed, turn = action_idx // NUM_TURNS, action_idx % NUM_TURNS
    out[:, 0] = torch.where(speed == 1, _VX_FORWARD_RAW, _VX_ZERO_RAW)
    out[turn == 1, 2] = _YAW_TURN_RAW
    out[turn == 2, 2] = -_YAW_TURN_RAW
    return out


class QNetwork(nn.Module):
    def __init__(self, state_dim=None, action_dim=None, hidden_dim=None):
        super().__init__()
        cfg = DQNConfig()
        state_dim = state_dim if state_dim is not None else cfg.state_dim
        action_dim = action_dim if action_dim is not None else cfg.num_actions
        hidden_dim = hidden_dim if hidden_dim is not None else cfg.hidden_dim
        if action_dim != NUM_SPEEDS * NUM_TURNS:
            raise ValueError(f"the two-head network outputs {NUM_SPEEDS * NUM_TURNS} actions, got action_dim={action_dim}")
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.speed_head = nn.Linear(hidden_dim, NUM_SPEEDS)
        self.turn_head = nn.Linear(hidden_dim, NUM_TURNS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Q(s, speed, turn) = Q_speed + Q_turn, flattened to the action index order above.
        q = self.speed_head(x).unsqueeze(-1) + self.turn_head(x).unsqueeze(-2)
        return q.flatten(start_dim=-2)
