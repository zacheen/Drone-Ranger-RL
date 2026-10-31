"""
Unified configuration for the DQN experiments -- network architecture,
action-space physical parameters, and training hyperparameters.

Supports command-line overrides (argparse) so A/B comparisons (e.g.
hidden_dim=64 vs 128) don't require manually editing code.

Reward weights (yaw_aligned, etc.) are NOT managed here -- those stay in
the environment file drone_nav_v2_env_dqn.py, following Isaac Lab's own
@configclass convention. This keeps DQN-algorithm config and
environment/task config as separate concerns.
"""
import argparse
from dataclasses import dataclass, fields

# The forward speeds the actions choose from, as fractions of forward_speed.
# None is zero, so the drone cannot stop. At 0.2 the slowest turn radius was 0.375 m, too wide to
# steer off an obstacle already close, and 0.05 brings it to about 0.09 m.
SPEED_FRACTIONS = (0.05, 0.2, 1.0)


@dataclass
class DQNConfig:
    # ── Network architecture ──
    state_dim: int = 5
    num_actions: int = 9  # len(SPEED_FRACTIONS) speeds times 3 turns
    hidden_dim: int = 128  # raised from 64 -- testing whether network
                            # capacity was the bottleneck (see experiment log)

    # ── Action-space physical parameters (forward speed / turn rate) ──
    forward_speed: float = 1.5
    turn_rate: float = 0.8
    max_xy_vel: float = 2.5
    max_yaw_rate: float = 1.57

    # ── Training hyperparameters (initial values from the LT-DQN paper's
    # Table, some since tuned) ──
    replay_buffer_size: int = 500000  # 50000 held only the last ~780 vector steps of 64 envs
    batch_size: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.98
    epsilon_start: float = 1.0
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.995
    target_update_every: int = 20  # counted in episodes, not gradient steps; unused while target_tau > 0
    # Polyak step toward the online net after every gradient step, 0 for the periodic hard copy.
    # 0.0005 gives a time constant of about 2000 gradient steps, matching the hard copy every 20 episodes of 100 steps.
    target_tau: float = 0.0005
    resume_epsilon: float = 0.3  # epsilon restart value on --resume_from
                                  # (neither 1.0 nor epsilon_min)
    # Exploratory actions are held for a zeta(mu) number of steps, at most explore_max_hold.
    # explore_max_hold 1 gives plain per-step epsilon-greedy.
    explore_hold_mu: float = 2.0
    explore_max_hold: int = 100
    # Per-step penalty for passing near an obstacle, rising linearly from 0 at proximity_margin metres
    # of clearance short of contact to proximity_penalty at contact. 0 leaves the reward unchanged.
    proximity_penalty: float = 0.0
    proximity_margin: float = 0.3


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add one command-line argument per DQNConfig field to an existing
    argparse parser (e.g. --hidden_dim), using DQNConfig's dataclass
    defaults as the argparse defaults. Callers build their own parser,
    call this to add these fields, then pass the parsed args to
    config_from_args() to get a DQNConfig with any CLI overrides applied."""
    defaults = DQNConfig()
    for f in fields(DQNConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=getattr(defaults, f.name))


def config_from_args(args: argparse.Namespace) -> DQNConfig:
    """Build a DQNConfig from already-parsed args (must include every
    field added by add_config_args)."""
    field_names = {f.name for f in fields(DQNConfig)}
    kwargs = {k: v for k, v in vars(args).items() if k in field_names}
    return DQNConfig(**kwargs)
