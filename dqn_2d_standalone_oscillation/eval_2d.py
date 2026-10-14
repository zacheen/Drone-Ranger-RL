"""
Greedy (epsilon=0) evaluation on reproducible layouts. Each env is scored on
its first episode only, so with the same --seed and --num_obstacles every
checkpoint faces the same starts, targets and obstacle layouts, and two
checkpoints can be compared episode for episode. dqn_diagnose_2d.py instead
keeps the envs running, so its later episodes depend on how fast the policy
finished the earlier ones.
"""
import argparse
import math

import torch

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork
from dqn_2d_env import OUTCOMES, add_env_args, env_from_args

DIST_BUCKETS = ((0.0, 3.0), (3.0, 6.0), (6.0, math.inf))  # start-to-target distance, m

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
add_env_args(parser)
add_config_args(parser)
args = parser.parse_args()
cfg: DQNConfig = config_from_args(args)

device = "cpu"
n = args.num_episodes
env = env_from_args(args, n, cfg, device)
q_net = QNetwork(state_dim=env.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
q_net.eval()

# Seeded after the network is built, whose init draws a hidden_dim-dependent
# amount of randomness, so layouts depend on --seed and the env options only.
torch.manual_seed(args.seed)
obs_dict, _ = env.reset()
state = obs_dict["dqn_state"]
init_dist = torch.norm(env.target - env.pos, dim=-1)
outcome = [None] * n
steps = torch.zeros(n, dtype=torch.long)

# step() truncates every episode at env.max_steps, so this loop ends every first episode.
for t in range(env.max_steps):
    with torch.no_grad():
        action_idx = q_net(state).argmax(dim=-1)
    obs_dict, _, terminated, truncated, _ = env.step(discrete_action_to_continuous(action_idx, device))
    state = obs_dict["dqn_state"]
    done = terminated | truncated
    if done.any():
        outcomes = env.last_outcomes()
        for i in done.nonzero(as_tuple=True)[0].tolist():
            if outcome[i] is None:
                outcome[i] = outcomes[i]
                steps[i] = t + 1
    if all(o is not None for o in outcome):
        break

print(f"[EVAL2D] {args.checkpoint}  episodes={n}  num_obstacles={args.num_obstacles}  "
      f"distance_mode={args.distance_mode}  seed={args.seed}")
for k in OUTCOMES:
    p = sum(o == k for o in outcome) / n
    half_width = 1.96 * math.sqrt(p * (1 - p) / n)  # 95% normal-approximation interval
    print(f"  {k:<14} {p * 100:6.1f}%  (+/-{half_width * 100:.1f})")
reached = torch.tensor([o == "reached" for o in outcome])
if reached.any():
    print(f"  mean steps to reach: {steps[reached].float().mean().item():.0f}")
for lo, hi in DIST_BUCKETS:
    in_bucket = (init_dist >= lo) & (init_dist < hi)
    if in_bucket.any():
        label = f"{lo:g}-{hi:g}" if math.isfinite(hi) else f"{lo:g}+"
        print(f"  reached | start-target {label} m: {reached[in_bucket].float().mean().item() * 100:5.1f}%  "
              f"(n={int(in_bucket.sum())})")
