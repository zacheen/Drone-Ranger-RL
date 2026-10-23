"""
Step-by-step trace for the 2D environment -- prints chosen action,
position delta, distance, heading, and the full reward breakdown for
every step of one episode. Used to check whether a policy that appears
"stuck" in trajectory plots is farming rew_yaw_aligned by standing still
(or oscillating in place) rather than genuinely committing to reach the
target.
"""
import argparse

import torch

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork, ACTION_NAMES
from dqn_2d_env import add_env_args, env_from_args

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--max_steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=None, help="torch manual seed, for reproducing a specific start/target")
add_env_args(parser)
add_config_args(parser)
args = parser.parse_args()
cfg: DQNConfig = config_from_args(args)

if args.seed is not None:
    torch.manual_seed(args.seed)

device = "cpu"
env = env_from_args(args, 1, cfg, device)
q_net = QNetwork(state_dim=env.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
q_net.load_state_dict(torch.load(args.checkpoint, map_location=device))
q_net.eval()
print(f"[TRACE] loaded checkpoint: {args.checkpoint}  "
      f"num_obstacles={args.num_obstacles}  distance_mode={args.distance_mode}")

obs_dict, _ = env.reset()
state = obs_dict["dqn_state"]
print(f"[TRACE] start={env.pos[0].tolist()}  target={env.target[0].tolist()}")
for i in range(args.num_obstacles):
    x, y = env.obstacle_xy[0, i].tolist()
    parked = "" if env.obstacle_active[0, i] else "  (parked, no valid spot found)"
    print(f"[TRACE] obstacle {i} at ({x:.2f}, {y:.2f}){parked}")
print()

# Coords mode shows its nearest slot in place of the sector readings.
reading_names = {"coords": ("s1_x", "s1_y", "v1"), "camera": ("d_front", "d_left", "d_right"),
                 "ground_truth": ("d_fl", "d_fr", "d_left", "d_right")}[args.distance_mode]
header = (f"{'step':>5} {'action':>10} {'pos_x':>7} {'pos_y':>7} {'d(pos)':>7} "
          + "".join(f"{name:>7} " for name in reading_names) +
          f"{'dist':>7} {'yaw_cos':>8} {'vx':>7} {'r_progress':>10} {'r_total':>8}")
print(header)
print("-" * len(header))

prev_pos = env.pos[0].clone()
for step in range(args.max_steps):
    with torch.no_grad():
        action_idx = q_net(state).argmax(dim=-1)
    actions_continuous = discrete_action_to_continuous(action_idx, device)

    obs_dict, reward, terminated, truncated, info = env.step(actions_continuous)
    state = obs_dict["dqn_state"]

    # On the episode's last step env.pos already holds the next spawn point.
    end = env.pre_reset_state
    breakdown = env.last_reward_breakdown
    new_pos = end["pos"][0]
    dist_now = torch.norm(end["target"][0] - new_pos)
    pos_delta = torch.norm(new_pos - prev_pos).item()
    prev_pos = new_pos

    print(f"{step:>5} {ACTION_NAMES[action_idx[0].item()]:>10} "
          f"{new_pos[0].item():>7.3f} {new_pos[1].item():>7.3f} {pos_delta:>7.4f} "
          + "".join(f"{state[0, c].item():>7.3f} " for c in range(len(reading_names))) +
          f"{dist_now.item():>7.3f} {breakdown['yaw_cos'][0].item():>8.3f} "
          f"{breakdown['vx_signed'][0].item():>7.3f} {breakdown['rew_progress'][0].item():>10.4f} "
          f"{reward[0].item():>8.4f}")

    if (terminated | truncated)[0].item():
        print(f"\n[TRACE] episode ended at step {step}, outcome={env.last_outcomes()[0]}")
        break
