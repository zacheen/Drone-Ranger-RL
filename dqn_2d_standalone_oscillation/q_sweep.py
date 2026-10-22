"""
q_sweep.py -- does not depend on Isaac Sim; offline sweep of the Q-network's
decision boundary across the full heading_error range (-180 to +180 degrees).
Used to check whether the left/right turn decision is symmetric and
whether the margin (Q-value gap between top-2 actions) collapses in any
angle range.

Usage:
    python3 q_sweep.py --checkpoint /path/to/checkpoint.pt
"""
import argparse
import math

import torch

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import QNetwork, ACTION_NAMES, NUM_TURNS
from dqn_2d_env import PREV_ACTION_DIM


def sweep(checkpoint_path: str, cfg: DQNConfig, distances, angle_step_deg: float, device: str):
    weights = torch.load(checkpoint_path, map_location=device)
    # Sized from the checkpoint, since the ground-truth state carries four readings and the camera state three.
    num_readings = weights["fc1.weight"].shape[1] - PREV_ACTION_DIM - 2
    if num_readings not in (3, 4):
        raise SystemExit(f"q_sweep supports sector-reading checkpoints only, got {num_readings} obstacle inputs")
    q_net = QNetwork(state_dim=num_readings + PREV_ACTION_DIM + 2, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim)
    q_net.load_state_dict(weights)
    q_net.to(device)
    q_net.eval()

    angles_deg = torch.arange(-180.0, 180.0 + 1e-6, angle_step_deg)

    for dist in distances:
        print("=" * 100)
        print(f"distance = {dist:.1f}m   (every obstacle reading 10.0, no-obstacle scenario)")
        print("=" * 100)
        col_headers = "   ".join(f"Q{i}({ACTION_NAMES[i]})" for i in range(cfg.num_actions))
        print(f"{'heading':>8}   {col_headers}   {'argmax':<12} {'top1 margin':>12}")
        print("-" * 100)

        states = []
        for ang_deg in angles_deg:
            ang_rad = math.radians(float(ang_deg))
            dx = dist * math.cos(ang_rad)
            dy = dist * math.sin(ang_rad)
            states.append([10.0] * num_readings + [0.0] * PREV_ACTION_DIM + [dx, dy])  # as on an episode's first step
        state_tensor = torch.tensor(states, dtype=torch.float32, device=device)

        with torch.no_grad():
            q_values = q_net(state_tensor)  # (N, num_actions)

        sorted_q, _ = torch.sort(q_values, dim=-1, descending=True)
        margin = (sorted_q[:, 0] - sorted_q[:, 1])
        best_action = q_values.argmax(dim=-1)

        for i, ang_deg in enumerate(angles_deg):
            q_row = "  ".join(f"{q:>9.3f}" for q in q_values[i].tolist())
            act = ACTION_NAMES[best_action[i].item()]
            print(f"{float(ang_deg):>7.1f}deg  {q_row}   {act:<12} {margin[i].item():>12.3f}")

        pos_mask = angles_deg > 5.0
        neg_mask = angles_deg < -5.0
        turn = best_action % NUM_TURNS  # 1 = left, 2 = right, at either speed
        pos_left_win = (turn[pos_mask] == 1).float().mean().item() * 100 if pos_mask.any() else float("nan")
        neg_right_win = (turn[neg_mask] == 2).float().mean().item() * 100 if neg_mask.any() else float("nan")
        print("-" * 100)
        print(f"[symmetry check] heading > +5 deg -> picked 'turn_left': {pos_left_win:.1f}%")
        print(f"[symmetry check] heading < -5 deg -> picked 'turn_right': {neg_right_win:.1f}%")
        print("[symmetry check] if the two percentages are close (both high or both low),")
        print("                 left/right turn judgment is symmetric; if one is much")
        print("                 higher than the other, there's an asymmetry problem.")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--distances", type=float, nargs="+", default=[1.0, 2.0, 5.0, 10.0],
                         help="fixed distances (meters) to sweep over; defaults cover near/mid/far")
    parser.add_argument("--angle_step_deg", type=float, default=5.0,
                         help="angular sweep step size in degrees, default 5")
    parser.add_argument("--device", type=str, default="cpu")
    add_config_args(parser)
    args = parser.parse_args()
    cfg: DQNConfig = config_from_args(args)

    sweep(args.checkpoint, cfg, args.distances, args.angle_step_deg, args.device)
