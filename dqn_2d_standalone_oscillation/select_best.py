"""
Scores one checkpoint on a fixed validation set and keeps it as the run's best
checkpoint when it beats every checkpoint scored before it. train_dqn.py calls
this in a separate process each time it saves a checkpoint, so the scoring
draws none of the training run's random numbers.

Validation is the greedy first episode of each env on layouts drawn from
VALIDATION_SEED, so every checkpoint of every run faces the same layouts. Use a
different seed for any final report, or the reported score is biased upward by
the selection itself.
"""
import argparse
import json
import os
import shutil

import torch

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import discrete_action_to_continuous, QNetwork
from dqn_2d_env import add_env_args, env_from_args

VALIDATION_SEED = 1000


def first_episode_success(checkpoint, args, cfg, num_episodes):
    env = env_from_args(args, num_episodes, cfg, "cpu")
    q_net = QNetwork(state_dim=env.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim)
    q_net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    q_net.eval()
    # Seeded after the network is built, as in eval_2d.py, so layouts depend on the seed and env options only.
    torch.manual_seed(VALIDATION_SEED)
    obs, _ = env.reset()
    state = obs["dqn_state"]
    open_ = torch.ones(num_episodes, dtype=torch.bool)
    reached = torch.zeros(num_episodes, dtype=torch.bool)
    for _ in range(env.max_steps):
        with torch.no_grad():
            action_idx = q_net(state).argmax(dim=-1)
        obs, _, terminated, truncated, info = env.step(discrete_action_to_continuous(action_idx, "cpu"))
        state = info["reset_obs"]["dqn_state"]
        done = (terminated | truncated) & open_
        reached |= done & env._last_reached_flag
        open_ &= ~done
        if not open_.any():
            break
    return reached.float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--best", type=str, required=True, help="where the best checkpoint is kept, its score sits beside it as .json")
    parser.add_argument("--num_episodes", type=int, default=1000)
    add_env_args(parser)
    add_config_args(parser)
    args = parser.parse_args()
    cfg: DQNConfig = config_from_args(args)

    score_path = os.path.splitext(args.best)[0] + ".json"
    record = {"best_checkpoint": None, "best_success": -1.0, "history": []}
    if os.path.exists(score_path):
        with open(score_path) as f:
            record = json.load(f)

    success = first_episode_success(args.checkpoint, args, cfg, args.num_episodes)
    record["history"].append({"checkpoint": os.path.basename(args.checkpoint), "success": success})
    # Strictly better only, so a tie keeps the earlier checkpoint.
    improved = success > record["best_success"]
    if improved:
        shutil.copyfile(args.checkpoint, args.best)
        record["best_checkpoint"] = os.path.basename(args.checkpoint)
        record["best_success"] = success
    with open(score_path, "w") as f:
        json.dump(record, f, indent=1)
    print(f"[BEST] {os.path.basename(args.checkpoint)} success={success * 100:.1f}%  "
          f"best={record['best_checkpoint']} ({record['best_success'] * 100:.1f}%){'  NEW BEST' if improved else ''}")


if __name__ == "__main__":
    main()
