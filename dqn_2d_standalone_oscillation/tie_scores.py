"""
Scores the checkpoints tied for a stage's best for curriculum.py and prints
{checkpoint: success} as one JSON line. curriculum.py runs it as a child
process, so torch and the env load only during the tie-break rather than
for the whole chain.

    python3 tie_scores.py <checkpoint dir> <episodes> <seed> <num_obstacles> <checkpoint>... -- <forwarded args>

The forwarded arguments are curriculum.py's, parsed for the env and config
options exactly as train_dqn.py sees them.
"""
import argparse
import json
import os
import sys

from dqn_config import add_config_args, config_from_args
from dqn_2d_env import add_env_args
from select_best import first_episode_success


def main():
    sep = sys.argv.index("--")
    ckpt_dir, episodes, seed, num_obstacles, *tied = sys.argv[1:sep]
    parser = argparse.ArgumentParser()
    add_env_args(parser)
    add_config_args(parser)
    env_args, _ = parser.parse_known_args(sys.argv[sep + 1:] + ["--num_obstacles", num_obstacles])
    cfg = config_from_args(env_args)
    scores = {c: first_episode_success(os.path.join(ckpt_dir, c), env_args, cfg, int(episodes), seed=int(seed))
              for c in tied}
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
