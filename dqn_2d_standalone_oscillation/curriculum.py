"""
Trains one policy through a sequence of obstacle counts, each stage resuming
the previous stage's best checkpoint with weights only, so every stage starts
with a fresh Adam state, an empty replay buffer and epsilon at resume_epsilon.

    python3 curriculum.py --init <stage-0 checkpoint> --output_root runs/my_chain

The default stages are 1, 2, 3 and 5 obstacles for 1500, 1500, 3000 and 3000
episodes. Pass --stages to change them, and include a 0 stage with no --init
for an experiment whose state size differs from the baseline, since its
network cannot load the shared stage-0 weights. Every argument this script
does not know is passed on to train_dqn.py unchanged.

When several checkpoints of a stage tie on the validation score, they are
rescored on a second seed and the winner replaces <stage>_best.pt and the
best_checkpoint entry in <stage>_best.json, so every later tool picks the
checkpoint that was handed on. <output_root>/curriculum.json records the
starting weights with their SHA-256, the stages, the forwarded arguments and
each stage's handoff.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_2d_env import add_env_args
from select_best import first_episode_success

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIE_SEED = 2000  # differs from select_best's VALIDATION_SEED and from any report seed
# train_dqn.py options this script sets per stage, so a forwarded copy would conflict.
# --resume_optimizer_from is refused too, since forwarding it would break the weights-only handoff.
OWNED_OPTIONS = ("--num_obstacles", "--num_episodes", "--resume_from", "--resume_optimizer_from", "--output_dir", "--run_name")


def parse_stages(text):
    stages = []
    for item in text.split(","):
        obstacles, episodes = item.split(":")
        stages.append((int(obstacles), int(episodes)))
    return stages


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def break_tie(stage_dir, run_name, num_obstacles, forwarded, tie_episodes):
    """Rescore the checkpoints tied for the stage's best on TIE_SEED and make the winner the stage's best.
    Returns the tie record, or None when a single checkpoint holds the best score."""
    ckpt_dir = os.path.join(stage_dir, "checkpoints")
    score_path = os.path.join(ckpt_dir, f"{run_name}_best.json")
    with open(score_path) as f:
        record = json.load(f)
    tied = [h["checkpoint"] for h in record["history"] if h["success"] == record["best_success"]]
    if len(tied) < 2:
        return None

    parser = argparse.ArgumentParser()
    add_env_args(parser)
    add_config_args(parser)
    env_args, _ = parser.parse_known_args(forwarded + ["--num_obstacles", str(num_obstacles)])
    cfg: DQNConfig = config_from_args(env_args)
    scores = {c: first_episode_success(os.path.join(ckpt_dir, c), env_args, cfg, tie_episodes, seed=TIE_SEED)
              for c in tied}
    winner = max(tied, key=lambda c: scores[c])  # max keeps the first of equal scores, the earliest checkpoint
    tie = {"tied": tied, "tie_seed": TIE_SEED, "tie_episodes": tie_episodes, "scores": scores,
           "selected_by_validation": record["best_checkpoint"], "winner": winner}
    shutil.copyfile(os.path.join(ckpt_dir, winner), os.path.join(ckpt_dir, f"{run_name}_best.pt"))
    # Only the validation pick's optimizer state was saved, so it cannot be paired with another winner.
    stale_opt = os.path.join(ckpt_dir, f"{run_name}_optimizer_best.pt")
    if winner != record["best_checkpoint"] and os.path.exists(stale_opt):
        os.remove(stale_opt)
    record["best_checkpoint"] = winner
    record["tie_break"] = tie
    with open(score_path, "w") as f:
        json.dump(record, f, indent=1)
    return tie


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_root", type=str, required=True, help="each stage writes to <output_root>/s<obstacles>")
    parser.add_argument("--init", type=str, default=None, help="weights the first stage resumes from, omitted to train it from scratch")
    parser.add_argument("--stages", type=str, default="1:1500,2:1500,3:3000,5:3000",
                        help="comma-separated obstacles:episodes pairs, run in order")
    parser.add_argument("--tie_episodes", type=int, default=10000)
    args, forwarded = parser.parse_known_args()
    # train_dqn.py's argparse accepts abbreviations, so any prefix of an owned option counts as that option.
    clash = [o for o in forwarded if o.startswith("--") and any(own.startswith(o.split("=")[0]) for own in OWNED_OPTIONS)]
    if clash:
        parser.error(f"curriculum.py sets {', '.join(OWNED_OPTIONS)} itself, got {clash}")
    stages = parse_stages(args.stages)
    counts = [k for k, _ in stages]
    if len(set(counts)) != len(counts):
        parser.error(f"each obstacle count names one stage directory, so it may appear once, got {counts}")
    # train_dqn.py runs with cwd=SCRIPT_DIR, so relative paths would resolve differently there.
    args.output_root = os.path.abspath(args.output_root)
    args.init = os.path.abspath(args.init) if args.init else None
    # A resumed stage passes train_dqn.py's own guard, so an old score file would carry its best into this chain.
    used = [os.path.join(args.output_root, f"s{k}") for k, _ in stages
            if os.path.exists(os.path.join(args.output_root, f"s{k}", "checkpoints", f"s{k}_best.json"))]
    if used:
        parser.error(f"stage directories already hold a run, pick a new --output_root: {used}")

    os.makedirs(args.output_root, exist_ok=True)
    lineage_path = os.path.join(args.output_root, "curriculum.json")
    lineage = {"init": args.init,
               "init_sha256": sha256(args.init) if args.init else None,
               "stages": [{"num_obstacles": k, "num_episodes": e} for k, e in stages],
               "forwarded_args": forwarded, "handoffs": []}

    prev_best = args.init
    for num_obstacles, episodes in stages:
        run_name = f"s{num_obstacles}"
        stage_dir = os.path.join(args.output_root, run_name)
        os.makedirs(stage_dir, exist_ok=True)
        cmd = [sys.executable, os.path.join(SCRIPT_DIR, "train_dqn.py"), "--output_dir", stage_dir, "--run_name", run_name,
               "--num_obstacles", str(num_obstacles), "--num_episodes", str(episodes)] + forwarded
        if prev_best is not None:
            cmd += ["--resume_from", prev_best]
        with open(os.path.join(stage_dir, "train.log"), "w") as log:
            rc = subprocess.run(cmd, cwd=SCRIPT_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            sys.exit(f"{run_name} failed with exit code {rc}, see {os.path.join(stage_dir, 'train.log')}")

        tie = break_tie(stage_dir, run_name, num_obstacles, forwarded, args.tie_episodes)
        best = os.path.join(stage_dir, "checkpoints", f"{run_name}_best.pt")
        with open(os.path.join(stage_dir, "checkpoints", f"{run_name}_best.json")) as f:
            record = json.load(f)
        lineage["handoffs"].append({"stage": run_name, "resumed_from": prev_best, "best_checkpoint": record["best_checkpoint"],
                                    "best_validation_success": record["best_success"], "tie_break": tie,
                                    "best_sha256": sha256(best)})
        with open(lineage_path, "w") as f:
            json.dump(lineage, f, indent=1)
        prev_best = best
        print(f"{run_name} done, best = {best} ({record['best_checkpoint']})", flush=True)


if __name__ == "__main__":
    main()
