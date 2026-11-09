"""
DQN training script -- pure-2D kinematic environment only.

This is a standalone, Isaac-Lab-free version of the DQN training loop,
extracted from the main Drone-Ranger-RL project's Isaac Lab-based
training script. It requires only torch (CPU is fine, no GPU needed) and
runs the same DQN algorithm, network, action space, and reward against
dqn_2d_env.py -- a pure-kinematic 2D navigation environment -- instead of
the full Isaac Lab simulation. This lets anyone (including on a Mac, no
Isaac Sim install required) reproduce and experiment with the DQN
algorithm/reward design in isolation from the simulator.

See the main repository's README for full background on why this 2D
environment exists (isolating the DQN/reward/network design from Isaac
Lab simulation details) and the diagnosed issues it uncovered.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import fields

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dqn_config import DQNConfig, add_config_args, config_from_args
from dqn_action_space import NUM_TURNS, discrete_action_to_continuous, QNetwork
from dqn_2d_env import OUTCOMES, add_env_args, env_from_args
from exploration import HeldRandomExploration
from nstep import NStepAccumulator
from lr_schedule import scheduled_lr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=3000)
parser.add_argument("--max_steps_per_episode", type=int, default=100)
parser.add_argument("--resume_from", type=str, default=None, help="path to a .pt checkpoint to resume from")
parser.add_argument("--resume_optimizer_from", type=str, default=None, help="path to a .pt optimizer state to resume from (paired with --resume_from)")
parser.add_argument("--run_name", type=str, default=None, help="name for this run's checkpoint/tensorboard files; auto-generated from timestamp if omitted")
parser.add_argument("--output_dir", type=str, default=SCRIPT_DIR,
                     help="base directory for checkpoints/ and tensorboard/ subdirectories; "
                          "defaults to this script's own directory")
add_env_args(parser)
add_config_args(parser)
args = parser.parse_args()
cfg: DQNConfig = config_from_args(args)

checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════
# Experience replay
# ══════════════════════════════════════════════════════════
class ReplayBuffer:
    """FIFO ring of preallocated CPU tensors holding the latest `capacity` transitions.

    A deque of per-transition tuples cost about 1.2 KB per transition,
    about 600 MB for a full 500k buffer, against about 200 bytes here.
    """

    def __init__(self, capacity, state_dim, alpha=0.0, eps=1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.s = torch.zeros(capacity, state_dim)
        self.a = torch.zeros(capacity, dtype=torch.long)
        self.r = torch.zeros(capacity)
        self.s_next = torch.zeros(capacity, state_dim)
        self.done = torch.zeros(capacity)
        self.discount = torch.zeros(capacity)  # gamma ** m per row, stays 0 unless an n-step push wrote it, so the one-step target must not read it
        self.priority = torch.zeros(capacity)
        self.max_priority = 1.0  # given to new transitions, so each is drawn at least about as often as any other
        self.pos = 0  # next slot to write
        self.size = 0

    def push(self, s, a, r, s_next, done, discount=None):
        """One row per env, written in env order, as consecutive appends to a deque would be."""
        n = s.shape[0]
        # Only the last `capacity` rows survive, as in a deque. Writing all of a larger batch would give
        # index_put_ duplicate slots, whose result PyTorch leaves undefined.
        first = max(n - self.capacity, 0)
        slots = (self.pos + first + torch.arange(n - first)) % self.capacity
        self.s[slots] = s[first:].cpu()
        self.a[slots] = a[first:].cpu()
        self.r[slots] = r[first:].cpu()
        self.s_next[slots] = s_next[first:].cpu()
        self.done[slots] = done[first:].cpu().float()
        if discount is not None:
            self.discount[slots] = discount[first:].cpu()
        self.priority[slots] = self.max_priority
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        # random.sample picks positions from the length alone, so drawing from range(size) and counting
        # from the oldest slot selects the same transitions random.sample over a deque of them would.
        oldest = (self.pos - self.size) % self.capacity
        slots = (torch.tensor(random.sample(range(self.size), batch_size)) + oldest) % self.capacity
        return (self.s[slots], self.a[slots].to(device), self.r[slots].to(device),
                self.s_next[slots], self.done[slots].to(device), self.discount[slots].to(device))

    def sample_prioritized(self, batch_size, beta):
        """Draw slots with probability proportional to priority, with replacement.

        Returns the batch, then the slots (for update_priorities) and importance-sampling weights whose largest over the buffer is 1."""
        # Slots fill from 0 and wrap only once full, so the first `size` slots are exactly the stored ones.
        stored = self.priority[:self.size].double()
        # float64 because a float32 running sum near 5e5 cannot resolve priorities near per_eps ** per_alpha.
        cum = torch.cumsum(stored, 0)
        draws = torch.rand(batch_size, dtype=torch.float64) * cum[-1]
        slots = torch.searchsorted(cum, draws, right=True).clamp_(max=self.size - 1)
        # Scaling by the batch's own largest weight would leave the sampling bias in place even at beta 1.
        weights = (self.priority[slots].double() / stored.min()) ** -beta
        weights = weights.float().to(device)
        return (self.s[slots], self.a[slots].to(device), self.r[slots].to(device),
                self.s_next[slots], self.done[slots].to(device), self.discount[slots].to(device), slots, weights)

    def update_priorities(self, slots, td_error):
        # A slot drawn twice in one batch gets the same TD error both times, so duplicate writes agree.
        p = (td_error.abs().cpu() + self.eps) ** self.alpha
        self.priority[slots] = p
        self.max_priority = max(self.max_priority, p.max().item())

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════
# Set up environment, network, optimizer
# ══════════════════════════════════════════════════════════
env = env_from_args(args, args.num_envs, cfg, device)

q_net = QNetwork(state_dim=env.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
q_target = QNetwork(state_dim=env.state_dim, action_dim=cfg.num_actions, hidden_dim=cfg.hidden_dim).to(device)
if args.resume_from:
    q_net.load_state_dict(torch.load(args.resume_from, map_location=device))
    print(f"[DQN] resumed weights from {args.resume_from}")

q_target.load_state_dict(q_net.state_dict())
if cfg.weight_decay > 0:
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
else:
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.learning_rate)
if args.resume_optimizer_from:
    optimizer.load_state_dict(torch.load(args.resume_optimizer_from, map_location=device))
    # A loaded state carries the decay it was saved with, so the run's own value is put back.
    for group in optimizer.param_groups:
        group["weight_decay"] = cfg.weight_decay
    print(f"[DQN] resumed optimizer state from {args.resume_optimizer_from}")
elif args.resume_from:
    print("[DQN] resumed weights but NOT optimizer state (no --resume_optimizer_from given) -- Adam's per-parameter momentum starts fresh this run")
replay_buffer = ReplayBuffer(cfg.replay_buffer_size, env.state_dim, cfg.per_alpha, cfg.per_eps)
use_per = cfg.per_alpha > 0
# None keeps the one-step path exactly as before, so n_step 1 trains bitwise unchanged.
nstep = (NStepAccumulator(cfg.n_step, cfg.gamma, args.num_envs, env.state_dim, device)
         if cfg.n_step > 1 else None)

epsilon = cfg.resume_epsilon if args.resume_from else cfg.epsilon_start
explorer = HeldRandomExploration(args.num_envs, cfg.num_actions, cfg.explore_hold_mu, cfg.explore_max_hold, device)
episode_returns = []

run_name = args.run_name or time.strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(tensorboard_dir, run_name)
writer = SummaryWriter(log_dir=log_dir)
print("=" * 70)
print("[DQN] Starting training (2D kinematic environment, standalone)")
print("=" * 70)
print(f"  device          : {device}")
print(f"  num_envs        : {args.num_envs}")
print(f"  num_episodes    : {args.num_episodes}")
print(f"  max_steps/ep    : {args.max_steps_per_episode}")
print(f"  num_obstacles   : {args.num_obstacles}")
print(f"  distance_mode   : {args.distance_mode}")
print(f"  state_dim       : {env.state_dim}")
print(f"  hidden_dim      : {cfg.hidden_dim}")
print(f"  resume_from     : {args.resume_from or '(from scratch)'}")
print(f"  checkpoints dir : {checkpoints_dir}")
print(f"  tensorboard log : {log_dir}")
print("=" * 70)
train_start_time = time.time()

best_ckpt = os.path.join(checkpoints_dir, f"{run_name}_best.pt")
best_opt = os.path.join(checkpoints_dir, f"{run_name}_optimizer_best.pt")
best_score = os.path.join(checkpoints_dir, f"{run_name}_best.json")
# select_best.py carries the best score over from this file, so a fresh run must not inherit an old one.
if os.path.exists(best_score) and not args.resume_from:
    raise RuntimeError(f"{best_score} already exists, pick a new --run_name or --output_dir")
select_cmd = ([sys.executable, os.path.join(SCRIPT_DIR, "select_best.py"), "--best", best_ckpt,
               "--num_obstacles", str(args.num_obstacles), "--distance_mode", args.distance_mode,
               "--k_nearest", str(args.k_nearest)]
              + [x for f in fields(DQNConfig) for x in (f"--{f.name}", str(getattr(cfg, f.name)))])


def _best_success():
    if not os.path.exists(best_score):
        return None
    with open(best_score) as f:
        return json.load(f)["best_success"]


def keep_if_best(ckpt_path):
    """Score ckpt_path in a separate process, so this run's random numbers are
    untouched, and keep it with the matching optimizer state if it is the best so far."""
    before = _best_success()
    result = subprocess.run(select_cmd + ["--checkpoint", ckpt_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"select_best.py failed on {ckpt_path}:\n{result.stderr}")
    print(result.stdout.strip())
    # A rise in the recorded score, not the recorded file name, which a resumed run reuses.
    if before is None or _best_success() > before:
        torch.save(optimizer.state_dict(), best_opt)


obs_dict, _ = env.reset()
state = obs_dict["dqn_state"]

episode_return = torch.zeros(args.num_envs, device=device)

WINDOW_SIZE = 200
outcome_window = deque(maxlen=WINDOW_SIZE)
global_train_step = 0

for episode in range(args.num_episodes):
    if cfg.lr_final_fraction != 1.0:
        lr = scheduled_lr(cfg.learning_rate, cfg.lr_final_fraction, cfg.lr_decay_start, episode, args.num_episodes)
        for group in optimizer.param_groups:
            group["lr"] = lr
    for step in range(args.max_steps_per_episode):
        with torch.no_grad():
            q_values = q_net(state)
            greedy_action = q_values.argmax(dim=-1)
            action_idx = explorer.select(greedy_action, epsilon)

        actions_continuous = discrete_action_to_continuous(action_idx, device, cfg)

        step_in_episode = env.step_count + 1  # read before step(), which resets the count of ended envs
        obs_dict, reward, terminated, truncated, info = env.step(actions_continuous)
        next_state = obs_dict["dqn_state"]
        done = terminated | truncated
        # Only reaching the target or crashing is terminal. A timeout or leaving the
        # bounds just cuts the episode off, so its target still bootstraps from next_state.
        terminal = terminated

        if nstep is None:
            keep = ((step_in_episode <= cfg.replay_full_steps) | terminal
                    | (torch.rand(args.num_envs, device=device) < cfg.replay_late_fraction))
            if cfg.replay_keep_stopped:
                keep |= action_idx < NUM_TURNS
            replay_buffer.push(state[keep], action_idx[keep], reward[keep], next_state[keep], terminal[keep])
        else:
            # Windows are built from every raw step, and only the emitted n-step transitions are
            # subsampled, judged by the step their first transition was taken at.
            env_i, s_n, a_n, r_n, s_next_n, term_n, disc_n, start_n = nstep.add(
                state, action_idx, reward, next_state, terminal, done, step_in_episode)
            late = torch.rand(args.num_envs, device=device)[env_i] < cfg.replay_late_fraction
            keep = (start_n <= cfg.replay_full_steps) | term_n | late
            if cfg.replay_keep_stopped:
                keep |= a_n < NUM_TURNS
            replay_buffer.push(s_n[keep], a_n[keep], r_n[keep], s_next_n[keep], term_n[keep], disc_n[keep])

        if done.any():
            outcomes = env.last_outcomes()
            for i in done.nonzero(as_tuple=True)[0].tolist():
                outcome_window.append(outcomes[i])

        episode_return += reward
        state = info["reset_obs"]["dqn_state"]

        if len(replay_buffer) >= cfg.batch_size:
            if use_per:
                beta = cfg.per_beta_start + (1 - cfg.per_beta_start) * episode / max(args.num_episodes - 1, 1)
                s, a, r, s_next, d, disc, slots, is_weight = replay_buffer.sample_prioritized(cfg.batch_size, beta)
            else:
                s, a, r, s_next, d, disc = replay_buffer.sample(cfg.batch_size)
            s, s_next = s.to(device), s_next.to(device)

            q_pred = q_net(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                # Double DQN: the online net picks the next joint action, the target net scores it.
                next_a = q_net(s_next).argmax(dim=-1, keepdim=True)
                q_next_max = q_target(s_next).gather(1, next_a).squeeze(-1)
                q_target_value = r + (cfg.gamma if nstep is None else disc) * q_next_max * (1 - d)
            if use_per:
                td_error = q_target_value - q_pred
                loss = (is_weight * td_error.pow(2)).mean()
            else:
                loss = F.mse_loss(q_pred, q_target_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_per:
                replay_buffer.update_priorities(slots, td_error.detach())
            if cfg.target_tau > 0:
                with torch.no_grad():
                    for p_t, p in zip(q_target.parameters(), q_net.parameters()):
                        p_t.lerp_(p, cfg.target_tau)

            global_train_step += 1
            if global_train_step % 20 == 0:
                writer.add_scalar("Loss/td_error", loss.item(), global_train_step)
                writer.add_scalar("Loss/q_pred_mean", q_pred.mean().item(), global_train_step)
                writer.add_scalar("Loss/q_pred_max", q_pred.max().item(), global_train_step)
                writer.add_scalar("Loss/q_target_mean", q_target_value.mean().item(), global_train_step)

    mean_return = episode_return.mean().item()
    episode_returns.append(mean_return)
    episode_return.zero_()

    epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

    if cfg.target_tau == 0 and episode % cfg.target_update_every == 0:
        q_target.load_state_dict(q_net.state_dict())

    recent = episode_returns[-50:]
    ma50 = sum(recent) / len(recent)
    writer.add_scalar("Train/mean_reward", mean_return, episode)
    writer.add_scalar("Train/mean_reward_ma50", ma50, episode)
    writer.add_scalar("Train/epsilon", epsilon, episode)
    writer.add_scalar("Train/learning_rate", optimizer.param_groups[0]["lr"], episode)
    writer.add_scalar("Train/buffer_size", len(replay_buffer), episode)

    total_outcomes = len(outcome_window)
    counts = Counter(outcome_window)
    pct = {k: counts[k] / total_outcomes * 100 if total_outcomes else 0.0 for k in OUTCOMES}
    if total_outcomes > 0:
        for k in OUTCOMES:
            writer.add_scalar(f"termination/{k}", pct[k], episode)

    if episode % 10 == 0:
        elapsed = time.time() - train_start_time
        eps_per_sec = (episode + 1) / elapsed if elapsed > 0 else 0
        remaining = (args.num_episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
        print(f"[DQN] episode {episode:>4}/{args.num_episodes}  "
              f"return={mean_return:>7.2f}  MA50={ma50:>7.2f}  epsilon={epsilon:.3f}  "
              f"buffer={len(replay_buffer):>5}  elapsed={elapsed/60:>5.1f}min  ETA={remaining/60:>5.1f}min")
        print(f"       [last {total_outcomes} terminations] "
              + "  ".join(f"{k}={pct[k]:>5.1f}%" for k in OUTCOMES))

    if episode % 100 == 0 and episode > 0:
        ckpt_path = os.path.join(checkpoints_dir, f"{run_name}_ep{episode}.pt")
        torch.save(q_net.state_dict(), ckpt_path)
        print("-" * 70)
        print(f"[DQN] checkpoint saved: {ckpt_path}")
        keep_if_best(ckpt_path)
        print("-" * 70)

print("=" * 70)
final_ckpt = os.path.join(checkpoints_dir, f"{run_name}_final.pt")
final_opt = os.path.join(checkpoints_dir, f"{run_name}_optimizer_final.pt")
torch.save(q_net.state_dict(), final_ckpt)
torch.save(optimizer.state_dict(), final_opt)
keep_if_best(final_ckpt)
total_elapsed = time.time() - train_start_time
print(f"[DQN] Training complete. Final checkpoint saved: {final_ckpt}")
print(f"[DQN] Optimizer state saved: {final_opt}")
print(f"[DQN] Best checkpoint: {best_ckpt}")
print(f"[DQN] Total time: {total_elapsed/60:.1f} minutes")
print(f"[DQN] tensorboard: tensorboard --logdir {tensorboard_dir}")
print("=" * 70)
