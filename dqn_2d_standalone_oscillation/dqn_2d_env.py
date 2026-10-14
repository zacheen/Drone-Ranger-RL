"""
Pure-2D, Isaac-Lab-free training environment -- isolates the DQN/reward/
network design from any Isaac Lab simulation detail (physics stepping,
attitude controller dynamics, camera rendering) to test whether the
oscillation/low-margin problem seen in q_sweep.py is caused by the DQN
side of things or by something in the Isaac Lab simulation layer.

Kinematics only: no thrust, no attitude, no gravity. Position and heading
are integrated directly from the same discrete action -> (vx, yaw_rate)
mapping used by the real environment, at the same control frequency
(dt=0.02s, matching Isaac Lab's sim.dt=1/100 * decimation=2).

State layout is identical to the real environment:
    state = [d_front, d_left, d_right, dx, dy]
With distance_mode="ground_truth" (the default), d_front/d_left/d_right are
exact distances to the nearest obstacle surface in each direction
(obstacle_geometry.py). With distance_mode="camera" they are what the Isaac
env's three depth cameras would report (camera_emulation.py). With
num_obstacles=0 (Stage 0) both modes read 10.0 (max sensing range) in all
three, what the real environment's cameras report when nothing is in view.

distance_mode="coords" is a diagnostic upper bound with no Isaac
counterpart. It replaces the three readings with k_nearest slots
(sx, sy, visible), the nearest surface point of each of the k_nearest
closest obstacles in the body frame, nearest first:
    state = [s1_x, s1_y, v1, ..., sK_x, sK_y, vK, dx, dy]
Its norm is the exact distance, with no sector and so no blind spot. An
obstacle beyond MAX_SENSE_RANGE or parked leaves its slot all zeros, so the
input size depends on k_nearest only and stays fixed across num_obstacles.

num_obstacles=1..5 (Stage 1) ports the layout from the Isaac env's
_reset_idx: obstacles strung along a random flight direction, the target in
a gap between them or past them, and rehearsal mixing. Its hard-case
injection is not ported because the Isaac env runs it with hard_case_frac=0.
Touching an obstacle ends the episode as a crash.

Interface intentionally mirrors gymnasium's reset()/step() so train_dqn.py
needs minimal changes to switch environments.
"""
import math

import torch

from dqn_config import DQNConfig
from camera_emulation import FAR_CLIP, TAN_HALF_FOV, camera_readings
from obstacle_geometry import _nearest_surface, obstacle_sector_distances, obstacle_surface_distance

DT = 0.02  # seconds per control step, matches Isaac Lab's 50Hz control rate
MAX_SENSE_RANGE = FAR_CLIP  # both distance modes cap at the cameras' far clip
EPISODE_LENGTH_S = 20.0  # matches the real env's episode_length_s
TARGET_POS_RANGE = 5.0   # start/target sampled uniformly in [-5, 5] on each axis
REACHED_THRESHOLD = 0.75
OUT_OF_BOUNDS_DIST = 40.0  # matches the real env's widened boundary

# Obstacle roster in the Isaac env's order, three upright cylinders and then
# two axis-aligned boxes, with the physical footprints its _setup_scene spawns.
CYLINDER_RADII = (0.3, 0.3, 0.4)
BOX_HALF_EXTENTS = ((0.5, 0.5), (0.75, 0.75))
NUM_CYLINDERS = len(CYLINDER_RADII)
MAX_OBSTACLES = NUM_CYLINDERS + len(BOX_HALF_EXTENTS)
DRONE_RADIUS = 0.3  # prop span, a crash once the drone's center is this close to a surface
# Layout-only clearance radii, copied from the Isaac env's _obstacle_radius
# (footprint bound + DRONE_RADIUS). Its 0.8 for Cyl0 exceeds the 0.3 radius
# that cylinder physically has.
PLACEMENT_RADII = tuple(r + DRONE_RADIUS for r in (0.8, 0.3, 0.4, 0.707, 1.06))
OBSTACLE_MIN_SPACING = 2.5  # along-path distance between consecutive obstacles
REHEARSAL_PROB = 0.5
SENSOR_HALF_FOV = math.atan(TAN_HALF_FOV)  # half of the cameras' 73 deg HFOV
DISTANCE_MODES = ("ground_truth", "camera", "coords")
DEFAULT_K_NEAREST = 3

# Episode outcomes, highest priority first, for a step that meets several conditions.
OUTCOMES = ("reached", "contact_crash", "out_of_bounds", "timed_out")


class DroneNav2DEnv:
    def __init__(self, num_envs: int, cfg: DQNConfig, device: str, num_obstacles: int = 0,
                 distance_mode: str = "ground_truth", k_nearest: int = DEFAULT_K_NEAREST):
        if not 0 <= num_obstacles <= MAX_OBSTACLES:
            raise ValueError(f"num_obstacles must be in [0, {MAX_OBSTACLES}], got {num_obstacles}")
        if distance_mode not in DISTANCE_MODES:
            raise ValueError(f"distance_mode must be one of {DISTANCE_MODES}, got {distance_mode!r}")
        if not 1 <= k_nearest <= MAX_OBSTACLES:
            raise ValueError(f"k_nearest must be in [1, {MAX_OBSTACLES}], got {k_nearest}")
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.num_obstacles = num_obstacles
        self.distance_mode = distance_mode
        self.k_nearest = k_nearest
        # Networks are sized from this, not from cfg.state_dim, because coords mode changes it.
        self.state_dim = 3 * k_nearest + 2 if distance_mode == "coords" else 5
        self.max_steps = int(EPISODE_LENGTH_S / DT)

        self.pos = torch.zeros(num_envs, 2, device=device)      # (N, 2) x,y
        self.yaw = torch.zeros(num_envs, device=device)          # (N,) radians
        self.target = torch.zeros(num_envs, 2, device=device)    # (N, 2)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)

        self._circle_r = torch.tensor(CYLINDER_RADII, device=device)
        self._box_half = torch.tensor(BOX_HALF_EXTENTS, device=device)
        self._placement_r = torch.tensor(PLACEMENT_RADII, device=device)
        # Inactive obstacles are parked where the Isaac env parks them, far
        # beyond sensing range, so they never register in a reading.
        self._parked_xy = torch.stack([
            1000.0 + torch.arange(MAX_OBSTACLES, device=device, dtype=torch.float32),
            torch.full((MAX_OBSTACLES,), 1000.0, device=device),
        ], dim=-1)
        self.obstacle_xy = self._parked_xy.expand(num_envs, -1, -1).clone()  # (N, MAX_OBSTACLES, 2)
        # False for obstacles parked this episode, by num_obstacles or by rehearsal mixing.
        self.obstacle_active = torch.zeros(num_envs, MAX_OBSTACLES, dtype=torch.bool, device=device)

    def _sample_positions(self, env_ids):
        n = len(env_ids)
        # Matches the real environment exactly: the drone starts near the
        # origin with only a small perturbation (uniform +/-1.0m), while
        # the TARGET is sampled across the full +/-5m box. Getting this
        # wrong (e.g. randomizing both drone and target across the same
        # +/-5m range) would give a different start-to-target distance
        # distribution than Isaac Lab's, making any comparison between the
        # two environments' q_sweep/margin results invalid.
        self.pos[env_ids] = torch.zeros(n, 2, device=self.device).uniform_(-1.0, 1.0)
        # The Stage 0 branch draws random numbers in the same order as before
        # obstacles existed, so a seeded Stage 0 run reproduces older runs.
        if self.num_obstacles == 0:
            self.target[env_ids] = (torch.rand(n, 2, device=self.device) * 2 - 1) * TARGET_POS_RANGE
        else:
            target, obstacle_xy, active = self._sample_obstacle_layout(self.pos[env_ids])
            self.target[env_ids] = target
            self.obstacle_xy[env_ids] = obstacle_xy
            self.obstacle_active[env_ids] = active
        self.yaw[env_ids] = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        self.step_count[env_ids] = 0

    def _sample_obstacle_layout(self, start_xy):
        """The Isaac env's obstacle and target placement for num_obstacles
        active obstacles. Returns target (n, 2), obstacle centers
        (n, MAX_OBSTACLES, 2) with the inactive ones parked, and the
        (n, MAX_OBSTACLES) active mask."""
        n, k, dev = start_xy.shape[0], self.num_obstacles, self.device
        heading = torch.rand(n, device=dev) * 2 * math.pi
        path_dir = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
        perp_dir = torch.stack([-path_dir[:, 1], path_dir[:, 0]], dim=-1)

        jitter = OBSTACLE_MIN_SPACING * 0.4
        along = (2.5 + torch.arange(MAX_OBSTACLES, device=dev) * OBSTACLE_MIN_SPACING
                 + torch.empty(n, MAX_OBSTACLES, device=dev).uniform_(-jitter, jitter))
        lateral = torch.empty(n, MAX_OBSTACLES, device=dev).uniform_(-3.0, 3.0)

        # Target at the middle of a random gap, between two along-path
        # neighbors, wide enough to fly through. Its lateral position then
        # depends on the layout, so one wide swing around everything misses it.
        has_gap = torch.zeros(n, dtype=torch.bool, device=dev)
        gap_along = torch.zeros(n, device=dev)
        gap_lateral = torch.zeros(n, device=dev)
        if k >= 2:
            order = torch.argsort(along[:, :k], dim=-1)
            s_along = along[:, :k].gather(1, order)
            s_lat = lateral[:, :k].gather(1, order)
            s_r = self._placement_r[:k][order]
            valid = (s_lat[:, 1:] - s_lat[:, :-1]).abs() > s_r[:, :-1] + s_r[:, 1:] + 0.3
            has_gap = valid.any(dim=-1)
            score = torch.where(valid, torch.rand(n, k - 1, device=dev), torch.full((n, k - 1), -1.0, device=dev))
            g = score.argmax(dim=-1, keepdim=True)
            gap_along = (s_along.gather(1, g) + s_along.gather(1, g + 1)).squeeze(1) / 2.0
            gap_lateral = (s_lat.gather(1, g) + s_lat.gather(1, g + 1)).squeeze(1) / 2.0

        # Without a usable gap, the target goes 0 to 10 m beyond the last obstacle's
        # clearance radius plus 1 m.
        near = along[:, k - 1] + self._placement_r[k - 1] + 1.0
        fallback_along = torch.rand(n, device=dev) * 10.0 + near
        if k == 1:
            # 0.4 to 1.2 m to a random side of the lone obstacle keeps it on the
            # direct path without making both ways around it equally good.
            side = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
            fallback_lateral = lateral[:, 0] + side * torch.empty(n, device=dev).uniform_(0.4, 1.2)
        else:
            fallback_lateral = torch.empty(n, device=dev).uniform_(-1.0, 1.0)
        target_along = torch.where(has_gap, gap_along, fallback_along)
        target_lateral = torch.where(has_gap, gap_lateral, fallback_lateral)
        target = start_xy + target_along.unsqueeze(-1) * path_dir + target_lateral.unsqueeze(-1) * perp_dir

        obstacles = (start_xy.unsqueeze(1) + along.unsqueeze(-1) * path_dir.unsqueeze(1)
                     + lateral.unsqueeze(-1) * perp_dir.unsqueeze(1))
        obstacles[:, k:] = self._parked_xy[k:]
        # Rehearsal mixing parks obstacle 0 on some resets, so the replay
        # buffer keeps pure-navigation transitions (catastrophic forgetting,
        # main README finding 2). The target keeps the spot obstacle 0 gave it.
        rehearsal = (torch.rand(n, device=dev) < REHEARSAL_PROB).unsqueeze(-1)
        obstacles[:, 0] = torch.where(rehearsal, torch.full_like(obstacles[:, 0], 2000.0), obstacles[:, 0])
        active = (torch.arange(MAX_OBSTACLES, device=dev) < k).expand(n, -1).clone()
        active[:, 0] &= ~rehearsal[:, 0]

        # Push the target out of each obstacle's clearance radius plus a 0.5 m
        # arrival margin. One obstacle at a time, as in the Isaac env, so a push
        # can land inside another obstacle's radius on a crowded layout. Over
        # 50000 resets per k, with k >= 3 that left about 0.03% of targets
        # within DRONE_RADIUS of a surface (closest 0.23 m) and none inside one.
        for i in range(MAX_OBSTACLES):
            safety = self._placement_r[i] + 0.5
            diff = target - obstacles[:, i]
            d = diff.norm(dim=-1, keepdim=True)
            pushed = obstacles[:, i] + diff / d.clamp(min=1e-6) * safety
            target = torch.where(d < safety, pushed, target)
        return target, obstacles, active

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_positions(env_ids)
        obs_dict, _dist = self._get_obs()  # discard dist here; step() re-derives it
        return obs_dict, {}

    def _get_obs(self):
        rel = self.target - self.pos  # (N, 2) world-frame offset to target
        dist = torch.norm(rel, dim=-1)
        # Rotate the world-frame offset into the body frame to get dx, dy
        cos_yaw = torch.cos(self.yaw)
        sin_yaw = torch.sin(self.yaw)
        dx = rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
        dy = -rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw

        obstacles = (self.obstacle_xy[:, :NUM_CYLINDERS], self._circle_r,
                     self.obstacle_xy[:, NUM_CYLINDERS:], self._box_half)
        if self.distance_mode == "coords":
            if self.num_obstacles == 0:
                d = torch.zeros(self.num_envs, 3 * self.k_nearest, device=self.device)
            else:
                d = self._nearest_obstacle_points(obstacles)
        elif self.num_obstacles == 0:
            # Every obstacle is parked, so either mode would return this, without
            # its cost on every Stage 0 step.
            d = torch.full((self.num_envs, 3), MAX_SENSE_RANGE, device=self.device)
        elif self.distance_mode == "ground_truth":
            d = obstacle_sector_distances(self.pos, self.yaw, *obstacles, SENSOR_HALF_FOV, MAX_SENSE_RANGE)
        else:
            d = camera_readings(self.pos, self.yaw, *obstacles)

        state = torch.cat([d, dx.unsqueeze(-1), dy.unsqueeze(-1)], dim=-1)  # [obstacle inputs..., dx, dy]
        return {"dqn_state": state}, dist

    def _nearest_obstacle_points(self, obstacles):
        """(N, 3 * k_nearest) coords-mode slots (sx, sy, visible), nearest first."""
        dist, bearing = _nearest_surface(self.pos, *obstacles)
        seen = self.obstacle_active & (dist <= MAX_SENSE_RANGE)
        dist = torch.where(seen, dist, torch.full_like(dist, float("inf")))
        near, idx = torch.topk(dist, self.k_nearest, dim=1, largest=False)
        visible = torch.isfinite(near)
        near = torch.where(visible, near, torch.zeros_like(near))
        rel_bearing = bearing.gather(1, idx) - self.yaw.unsqueeze(1)
        slots = torch.stack([near * torch.cos(rel_bearing), near * torch.sin(rel_bearing), visible.float()], dim=-1)
        return slots.reshape(self.num_envs, -1)

    def step(self, actions_continuous: torch.Tensor):
        """actions_continuous: (N, 3) raw [vx_raw, vz_raw, yaw_rate_raw],
        same format discrete_action_to_continuous() produces -- squashed
        through the SAME tanh formulas the real environment uses, so the
        physical vx/yaw_rate values match exactly."""
        vx_cmd = ((torch.tanh(actions_continuous[:, 0]) + 1.0) / 2.0) * self.cfg.max_xy_vel
        yaw_rate_cmd = torch.tanh(actions_continuous[:, 2]) * self.cfg.max_yaw_rate

        # Pure kinematic integration -- no thrust, no attitude dynamics.
        self.yaw = self.yaw + yaw_rate_cmd * DT
        self.yaw = torch.atan2(torch.sin(self.yaw), torch.cos(self.yaw))  # wrap to [-pi, pi]
        vx_world = vx_cmd * torch.cos(self.yaw)
        vy_world = vx_cmd * torch.sin(self.yaw)
        self.pos = self.pos + torch.stack([vx_world, vy_world], dim=-1) * DT

        self.step_count += 1

        obs_dict, dist = self._get_obs()
        reached = dist < REACHED_THRESHOLD
        out_of_bounds = dist > OUT_OF_BOUNDS_DIST
        timed_out = self.step_count >= self.max_steps
        # The only crash is obstacle contact (no ground, no attitude to lose), so
        # Stage 0 cannot crash. Contact is tested after the move, which cannot
        # tunnel through an obstacle because a step is at most max_xy_vel * DT
        # (0.05 m by default), far thinner than any obstacle.
        if self.num_obstacles == 0:
            crashed = torch.zeros_like(reached)
        else:
            crashed = obstacle_surface_distance(
                self.pos,
                self.obstacle_xy[:, :NUM_CYLINDERS], self._circle_r,
                self.obstacle_xy[:, NUM_CYLINDERS:], self._box_half,
            ) < DRONE_RADIUS

        terminated = reached | crashed
        truncated = out_of_bounds | timed_out

        # Named like the Isaac env's flags. last_outcomes() and trace_2d.py read them.
        self._last_reached_flag = reached
        self._last_ground_crash = torch.zeros_like(reached)
        self._last_any_contact = crashed
        self._last_out_of_bounds = out_of_bounds

        reward, self.last_reward_breakdown = self._compute_reward(
            dist, actions_continuous, reached, crashed, vx_signed=vx_cmd, return_breakdown=True)

        # The episode's true last state, which the resample below overwrites
        # for every env that just finished.
        self.pre_reset_state = {
            "pos": self.pos.clone(), "yaw": self.yaw.clone(), "target": self.target.clone(),
            "obstacle_xy": self.obstacle_xy.clone(), "obstacle_active": self.obstacle_active.clone(),
        }
        done = terminated | truncated
        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            self._sample_positions(done_idx)

        return obs_dict, reward, terminated, truncated, {}

    def last_outcomes(self):
        """One OUTCOMES name per env for the latest step(). Only meaningful
        for envs whose episode ended on that step."""
        flags = torch.stack([
            self._last_reached_flag, self._last_any_contact, self._last_out_of_bounds,
            torch.ones_like(self._last_reached_flag),  # timed_out, the remaining way to end
        ], dim=-1)
        # argmax returns the first True, so OUTCOMES order is the priority order.
        return [OUTCOMES[i] for i in flags.int().argmax(dim=-1).tolist()]

    # Weight on the multiplicative progress term. v*cos(theta) ranges
    # [-forward_speed, +forward_speed] (1.5 at default settings), so this
    # keeps per-step reward in roughly [-0.075, +0.075] -- a similar scale
    # to the old rew_distance term, for continuity with earlier runs.
    K_PROGRESS = 0.05

    def _compute_reward(self, dist, actions_continuous, reached, crashed,
                         vx_signed=None, return_breakdown=False):
        """Redesigned reward, replacing the additive rew_distance +
        rew_yaw_aligned + rew_action_rate combination (which had two
        separate, independently-farmable exploits diagnosed via
        trace_2d.py: reward for facing the target while not moving, and
        reward for being close while not making progress).

        Following an established pattern from mobile-robot navigation
        literature (e.g. Sathyamoorthy et al.-style designs; see also
        RUMOR, arXiv:2404.16672), the per-step shaping reward is now a
        SINGLE multiplicative term: signed forward velocity times the
        cosine of the heading error. This is structurally exploit-proof
        for the "loiter while facing the target" failure mode: turning in
        place has v=0, which zeroes the ENTIRE term regardless of how
        well-aligned the heading is -- there is no separate "heading
        credit" to bank independently of actually moving. rew_action_rate
        is also removed: with turning contributing nothing on its own,
        there's no longer an incentive to flip-flop between actions, so
        the smoothness penalty that existed specifically to suppress that
        is no longer needed.
        """
        if vx_signed is None:
            # vx_signed: the actual physical forward speed for this step
            # (0 for turn actions, forward_speed for the forward action).
            # Callers that already computed vx_cmd in step() should pass
            # it in directly rather than re-deriving it here.
            vx_signed = ((torch.tanh(actions_continuous[:, 0]) + 1.0) / 2.0) * self.cfg.max_xy_vel

        rel = self.target - self.pos
        cos_yaw = torch.cos(self.yaw)
        sin_yaw = torch.sin(self.yaw)
        dx = rel[:, 0] * cos_yaw + rel[:, 1] * sin_yaw
        horiz_dist = torch.norm(rel, dim=-1).clamp(min=1e-6)
        yaw_cos = dx / horiz_dist

        # REVERTED from yaw_cos**3 (v4): that change reduced reached-rate
        # from 29.0% to 8.0% and made trajectories markedly worse overall,
        # not just at the near-target orbiting case it targeted -- likely
        # because cubing suppresses reward too aggressively across the
        # WHOLE range (e.g. yaw_cos=0.5 -> 0.125, a 4x reduction), not just
        # near yaw_cos=1, which hurt early-training learning of basic
        # "which way to turn" before alignment is even close. Back to
        # linear pending a properly isolated test of the actual mechanism
        # behind the near-target orbiting (hypothesis: turn action's fixed
        # angular step size, not reward shape, prevents settling exactly
        # at zero heading error -- see test_turn_granularity.py).
        rew_progress = vx_signed * yaw_cos * self.K_PROGRESS
        rew_reached = reached.float() * 100.0
        rew_crashed = crashed.float() * -50.0

        total = rew_progress + rew_reached + rew_crashed
        if return_breakdown:
            return total, {
                "rew_progress": rew_progress, "rew_reached": rew_reached,
                "rew_crashed": rew_crashed, "yaw_cos": yaw_cos, "vx_signed": vx_signed,
            }
        return total

    @property
    def unwrapped(self):
        return self


def add_env_args(parser):
    """The obstacle and distance-input options every 2D script takes."""
    parser.add_argument("--num_obstacles", type=int, default=0,
                        help=f"active obstacles per episode, 0 (Stage 0, pure navigation) to {MAX_OBSTACLES}")
    parser.add_argument("--distance_mode", choices=DISTANCE_MODES, default="ground_truth",
                        help="obstacle inputs, exact sector distances, emulated depth cameras, "
                             "or nearest-surface-point coordinates")
    parser.add_argument("--k_nearest", type=int, default=DEFAULT_K_NEAREST,
                        help="obstacle slots in the state, coords mode only")


def env_from_args(args, num_envs, cfg, device):
    return DroneNav2DEnv(num_envs=num_envs, cfg=cfg, device=device, num_obstacles=args.num_obstacles,
                         distance_mode=args.distance_mode, k_nearest=args.k_nearest)
