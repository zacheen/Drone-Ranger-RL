"""
Pure-2D, Isaac-Lab-free training environment -- isolates the DQN/reward/
network design from any Isaac Lab simulation detail (physics stepping,
attitude controller dynamics, camera rendering) to test whether the
oscillation/low-margin problem seen in q_sweep.py is caused by the DQN
side of things or by something in the Isaac Lab simulation layer.

Kinematics only: no thrust, no attitude, no gravity. Position and heading
are integrated directly from the same discrete action -> (vx, yaw_rate)
mapping used by the real environment. The control step is 0.1 s here, five
times Isaac Lab's 0.02 s (sim.dt=1/100 * decimation=2).

With distance_mode="ground_truth" (the default) the front sector is split in two halves:
    state = [d_front_left, d_front_right, d_left, d_right, dx, dy]
each an exact distance to the nearest obstacle surface in that sector
(obstacle_geometry.py), so the state says on which side of the heading a
front obstacle lies. With distance_mode="camera" the state keeps the real
environment's layout, state = [d_front, d_left, d_right, dx, dy], with what
the Isaac env's three depth cameras would report (camera_emulation.py). With
num_obstacles=0 (Stage 0) both modes read 10.0 (max sensing range) in every
reading, what the real environment's cameras report when nothing is in view.

distance_mode="coords" is a diagnostic upper bound with no Isaac
counterpart. It replaces the three readings with k_nearest slots
(sx, sy, visible), the nearest surface point of each of the k_nearest
closest obstacles in the body frame, nearest first:
    state = [s1_x, s1_y, v1, ..., sK_x, sK_y, vK, dx, dy]
Its norm is the exact distance, with no sector and so no blind spot. An
obstacle beyond MAX_SENSE_RANGE or parked leaves its slot all zeros, so the
input size depends on k_nearest only and stays fixed across num_obstacles.

num_obstacles=1..8 (Stage 1) uses the layout of the collaborator's commit
1643e0e rather than the Isaac env's. The target is drawn as in Stage 0 and
redrawn until it lies MIN_START_TARGET_DIST from the start, then circles of
radius OBSTACLE_RADIUS are rejection-sampled in the same box, clear of the start
and the target, with obstacle 0 forced onto the start-target segment so it has
to be avoided. Touching an obstacle ends the episode as a crash.

Interface intentionally mirrors gymnasium's reset()/step() so train_dqn.py
needs minimal changes to switch environments.
"""
import math

import torch

from dqn_config import DQNConfig, SPEED_FRACTIONS
from camera_emulation import FAR_CLIP, TAN_HALF_FOV, camera_readings
from obstacle_geometry import _nearest_surface, obstacle_sector_distances, obstacle_surface_distance

DT = 0.1  # seconds per control step, 10 Hz instead of Isaac Lab's 50 Hz
MAX_SENSE_RANGE = FAR_CLIP  # both distance modes cap at the cameras' far clip
EPISODE_LENGTH_S = 20.0  # matches the real env's episode_length_s
TARGET_POS_RANGE = 5.0   # start/target sampled uniformly in [-5, 5] on each axis
REACHED_THRESHOLD = 0.75
OUT_OF_BOUNDS_DIST = 40.0  # matches the real env's widened boundary

# Obstacle shape and crash rule of commit 1643e0e, circles and no boxes, with room for eight.
OBSTACLE_RADIUS = 0.5
CYLINDER_RADII = (OBSTACLE_RADIUS,) * 8
BOX_HALF_EXTENTS = ()
NUM_CYLINDERS = len(CYLINDER_RADII)
MAX_OBSTACLES = NUM_CYLINDERS + len(BOX_HALF_EXTENTS)
DRONE_RADIUS = 0.2  # a crash once the drone's center is this close to a surface
OBSTACLE_CLEARANCE = 0.5  # extra spawn gap around the start and the target
START_KEEPOUT = OBSTACLE_RADIUS + DRONE_RADIUS + OBSTACLE_CLEARANCE  # min center distance from the start
TARGET_KEEPOUT = OBSTACLE_RADIUS + REACHED_THRESHOLD + OBSTACLE_CLEARANCE  # min center distance from the target
PLACEMENT_ATTEMPTS = 30  # a slot still invalid after this many draws stays parked
# With obstacles, the target is redrawn until it lies this far from the start. Below about 2.77 m no
# spot on the middle half of the segment clears both keepouts, so obstacle 0 could not block the path.
MIN_START_TARGET_DIST = 3.0
SENSOR_HALF_FOV = math.atan(TAN_HALF_FOV)  # half of the cameras' 73 deg HFOV
# ground_truth sectors as (offset, half width): front-left, front-right, left, right. The front pair
# covers the same 73 deg as one front sector, so only the side of a front obstacle is new.
GROUND_TRUTH_SECTORS = ((SENSOR_HALF_FOV / 2, SENSOR_HALF_FOV / 2), (-SENSOR_HALF_FOV / 2, SENSOR_HALF_FOV / 2),
                        (math.pi / 2, SENSOR_HALF_FOV), (-math.pi / 2, SENSOR_HALF_FOV))
DISTANCE_MODES = ("ground_truth", "camera", "coords")
DEFAULT_K_NEAREST = 3
PREV_ACTION_DIM = 3 + len(SPEED_FRACTIONS)
SIMILARITY_WINDOW = 20  # steps back to the pose the similarity penalty compares against
SIMILARITY_OBS_DIM = 3  # body-frame dx, dy and heading change since that pose

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
        self.num_readings = {"coords": 3 * k_nearest, "ground_truth": len(GROUND_TRUTH_SECTORS), "camera": 3}[distance_mode]
        self.state_dim = self.num_readings + PREV_ACTION_DIM + SIMILARITY_OBS_DIM + 2
        # One-hot of the last turn (none, left, right) followed by a one-hot of the last speed,
        # all zeros on an episode's first step.
        self.prev_action = torch.zeros(num_envs, PREV_ACTION_DIM, device=device)
        self.max_steps = int(EPISODE_LENGTH_S / DT)

        self.pos = torch.zeros(num_envs, 2, device=device)      # (N, 2) x,y
        self.yaw = torch.zeros(num_envs, device=device)          # (N,) radians
        self.target = torch.zeros(num_envs, 2, device=device)    # (N, 2)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Ring of the last SIMILARITY_WINDOW poses, slot step_count % SIMILARITY_WINDOW holding the pose SIMILARITY_WINDOW steps back.
        self.pose_hist_pos = torch.zeros(num_envs, SIMILARITY_WINDOW, 2, device=device)
        self.pose_hist_yaw = torch.zeros(num_envs, SIMILARITY_WINDOW, device=device)
        self.pose_similarity = torch.zeros(num_envs, device=device)
        # What the state shows of the pose SIMILARITY_WINDOW steps back, or of the start pose before then,
        # so the network can see the similarity penalty coming.
        self.pose_delta = torch.zeros(num_envs, SIMILARITY_OBS_DIM, device=device)

        self._circle_r = torch.tensor(CYLINDER_RADII, device=device)
        self._box_half = torch.tensor(BOX_HALF_EXTENTS, device=device).reshape(-1, 2)
        # Inactive obstacles are parked where the Isaac env parks them, far
        # beyond sensing range, so they never register in a reading.
        self._parked_xy = torch.stack([
            1000.0 + torch.arange(MAX_OBSTACLES, device=device, dtype=torch.float32),
            torch.full((MAX_OBSTACLES,), 1000.0, device=device),
        ], dim=-1)
        self.obstacle_xy = self._parked_xy.expand(num_envs, -1, -1).clone()  # (N, MAX_OBSTACLES, 2)
        # False for obstacles parked this episode, beyond num_obstacles or unplaceable.
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
        # Obstacles are drawn last, so a seeded Stage 0 run reproduces older runs.
        self.target[env_ids] = (torch.rand(n, 2, device=self.device) * 2 - 1) * TARGET_POS_RANGE
        self.yaw[env_ids] = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        self.step_count[env_ids] = 0
        self.prev_action[env_ids] = 0.0
        self.pose_hist_pos[env_ids] = self.pos[env_ids].unsqueeze(1)
        self.pose_hist_yaw[env_ids] = self.yaw[env_ids].unsqueeze(1)
        self.pose_similarity[env_ids] = 0.0
        self.pose_delta[env_ids] = 0.0
        if self.num_obstacles > 0:
            near = (self.target[env_ids] - self.pos[env_ids]).norm(dim=-1) < MIN_START_TARGET_DIST
            while near.any():
                ids = env_ids[near]
                self.target[ids] = (torch.rand(len(ids), 2, device=self.device) * 2 - 1) * TARGET_POS_RANGE
                near[near.clone()] = (self.target[ids] - self.pos[ids]).norm(dim=-1) < MIN_START_TARGET_DIST
            self._sample_obstacles(env_ids)

    def _sample_obstacles(self, env_ids):
        """Rejection-sample the active obstacle centers as commit 1643e0e does.
        A center must lie START_KEEPOUT from the start and TARGET_KEEPOUT from the
        target. Obstacle 0 is drawn on the middle half of the start-target segment,
        within OBSTACLE_RADIUS of it sideways, and only where it clears both keepouts,
        so with MIN_START_TARGET_DIST it is always placed. Another slot that fails every
        attempt stays parked, so an episode can have fewer than num_obstacles obstacles."""
        n, k, dev = len(env_ids), self.num_obstacles, self.device
        start, tgt = self.pos[env_ids], self.target[env_ids]
        seg = tgt - start
        length = seg.norm(dim=-1).clamp(min=1e-6)
        perp = torch.stack([-seg[:, 1], seg[:, 0]], dim=-1) / length[:, None]
        out = self._parked_xy[:k].expand(n, -1, -1).clone()
        todo = torch.ones(n, k, dtype=torch.bool, device=dev)
        for _ in range(PLACEMENT_ATTEMPTS):
            cand = (torch.rand(n, k, 2, device=dev) * 2 - 1) * TARGET_POS_RANGE
            lateral = (torch.rand(n, device=dev) * 2 - 1) * OBSTACLE_RADIUS
            # The stretch of the segment, at this sideways offset, that clears both keepouts.
            # The small margin keeps a point at the very edge from failing the >= test by rounding.
            lo = ((START_KEEPOUT * 1.0001) ** 2 - lateral ** 2).sqrt() / length
            hi = 1 - ((TARGET_KEEPOUT * 1.0001) ** 2 - lateral ** 2).sqrt() / length
            lo, hi = lo.clamp(min=0.25), hi.clamp(max=0.75)
            along = lo + torch.rand(n, device=dev) * (hi - lo).clamp(min=0)
            cand[:, 0] = start + seg * along[:, None] + perp * lateral[:, None]
            ok = (((cand - start[:, None]).norm(dim=-1) >= START_KEEPOUT)
                  & ((cand - tgt[:, None]).norm(dim=-1) >= TARGET_KEEPOUT))
            out = torch.where((todo & ok)[..., None], cand, out)
            todo &= ~ok
            if not todo.any():
                break
        self.obstacle_xy[env_ids] = self._parked_xy
        self.obstacle_xy[env_ids, :k] = out
        active = torch.zeros(n, MAX_OBSTACLES, dtype=torch.bool, device=dev)
        active[:, :k] = ~todo
        self.obstacle_active[env_ids] = active

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
            d = torch.full((self.num_envs, self.num_readings), MAX_SENSE_RANGE, device=self.device)
        elif self.distance_mode == "ground_truth":
            d = obstacle_sector_distances(self.pos, self.yaw, *obstacles, SENSOR_HALF_FOV, MAX_SENSE_RANGE,
                                          sectors=GROUND_TRUTH_SECTORS)
        else:
            d = camera_readings(self.pos, self.yaw, *obstacles)

        state = torch.cat([d, self.prev_action, self.pose_delta, dx.unsqueeze(-1), dy.unsqueeze(-1)], dim=-1)
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
        self._update_pose_similarity()
        turn = torch.sign(yaw_rate_cmd)
        # Nearest speed, since the tanh squash returns each speed only to within float error.
        speeds = torch.tensor(SPEED_FRACTIONS, device=self.device) * self.cfg.forward_speed
        speed = (vx_cmd.unsqueeze(-1) - speeds).abs().argmin(dim=-1)
        self.prev_action = torch.cat([torch.stack([turn == 0, turn > 0, turn < 0], dim=-1).float(),
                                      torch.nn.functional.one_hot(speed, len(SPEED_FRACTIONS)).float()], dim=-1)

        obs_dict, dist = self._get_obs()
        reached = dist < REACHED_THRESHOLD
        out_of_bounds = dist > OUT_OF_BOUNDS_DIST
        timed_out = self.step_count >= self.max_steps
        # The only crash is obstacle contact (no ground, no attitude to lose), so
        # Stage 0 cannot crash. Contact is tested after the move, which cannot
        # tunnel through an obstacle because a step is at most max_xy_vel * DT
        # (0.25 m by default), thinner than any obstacle.
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
        reset_obs = obs_dict
        if done.any():
            done_idx = done.nonzero(as_tuple=True)[0]
            self._sample_positions(done_idx)
            reset_obs, _ = self._get_obs()

        # For a finished env obs_dict still shows the episode that just ended;
        # choose the next action from info["reset_obs"] (post-resample).
        return obs_dict, reward, terminated, truncated, {"reset_obs": reset_obs}

    def _update_pose_similarity(self):
        """Compare the pose just reached with the one SIMILARITY_WINDOW steps back, 1 for the same
        position and heading, then store the current pose in its slot."""
        rows = torch.arange(self.num_envs, device=self.device)
        slot = self.step_count % SIMILARITY_WINDOW
        offset = self.pos - self.pose_hist_pos[rows, slot]
        moved = offset.norm(dim=-1)
        turned = self.yaw - self.pose_hist_yaw[rows, slot]
        cos_yaw, sin_yaw = torch.cos(self.yaw), torch.sin(self.yaw)
        self.pose_delta = torch.stack([offset[:, 0] * cos_yaw + offset[:, 1] * sin_yaw,
                                        -offset[:, 0] * sin_yaw + offset[:, 1] * cos_yaw,
                                        torch.atan2(torch.sin(turned), torch.cos(turned))], dim=-1)
        similarity = torch.exp(-moved / self.SIMILARITY_POS_SCALE - self.pose_delta[:, 2].abs() / self.SIMILARITY_YAW_SCALE)
        # Steps before SIMILARITY_WINDOW have no pose that far back, their slots still hold the start pose.
        self.pose_similarity = torch.where(self.step_count >= SIMILARITY_WINDOW, similarity, torch.zeros_like(similarity))
        self.pose_hist_pos[rows, slot] = self.pos
        self.pose_hist_yaw[rows, slot] = self.yaw

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
    # keeps per-step reward in roughly [-0.225, +0.225], three times the
    # old rew_distance scale.
    K_PROGRESS = 0.15
    # Standing still forever costs about -5 discounted at gamma 0.98, far above the -50 crash.
    SIMILARITY_PENALTY = -0.1
    # Metres of travel over SIMILARITY_WINDOW steps that cut the similarity to 1/e. The slowest speed covers
    # 0.15 m in that time, so crawling forward keeps about 5% of the penalty.
    SIMILARITY_POS_SCALE = 0.05
    # Radians of heading change over SIMILARITY_WINDOW steps that cut the similarity to 1/e. Steepest at no
    # change, so one 0.08 rad turn step already drops it to 0.67, while (1 + cos) / 2 was flat there and a
    # stopped drone swinging left and right saw almost no gain from turning one way.
    SIMILARITY_YAW_SCALE = 0.2

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

        rew_similarity = self.pose_similarity * self.SIMILARITY_PENALTY

        total = rew_progress + rew_reached + rew_crashed + rew_similarity
        if return_breakdown:
            return total, {
                "rew_progress": rew_progress, "rew_reached": rew_reached,
                "rew_crashed": rew_crashed, "rew_similarity": rew_similarity, "yaw_cos": yaw_cos, "vx_signed": vx_signed,
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
