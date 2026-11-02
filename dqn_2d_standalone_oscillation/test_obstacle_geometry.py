"""
Tests for obstacle_geometry.py and for how dqn_2d_env.py uses it. Runs as a
plain script, no pytest:

    python test_obstacle_geometry.py

The brute-force reference samples points densely along every obstacle's
boundary and keeps the nearest one inside the sector. It shares no code with
the ray intersection it checks, so agreement means the exact method is right.
"""
import ast
import math
import os

import torch

import dqn_2d_env as E
import obstacle_geometry as og
from dqn_config import DQNConfig
from dqn_action_space import discrete_action_to_continuous

HALF = math.radians(36.5)
MAX_RANGE = 10.0
ISAAC_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "isaaclab_env", "drone_nav_v2_env_dqn.py")


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def _boundary_points(circle_xy, circle_r, box_xy, box_half, m=20000):
    """(N, K, M, 2) float64 points on each obstacle outline."""
    theta = torch.linspace(0.0, 2.0 * math.pi, m, dtype=torch.float64)
    ring = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    circles = circle_xy.double().unsqueeze(2) + circle_r.double()[None, :, None, None] * ring
    s = torch.linspace(-1.0, 1.0, m // 4, dtype=torch.float64)
    one = torch.ones_like(s)
    square = torch.cat([torch.stack(e, dim=-1) for e in
                        ((s, one), (s, -one), (one, s), (-one, s))], dim=0)
    boxes = box_xy.double().unsqueeze(2) + box_half.double()[None, :, None, :] * square
    # Pad the box outlines to the circles' point count so the two can be concatenated.
    pad = circles.shape[2] - boxes.shape[2]
    boxes = torch.cat([boxes, boxes[:, :, :pad]], dim=2)
    return torch.cat([circles, boxes], dim=1)


def _brute_force(pos, yaw, circle_xy, circle_r, box_xy, box_half):
    pts = _boundary_points(circle_xy, circle_r, box_xy, box_half)
    rel = pts - pos.double()[:, None, None, :]
    dist = rel.norm(dim=-1)
    bearing = torch.atan2(rel[..., 1], rel[..., 0])
    inside_c = (circle_xy.double() - pos.double()[:, None]).norm(dim=-1) < circle_r.double()
    inside_b = ((box_xy.double() - pos.double()[:, None]).abs() < box_half.double()).all(dim=-1)
    inside = torch.cat([inside_c, inside_b], dim=1).any(dim=1)
    out = []
    for offset in (0.0, math.pi / 2.0, -math.pi / 2.0):
        center = (yaw.double() + offset)[:, None, None]
        in_sector = _wrap(bearing - center).abs() <= HALF
        d = torch.where(in_sector, dist, torch.full_like(dist, float("inf")))
        d = d.amin(dim=(1, 2)).clamp(max=MAX_RANGE)
        out.append(torch.where(inside, torch.zeros_like(d), d))
    return torch.stack(out, dim=-1)


def _random_scene(n, gen):
    pos = (torch.rand(n, 2, generator=gen) * 2 - 1) * 2.0
    yaw = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
    circle_xy = (torch.rand(n, 3, 2, generator=gen) * 2 - 1) * 8.0
    box_xy = (torch.rand(n, 2, 2, generator=gen) * 2 - 1) * 8.0
    circle_r = torch.tensor([0.3, 0.6, 1.0])
    box_half = torch.tensor([[0.5, 0.5], [0.9, 0.3]])
    return pos, yaw, circle_xy, circle_r, box_xy, box_half


def test_matches_brute_force():
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    edge_cases = 0
    for _ in range(6):
        scene = _random_scene(100, gen)
        exact = og.obstacle_sector_distances(*scene, HALF, MAX_RANGE)
        ref = _brute_force(*scene).float()
        worst = max(worst, (exact - ref).abs().max().item())
        # Count readings decided by a sector-edge ray (longer than the nearest
        # surface point), which a nearest-point shortcut would get wrong.
        nearest = og._nearest_surface(scene[0], scene[2], scene[3], scene[4], scene[5])[0].amin(dim=1)
        edge_cases += ((exact < MAX_RANGE) & (exact > nearest[:, None] + 1e-3)).sum().item()
    assert worst < 2e-3, f"exact vs brute force differ by {worst:.5f} m"
    assert edge_cases > 50, f"only {edge_cases} sector-edge readings exercised"
    print(f"  brute force: worst gap {worst:.2e} m over 600 scenes x 3 sectors, {edge_cases} edge-decided readings")


def _one(pos, yaw, circles=(), boxes=()):
    """Readings for a single drone. circles is ((x, y, r), ...), boxes is ((x, y, hx, hy), ...)."""
    far_c, far_b = (1000.0, 1000.0, 0.3), (1000.0, 1001.0, 0.5, 0.5)
    circles, boxes = list(circles) or [far_c], list(boxes) or [far_b]
    f = lambda x: torch.tensor(x, dtype=torch.float32)
    out = og.obstacle_sector_distances(
        f([pos]), f([yaw]),
        f([[c[:2] for c in circles]]), f([c[2] for c in circles]),
        f([[b[:2] for b in boxes]]), f([b[2:] for b in boxes]),
        HALF, MAX_RANGE)
    return out[0].tolist()


def _close(a, b, tol=1e-4):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_hand_checked_cases():
    assert _close(_one((0, 0), 0.0), [10, 10, 10]), "parked obstacles must read max_range"
    assert _close(_one((0, 0), 0.0, circles=[(5, 0, 1)]), [4, 10, 10]), "circle dead ahead"
    assert _close(_one((0, 0), 0.0, circles=[(0, 3, 0.5)]), [10, 2.5, 10]), "circle on the left"
    assert _close(_one((0, 0), 0.0, circles=[(0, -3, 0.5)]), [10, 10, 2.5]), "circle on the right"
    assert _close(_one((0, 0), math.pi / 2, circles=[(0, 5, 1)]), [4, 10, 10]), "heading follows yaw"
    # 45 deg sits between the front and left sectors, and a small circle there is seen by neither.
    blind = (3 * math.cos(math.pi / 4), 3 * math.sin(math.pi / 4), 0.2)
    assert _close(_one((0, 0), 0.0, circles=[blind]), [10, 10, 10]), "blind spot"
    # A circle centered at 40 deg pokes past the 36.5 deg front edge. The reading
    # is the edge ray's hit, longer than the nearest-point distance 3.5.
    b = 4 * math.cos(math.radians(3.5))
    expected = b - math.sqrt(b * b - (16 - 0.25))
    c40 = (4 * math.cos(math.radians(40)), 4 * math.sin(math.radians(40)), 0.5)
    got = _one((0, 0), 0.0, circles=[c40])
    assert abs(got[0] - expected) < 1e-4 and expected > 3.5 + 0.05, f"edge hit {got[0]} vs {expected}"
    assert _close(_one((0, 0), 0.0, boxes=[(3, 0, 0.5, 0.5)]), [2.5, 10, 10]), "box face ahead"
    assert _close(_one((0, 0), 0.0, circles=[(0.1, 0, 0.5)]), [0, 0, 0]), "inside an obstacle"
    assert _close(_one((0, 0), 0.0, circles=[(12, 0, 1)]), [10, 10, 10]), "beyond max_range"
    print("  hand-checked cases: ok")


def test_surface_distance():
    pos = torch.tensor([[0.0, 0.0], [5.0, 0.2]])
    circle_xy = torch.tensor([[[3.0, 0.0]], [[3.0, 0.0]]])
    box_xy = torch.tensor([[[0.0, -4.0]], [[5.0, 0.0]]])
    d = og.obstacle_surface_distance(pos, circle_xy, torch.tensor([1.0]), box_xy, torch.tensor([[1.0, 1.0]]))
    assert _close(d.tolist(), [2.0, 0.0]), d.tolist()
    print("  surface distance: ok")


def _isaac_tree():
    with open(ISAAC_ENV, encoding="utf-8") as f:
        return ast.parse(f.read())


def _isaac_literal(tree, name):
    """Value of the first `name = <literal>` or `name: type = <literal>` anywhere in the Isaac env file."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not assigned in {ISAAC_ENV}")


def test_isaac_constants_match():
    """The sensor field of view must keep matching the Isaac env, or the 2D readings stop
    being the noise-free version of what its cameras see. The obstacle layout follows
    commit 1643e0e instead and is checked by test_layout_invariants.
    The far clip is checked by test_camera_emulation.py."""
    tree = _isaac_tree()
    half_fov = math.atan(_isaac_literal(tree, "_horizontal_aperture") / (2.0 * _isaac_literal(tree, "_focal_length")))
    assert math.isclose(half_fov, E.SENSOR_HALF_FOV)
    print("  Isaac Lab constants: FOV matches")


def _env_obstacles(env):
    return (env.obstacle_xy[:, :E.NUM_CYLINDERS], env._circle_r,
            env.obstacle_xy[:, E.NUM_CYLINDERS:], env._box_half)


def test_stage0_shortcuts_match_geometry():
    """With num_obstacles=0 the env skips the geometry. Both shortcuts must equal what it would compute."""
    torch.manual_seed(0)
    env = E.DroneNav2DEnv(512, DQNConfig(), "cpu", num_obstacles=0)
    obs, _ = env.reset()
    cx, cr, bx, bh = _env_obstacles(env)
    full = og.obstacle_sector_distances(env.pos, env.yaw, cx, cr, bx, bh, E.SENSOR_HALF_FOV, E.MAX_SENSE_RANGE,
                                        sectors=E.GROUND_TRUTH_SECTORS)
    assert torch.equal(obs["dqn_state"][:, :env.num_readings], full)
    assert (og.obstacle_surface_distance(env.pos, cx, cr, bx, bh) >= E.DRONE_RADIUS).all()
    print("  Stage 0 shortcuts: identical to the full geometry")


def test_layout_invariants():
    torch.manual_seed(0)
    stage0 = E.DroneNav2DEnv(2000, DQNConfig(), "cpu")
    stage0.reset()
    for k in range(1, E.MAX_OBSTACLES + 1):
        torch.manual_seed(0)
        env = E.DroneNav2DEnv(2000, DQNConfig(), "cpu", num_obstacles=k)
        obs, _ = env.reset()
        # The start and heading follow Stage 0 exactly, and so does every target already far enough away.
        assert torch.equal(env.pos, stage0.pos) and torch.equal(env.yaw, stage0.yaw)
        far = (stage0.target - stage0.pos).norm(dim=-1) >= E.MIN_START_TARGET_DIST
        assert torch.equal(env.target[far], stage0.target[far]) and (~far).any()
        assert ((env.target - env.pos).norm(dim=-1) >= E.MIN_START_TARGET_DIST).all(), f"k={k} target too close"
        active = env.obstacle_active
        assert not active[:, k:].any(), f"k={k} active beyond num_obstacles"
        parked_far = (env.obstacle_xy - env.pos.unsqueeze(1)).norm(dim=-1) > 100
        assert torch.equal(parked_far, ~active), f"k={k} parked obstacles must be far away"
        centers = env.obstacle_xy[:, :k]
        placed = active[:, :k]
        assert ((centers - env.pos[:, None]).norm(dim=-1)[placed] >= E.START_KEEPOUT - 1e-5).all(), f"k={k} start keepout"
        assert ((centers - env.target[:, None]).norm(dim=-1)[placed] >= E.TARGET_KEEPOUT - 1e-5).all(), f"k={k} target keepout"
        assert (centers[placed].abs() <= E.TARGET_POS_RANGE + E.OBSTACLE_RADIUS + 1e-5).all(), f"k={k} outside the box"
        # Obstacle 0 is always placed, on the middle half of the start-target segment.
        seg = env.target - env.pos
        rel = centers[:, 0] - env.pos
        along = (rel * seg).sum(-1) / (seg * seg).sum(-1)
        side = (rel[:, 0] * seg[:, 1] - rel[:, 1] * seg[:, 0]).abs() / seg.norm(dim=-1)
        assert placed[:, 0].all(), f"k={k} obstacle 0 parked"
        assert ((along >= 0.25 - 1e-5) & (along <= 0.75 + 1e-5)).all() and (side <= E.OBSTACLE_RADIUS + 1e-5).all(), f"k={k} obstacle 0 off the path"
        cx, cr, bx, bh = _env_obstacles(env)
        assert (og.obstacle_surface_distance(env.pos, cx, cr, bx, bh) > E.DRONE_RADIUS).all(), f"k={k} spawn in contact"
        assert torch.isfinite(obs["dqn_state"]).all()
    print(f"  layout invariants: hold for 2000 resets at each k = 1..{E.MAX_OBSTACLES}")


def test_contact_crash_and_obs_wiring():
    """step() and _get_obs() with obstacles on hand-placed layouts, one per env."""
    env = E.DroneNav2DEnv(4, DQNConfig(), "cpu", num_obstacles=5)
    env.reset()
    env.obstacle_xy[:] = env._parked_xy
    env.pos[:] = 0.0
    env.yaw[:] = 0.0
    env.target[:] = torch.tensor([-20.0, 0.0])         # behind the drone, never reached
    env.obstacle_xy[0, 1] = torch.tensor([0.74, 0.0])  # r=0.5, surface 0.24 m ahead, so one 0.25 m step ends within DRONE_RADIUS
    env.obstacle_xy[1, 3] = torch.tensor([0.25, 0.69])  # beside where the 0.25 m step ends, surface 0.19 m away, within DRONE_RADIUS
    env.obstacle_xy[2, 1] = torch.tensor([5.0, 0.0])   # clear
    env.yaw[3] = math.pi / 2
    env.obstacle_xy[3, 1] = torch.tensor([0.0, 3.0])   # ahead only because of the heading
    d = env._get_obs()[0]["dqn_state"]
    c = E.CAMERA_BINS  # the front-left camera's rightmost bin, whose right edge is the heading
    assert abs(d[2, c].item() - 4.5) < 1e-4 and abs(d[3, c].item() - 2.5) < 1e-4
    _, reward, terminated, _, _ = env.step(torch.tensor([[10.0, 0.0, 0.0]]).expand(4, -1))  # full-speed forward
    assert terminated.tolist() == [True, True, False, False]
    assert (reward[:2] < -49).all() and (reward[2:] > -1).all()
    assert env.last_outcomes()[:2] == ["contact_crash"] * 2
    print("  crash detection and obstacle readings through step(): ok")


def test_proximity_penalty():
    """The penalty ramps from 0 at proximity_margin of clearance to proximity_penalty at contact, and is off by default."""
    for cfg, expected in ((DQNConfig(proximity_penalty=0.2, proximity_margin=0.3), [-0.1, 0.0, -0.2]),
                          (DQNConfig(), [0.0, 0.0, 0.0])):
        env = E.DroneNav2DEnv(3, cfg, "cpu", num_obstacles=1)
        env.reset()
        env.obstacle_xy[:] = env._parked_xy
        env.pos[:] = 0.0
        env.yaw[:] = 0.0
        env.target[:] = torch.tensor([-20.0, 0.0])
        # Beside where the 0.25 m step ends, r=0.5, surface 0.35, 0.6 and 0.05 m from the drone's center.
        env.obstacle_xy[:, 0] = torch.tensor([[0.25, 0.85], [0.25, 1.1], [0.25, 0.55]])
        env.step(torch.tensor([[10.0, 0.0, 0.0]]).expand(3, -1))
        assert torch.allclose(env.last_reward_breakdown["rew_proximity"], torch.tensor(expected), atol=1e-5)
    print("  proximity penalty: linear inside the margin, capped at contact, zero by default")


def test_camera_bins_tile_each_camera():
    """Each camera's bins tile its 73 deg, so their minimum is the single-sector reading."""
    n = E.CAMERA_BINS
    for k, cam in enumerate(E.CAMERA_OFFSETS):
        edges = [(c - w, c + w) for c, w in E.GROUND_TRUTH_SECTORS[k * n:(k + 1) * n]]
        assert abs(edges[0][0] - (cam - E.SENSOR_HALF_FOV)) < 1e-12 and abs(edges[-1][1] - (cam + E.SENSOR_HALF_FOV)) < 1e-12
        assert all(abs(edges[b][1] - edges[b + 1][0]) < 1e-12 for b in range(n - 1))
    gen = torch.Generator().manual_seed(1)
    worst = 0.0
    for _ in range(4):
        scene = _random_scene(200, gen)
        whole = og.obstacle_sector_distances(*scene, E.SENSOR_HALF_FOV, MAX_RANGE,
                                             sectors=tuple((cam, E.SENSOR_HALF_FOV) for cam in E.CAMERA_OFFSETS))
        binned = og.obstacle_sector_distances(*scene, E.SENSOR_HALF_FOV, MAX_RANGE, sectors=E.GROUND_TRUTH_SECTORS)
        worst = max(worst, (binned.view(-1, len(E.CAMERA_OFFSETS), n).amin(dim=-1) - whole).abs().max().item())
    assert worst < 1e-4, f"bin minimum and single sector differ by {worst:.5f} m"
    print(f"  camera bins: tile each camera, minimum matches the single sector within {worst:.1e} m")


def test_front_bins_tell_the_side():
    """A front obstacle off the heading by 20 deg is nearest in the front bin on its side."""
    env = E.DroneNav2DEnv(2, DQNConfig(), "cpu", num_obstacles=5)
    env.reset()
    env.obstacle_xy[:] = env._parked_xy
    env.pos[:] = 0.0
    env.yaw[:] = 0.0
    a = math.radians(20.0)
    env.obstacle_xy[0, 1] = torch.tensor([2.0 * math.cos(a), 2.0 * math.sin(a)])    # left of the heading
    env.obstacle_xy[1, 1] = torch.tensor([2.0 * math.cos(a), -2.0 * math.sin(a)])   # right of the heading
    d = env._get_obs()[0]["dqn_state"][:, :len(E.GROUND_TRUTH_SECTORS)]
    # Bins run right to left from -73 deg in steps of 14.6 deg, so +20 deg falls in bin 6 and -20 deg in bin 3.
    assert abs(d[0, 6].item() - 1.5) < 1e-4 and d[0, 3].item() > 1.5
    assert abs(d[1, 3].item() - 1.5) < 1e-4 and d[1, 6].item() > 1.5
    assert torch.allclose(d[0], d[1].flip(0), atol=1e-5)
    print("  front bins: the nearest reading moves to the side the obstacle is on")


def test_outcome_priority():
    env = E.DroneNav2DEnv(5, DQNConfig(), "cpu")
    t = lambda *v: torch.tensor(v, dtype=torch.bool)
    env._last_reached_flag = t(1, 0, 0, 0, 1)
    env._last_any_contact = t(1, 1, 0, 0, 0)
    env._last_out_of_bounds = t(1, 1, 1, 0, 0)
    assert env.last_outcomes() == ["reached", "contact_crash", "out_of_bounds", "timed_out", "reached"]
    print("  outcome priority: ok")


def test_coords_mode_hand_checked():
    env = E.DroneNav2DEnv(3, DQNConfig(), "cpu", num_obstacles=5, distance_mode="coords", k_nearest=2)
    assert env.state_dim == 8 + E.PREV_ACTION_DIM + E.SIMILARITY_OBS_DIM
    env.reset()
    env.obstacle_xy[:] = env._parked_xy
    env.obstacle_active[:] = True
    env.pos[:] = 0.0
    env.yaw[:] = 0.0
    env.target[:] = torch.tensor([3.0, 4.0])
    env.obstacle_xy[0, 1] = torch.tensor([2.0, 0.0])   # r=0.5, surface point (1.5, 0)
    env.obstacle_xy[0, 3] = torch.tensor([0.0, -3.0])  # r=0.5, surface point (0, -2.5)
    env.yaw[1] = math.pi / 2
    env.obstacle_xy[1, 1] = torch.tensor([0.0, 2.0])   # world +y is body +x at this heading
    env.obstacle_xy[2, 1] = torch.tensor([12.0, 0.0])  # surface 11.5 m away, beyond sensing range
    env.obstacle_xy[2, 2] = torch.tensor([1.0, 0.0])
    env.obstacle_active[2, 2] = False                  # inactive, so never seen
    s = env._get_obs()[0]["dqn_state"]
    expected = torch.tensor([
        [1.5, 0.0, 1.0, 0.0, -2.5, 1.0],
        [1.5, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    assert torch.allclose(s[:, :6], expected, atol=1e-5), s
    assert (s[:, 6:-2] == 0).all()  # previous-action slots before any step
    assert torch.allclose(s[0, -2:], torch.tensor([3.0, 4.0])) and torch.allclose(s[1, -2:], torch.tensor([4.0, -3.0]), atol=1e-5)
    stage0 = E.DroneNav2DEnv(4, DQNConfig(), "cpu", num_obstacles=0, distance_mode="coords")
    obs, _ = stage0.reset()
    assert obs["dqn_state"].shape == (4, stage0.state_dim) and (obs["dqn_state"][:, :-2] == 0).all()
    print("  coords mode: hand-placed slots, range and parking filters, Stage 0 padding ok")


def test_coords_mode_matches_geometry():
    """Slot 1's length is the nearest active surface distance, and the slots come nearest first."""
    torch.manual_seed(0)
    env = E.DroneNav2DEnv(2000, DQNConfig(), "cpu", num_obstacles=5, distance_mode="coords", k_nearest=5)
    s = env.reset()[0]["dqn_state"][:, :15].reshape(-1, 5, 3)
    cx, cr, bx, bh = _env_obstacles(env)
    dist, _ = og._nearest_surface(env.pos, cx, cr, bx, bh)
    dist = torch.where(env.obstacle_active & (dist <= E.MAX_SENSE_RANGE), dist, torch.full_like(dist, float("inf")))
    nearest = dist.amin(dim=1)
    seen = torch.isfinite(nearest)
    assert torch.equal(s[:, 0, 2] == 1, seen)
    assert torch.allclose(s[seen, 0, :2].norm(dim=-1), nearest[seen], atol=1e-4)
    lengths = torch.where(s[..., 2] == 1, s[..., :2].norm(dim=-1), torch.full_like(s[..., 2], float("inf")))
    assert (lengths[:, 1:] >= lengths[:, :-1] - 1e-5).all()
    assert torch.equal((s[..., 2] == 1).sum(dim=1), torch.isfinite(dist).sum(dim=1))
    print("  coords mode: matches _nearest_surface on 2000 random layouts")


def test_pre_reset_state_and_breakdown():
    env = E.DroneNav2DEnv(1, DQNConfig(), "cpu")
    env.reset()
    env.pos[:] = 0.0
    env.yaw[:] = 0.0
    step = DQNConfig().max_xy_vel * E.DT  # one full-speed step
    env.target[:] = torch.tensor([E.REACHED_THRESHOLD + step - 0.02, 0.0])  # one step brings it within REACHED_THRESHOLD
    _, reward, terminated, _, _ = env.step(torch.tensor([[10.0, 0.0, 0.0]]))
    assert terminated.item()
    end = env.pre_reset_state
    assert torch.allclose(end["pos"][0], torch.tensor([step, 0.0]), atol=1e-4)
    assert torch.allclose(end["target"][0], torch.tensor([E.REACHED_THRESHOLD + step - 0.02, 0.0]))
    assert not torch.equal(env.target[0], end["target"][0])  # resampled for the next episode
    assert env.last_reward_breakdown["rew_reached"].item() == 100.0
    assert torch.allclose(sum(env.last_reward_breakdown[k] for k in ("rew_progress", "rew_reached", "rew_crashed",
                                                                      "rew_similarity", "rew_proximity")), reward)
    print("  pre-reset snapshot and reward breakdown: hold the finished episode's last step")


def test_reset_obs_starts_next_episode():
    env = E.DroneNav2DEnv(2, DQNConfig(), "cpu")
    env.reset()
    env.pos[:] = 0.0
    env.yaw[:] = 0.0
    near = E.REACHED_THRESHOLD + DQNConfig().max_xy_vel * E.DT - 0.02
    env.target[:] = torch.tensor([[near, 0.0], [3.0, 0.0]])  # env 0 reaches the target this step, env 1 does not
    obs, _, terminated, truncated, info = env.step(torch.tensor([[10.0, 0.0, 0.0]] * 2))
    assert (terminated | truncated).tolist() == [True, False]
    fresh, _ = env._get_obs()
    assert torch.equal(info["reset_obs"]["dqn_state"], fresh["dqn_state"])
    assert not torch.equal(obs["dqn_state"][0], fresh["dqn_state"][0])  # env 0 resampled
    assert torch.equal(obs["dqn_state"][1], fresh["dqn_state"][1])
    print("  reset_obs: finished envs show the new episode, running envs are unchanged")


def test_prev_action_in_state():
    env = E.DroneNav2DEnv(4, DQNConfig(), "cpu")
    obs, _ = env.reset()
    r = env.num_readings
    assert env.state_dim == r + E.PREV_ACTION_DIM + E.SIMILARITY_OBS_DIM + 2 and (obs["dqn_state"][:, r:r + E.PREV_ACTION_DIM] == 0).all()
    env.target[:] = torch.tensor([4.0, 4.0])  # far enough that no env finishes on this step
    env.pos[:] = 0.0
    # stop straight, slow turning left, mid turning right, fast straight
    obs, _, _, _, info = env.step(discrete_action_to_continuous(torch.tensor([0, 4, 8, 9]), "cpu"))
    assert torch.equal(obs["dqn_state"][:, r:r + 3], torch.eye(3)[[0, 1, 2, 0]])
    assert torch.equal(obs["dqn_state"][:, r + 3:r + 7], torch.eye(4))
    expected = torch.tensor(E.SPEED_FRACTIONS) * env.cfg.forward_speed * E.DT
    assert env.pos[0].abs().max() == 0 and torch.allclose(env.pos.norm(dim=-1), expected, atol=1e-5)
    assert torch.equal(info["reset_obs"]["dqn_state"], obs["dqn_state"])
    print("  previous action: one-hot after a step, zeros at reset")


def test_pose_similarity_penalty():
    env = E.DroneNav2DEnv(3, DQNConfig(), "cpu")
    env.reset()
    env.target[:] = torch.tensor([4.0, 4.0])  # far enough that no env finishes in 21 steps
    # stop straight, stop turning left, slow straight
    actions = discrete_action_to_continuous(torch.tensor([0, 1, 3]), "cpu")
    for t in range(1, E.SIMILARITY_WINDOW + 2):
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert not (terminated | truncated).any()
        if t < E.SIMILARITY_WINDOW:
            assert (env.pose_similarity == 0).all()
    turned = E.SIMILARITY_WINDOW * env.cfg.turn_rate * E.DT
    crawled = E.SIMILARITY_WINDOW * E.SPEED_FRACTIONS[1] * env.cfg.forward_speed * E.DT
    expected = torch.tensor([1.0, math.exp(-turned / env.SIMILARITY_YAW_SCALE), math.exp(-crawled / env.SIMILARITY_POS_SCALE)])
    assert torch.allclose(env.pose_similarity, expected, atol=1e-4)
    assert torch.allclose(env.last_reward_breakdown["rew_similarity"], expected * env.SIMILARITY_PENALTY, atol=1e-5)
    assert math.isclose(reward[0].item(), env.SIMILARITY_PENALTY, rel_tol=1e-6)  # env 0 never moves, so no progress term
    r = env.num_readings + E.PREV_ACTION_DIM
    seen = obs["dqn_state"][:, r:r + E.SIMILARITY_OBS_DIM]
    assert torch.allclose(seen, torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, turned], [crawled, 0.0, 0.0]]), atol=1e-4), seen
    env._sample_positions(torch.tensor([0]))
    assert env.pose_similarity[0] == 0 and (env.pose_hist_pos[0] == env.pos[0]).all()
    assert (env._get_obs()[0]["dqn_state"][0, r:r + E.SIMILARITY_OBS_DIM] == 0).all()
    print("  similarity penalty: zero for the first window, then by position and heading against the pose a window back")


if __name__ == "__main__":
    for test in (test_hand_checked_cases, test_surface_distance, test_matches_brute_force,
                 test_isaac_constants_match,
                 test_stage0_shortcuts_match_geometry, test_layout_invariants,
                 test_contact_crash_and_obs_wiring, test_proximity_penalty, test_camera_bins_tile_each_camera,
                 test_front_bins_tell_the_side, test_outcome_priority,
                 test_coords_mode_hand_checked, test_coords_mode_matches_geometry,
                 test_pre_reset_state_and_breakdown, test_reset_obs_starts_next_episode,
                 test_prev_action_in_state, test_pose_similarity_penalty):
        print(test.__name__)
        test()
    print("all tests passed")
