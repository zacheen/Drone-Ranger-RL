"""
Tests for obstacle_geometry.py and for how dqn_2d_env.py uses it. Runs as a
plain script, no pytest:

    python test_obstacle_geometry.py

The brute-force reference samples points densely along every obstacle's
boundary and keeps the nearest one inside the sector. It shares no code with
the ray intersection it checks, so agreement means the exact method is right.
"""
import ast
import inspect
import math
import os

import torch

import dqn_2d_env as E
import obstacle_geometry as og
from dqn_config import DQNConfig

HALF = math.radians(36.5)
MAX_RANGE = 10.0
ISAAC_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "isaaclab_env", "drone_nav_v2_env_dqn.py")
COPIED_FUNCTIONS = ["_ray_hit_circle", "_ray_hit_box", "_nearest_surface", "obstacle_sector_distances"]


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


def test_isaac_copy_matches():
    isaac = {n.name: n for n in _isaac_tree().body if isinstance(n, ast.FunctionDef)}
    local = {n.name: n for n in ast.parse(inspect.getsource(og)).body if isinstance(n, ast.FunctionDef)}
    for name in COPIED_FUNCTIONS:
        assert name in isaac, f"{name} missing from {ISAAC_ENV}"
        assert ast.dump(isaac[name]) == ast.dump(local[name]), f"{name} differs between the two copies"
    print(f"  Isaac Lab copy: {len(COPIED_FUNCTIONS)} functions identical")


def test_isaac_constants_match():
    """The values the two environments feed the shared functions must agree too,
    or the network sees different distances in each with every test still green."""
    tree = _isaac_tree()
    assert _isaac_literal(tree, "_OBSTACLE_CYLINDER_RADII") == E.CYLINDER_RADII
    sizes = _isaac_literal(tree, "_OBSTACLE_BOX_SIZES_XY")
    assert tuple(tuple(s / 2 for s in b) for b in sizes) == E.BOX_HALF_EXTENTS
    assert _isaac_literal(tree, "_DQN_MAX_RANGE") == E.MAX_SENSE_RANGE
    half_fov = math.atan(_isaac_literal(tree, "_horizontal_aperture") / (2.0 * _isaac_literal(tree, "_focal_length")))
    assert math.isclose(half_fov, E.SENSOR_HALF_FOV)
    drone_radius = _isaac_literal(tree, "drone_radius")
    assert drone_radius == E.DRONE_RADIUS
    # _obstacle_radius is built from expressions such as `0.8 + drone_radius`, so evaluate them.
    radius_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and getattr(n.targets[0], "attr", None) == "_obstacle_radius")
    placement = tuple(eval(compile(ast.Expression(e), ISAAC_ENV, "eval"), {"drone_radius": drone_radius})
                      for e in radius_node.value.args[0].elts)
    assert all(math.isclose(a, b) for a, b in zip(placement, E.PLACEMENT_RADII)) and len(placement) == len(E.PLACEMENT_RADII)
    assert _isaac_literal(tree, "obstacle_min_spacing") == E.OBSTACLE_MIN_SPACING
    assert _isaac_literal(tree, "REHEARSAL_PROB") == E.REHEARSAL_PROB
    print("  Isaac Lab constants: footprints, range, FOV, layout radii, spacing and rehearsal match")


def _env_obstacles(env):
    return (env.obstacle_xy[:, :E.NUM_CYLINDERS], torch.tensor(E.CYLINDER_RADII),
            env.obstacle_xy[:, E.NUM_CYLINDERS:], torch.tensor(E.BOX_HALF_EXTENTS))


def test_stage0_shortcuts_match_geometry():
    """With num_obstacles=0 the env skips the geometry. Both shortcuts must equal what it would compute."""
    torch.manual_seed(0)
    env = E.DroneNav2DEnv(512, DQNConfig(), "cpu", num_obstacles=0)
    obs, _ = env.reset()
    cx, cr, bx, bh = _env_obstacles(env)
    full = og.obstacle_sector_distances(env.pos, env.yaw, cx, cr, bx, bh, E.SENSOR_HALF_FOV, E.MAX_SENSE_RANGE)
    assert torch.equal(obs["dqn_state"][:, :3], full)
    assert (og.obstacle_surface_distance(env.pos, cx, cr, bx, bh) >= E.DRONE_RADIUS).all()
    print("  Stage 0 shortcuts: identical to the full geometry")


def test_layout_invariants():
    for k in range(1, E.MAX_OBSTACLES + 1):
        torch.manual_seed(k)
        env = E.DroneNav2DEnv(2000, DQNConfig(), "cpu", num_obstacles=k)
        obs, _ = env.reset()
        active = env.obstacle_active
        assert active[:, 1:k].all() and not active[:, k:].any(), f"k={k} active mask"
        assert 0.4 < (~active[:, 0]).float().mean() < 0.6, f"k={k} rehearsal fraction"
        parked_far = (env.obstacle_xy - env.pos.unsqueeze(1)).norm(dim=-1) > 100
        assert torch.equal(parked_far, ~active), f"k={k} parked obstacles must be far away"
        cx, cr, bx, bh = _env_obstacles(env)
        assert (og.obstacle_surface_distance(env.pos, cx, cr, bx, bh) > E.DRONE_RADIUS).all(), f"k={k} spawn in contact"
        # The one-at-a-time target push (see _sample_obstacle_layout) can leave a
        # target within DRONE_RADIUS of a surface but never inside an obstacle, and
        # REACHED_THRESHOLD still lets the drone reach it from a clear side.
        assert (og.obstacle_surface_distance(env.target, cx, cr, bx, bh) > 0).all(), f"k={k} target inside an obstacle"
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
    env.obstacle_xy[0, 1] = torch.tensor([0.64, 0.0])  # cylinder r=0.3, surface 0.34 m ahead, so one 0.05 m step ends within DRONE_RADIUS
    env.obstacle_xy[1, 3] = torch.tensor([0.84, 0.0])  # box half 0.5, same geometry
    env.obstacle_xy[2, 1] = torch.tensor([5.0, 0.0])   # clear
    env.yaw[3] = math.pi / 2
    env.obstacle_xy[3, 1] = torch.tensor([0.0, 3.0])   # ahead only because of the heading
    d = env._get_obs()[0]["dqn_state"]
    assert abs(d[2, 0].item() - 4.7) < 1e-4 and abs(d[3, 0].item() - 2.7) < 1e-4
    _, reward, terminated, _, _ = env.step(torch.tensor([[10.0, 0.0, 0.0]]).expand(4, -1))  # full-speed forward
    assert terminated.tolist() == [True, True, False, False]
    assert (reward[:2] < -49).all() and (reward[2:] > -1).all()
    assert env.last_outcomes()[:2] == ["contact_crash"] * 2
    print("  crash detection and obstacle readings through step(): ok")


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
    assert env.state_dim == 8
    env.reset()
    env.obstacle_xy[:] = env._parked_xy
    env.obstacle_active[:] = True
    env.pos[:] = 0.0
    env.yaw[:] = 0.0
    env.target[:] = torch.tensor([3.0, 4.0])
    env.obstacle_xy[0, 1] = torch.tensor([2.0, 0.0])   # cylinder r=0.3, surface point (1.7, 0)
    env.obstacle_xy[0, 3] = torch.tensor([0.0, -3.0])  # box half 0.5, surface point (0, -2.5)
    env.yaw[1] = math.pi / 2
    env.obstacle_xy[1, 1] = torch.tensor([0.0, 2.0])   # world +y is body +x at this heading
    env.obstacle_xy[2, 1] = torch.tensor([12.0, 0.0])  # surface 11.7 m away, beyond sensing range
    env.obstacle_xy[2, 2] = torch.tensor([1.0, 0.0])
    env.obstacle_active[2, 2] = False                  # parked by rehearsal mixing, so never seen
    s = env._get_obs()[0]["dqn_state"]
    expected = torch.tensor([
        [1.7, 0.0, 1.0, 0.0, -2.5, 1.0],
        [1.7, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    assert torch.allclose(s[:, :6], expected, atol=1e-5), s
    assert torch.allclose(s[0, 6:], torch.tensor([3.0, 4.0])) and torch.allclose(s[1, 6:], torch.tensor([4.0, -3.0]), atol=1e-5)
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
    env.target[:] = torch.tensor([0.78, 0.0])  # one 0.05 m step brings it within REACHED_THRESHOLD
    _, reward, terminated, _, _ = env.step(torch.tensor([[10.0, 0.0, 0.0]]))
    assert terminated.item()
    end = env.pre_reset_state
    assert torch.allclose(end["pos"][0], torch.tensor([0.05, 0.0]), atol=1e-4)
    assert torch.allclose(end["target"][0], torch.tensor([0.78, 0.0]))
    assert not torch.equal(env.target[0], end["target"][0])  # resampled for the next episode
    assert env.last_reward_breakdown["rew_reached"].item() == 100.0
    assert torch.allclose(sum(env.last_reward_breakdown[k] for k in ("rew_progress", "rew_reached", "rew_crashed")), reward)
    print("  pre-reset snapshot and reward breakdown: hold the finished episode's last step")


if __name__ == "__main__":
    for test in (test_hand_checked_cases, test_surface_distance, test_matches_brute_force,
                 test_isaac_copy_matches, test_isaac_constants_match,
                 test_stage0_shortcuts_match_geometry, test_layout_invariants,
                 test_contact_crash_and_obs_wiring, test_outcome_priority,
                 test_coords_mode_hand_checked, test_coords_mode_matches_geometry,
                 test_pre_reset_state_and_breakdown):
        print(test.__name__)
        test()
    print("all tests passed")
