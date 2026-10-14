"""
Tests for camera_emulation.py. Runs as a plain script, no pytest:

    python test_camera_emulation.py

The reference is obstacle_sector_distances from obstacle_geometry.py, an
independent exact method already checked against brute force there.
"""
import ast
import math
import os

import torch

import camera_emulation as cam
import dqn_2d_env as E
import obstacle_geometry as og
from dqn_config import DQNConfig

f = lambda x: torch.tensor(x, dtype=torch.float32)
FAR_C, FAR_B = (1000.0, 1000.0, 0.3), (1000.0, 1001.0, 0.5, 0.5)
ISAAC_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "isaaclab_env", "drone_nav_v2_env_dqn.py")


def _scene(circles=(), boxes=()):
    circles, boxes = list(circles) or [FAR_C], list(boxes) or [FAR_B]
    return (f([[c[:2] for c in circles]]), f([c[2] for c in circles]),
            f([[b[:2] for b in boxes]]), f([b[2:] for b in boxes]))


def _both(pos, yaw, **obstacles):
    scene = _scene(**obstacles)
    p, y = f([pos]), f([yaw])
    exact = og.obstacle_sector_distances(p, y, *scene, math.atan(cam.TAN_HALF_FOV), cam.FAR_CLIP)[0].tolist()
    return cam.camera_readings(p, y, *scene)[0].tolist(), exact


def test_hand_checked_cases():
    c, e = _both((0, 0), 0.0)
    assert c == [10.0, 10.0, 10.0] and e == [10.0, 10.0, 10.0], "nothing in range"
    # Dead ahead, the front camera sits 0.25 m closer than the drone's center.
    c, e = _both((0, 0), 0.0, circles=[(5, 0, 1)])
    assert abs(e[0] - 4.0) < 1e-4 and abs(c[0] - 3.75) < 2e-3, (c, e)
    # A 0.3 m cylinder straight to the left at 5 m subtends +/-3.6 deg from the left
    # camera, inside the 10.6 deg gap between its two central rays, so it goes unseen.
    c, e = _both((0, 0), 0.0, circles=[(0, 5, 0.3)])
    assert c[1] == 10.0 and abs(e[1] - 4.7) < 1e-4, (c, e)
    # The same cylinder widened to 0.6 m reaches those rays.
    c, e = _both((0, 0), 0.0, circles=[(0, 5, 0.6)])
    assert c[1] < 4.6, (c, e)
    # The front camera's 84 rays are about 0.9 deg apart, so it sees the thin one.
    c, e = _both((0, 0), 0.0, circles=[(5, 0, 0.3)])
    assert abs(c[0] - 4.45) < 1e-2, (c, e)
    c, e = _both((0, 0), 0.0, circles=[(12, 0, 1)])
    assert c[0] == 10.0, "beyond the far clip"
    print("  hand-checked cases: ok")


def test_planar_walls():
    """A wall 5 m out on each side pins that camera's mount point, facing and slant.
    The central column's pixel ray leaves at atan(u) sideways and half a pixel (v = u)
    vertically, so it meets a wall 4.75 m from the camera at 4.75 * sqrt(1 + 2 u^2)."""
    walls = [(0, (55, 0, 50, 50), 84), (1, (0, 55, 50, 50), 8), (2, (0, -55, 50, 50), 8)]
    for column, box, columns in walls:
        u = 0.5 / (columns / 2) * cam.TAN_HALF_FOV
        c, _ = _both((0, 0), 0.0, boxes=[box])
        # The front camera's slant adds only 1.8e-4 m here, so the tolerance must sit well below it.
        assert abs(c[column] - 4.75 * math.sqrt(1 + 2 * u * u)) < 2e-5, (column, c)
    print("  planar walls: mount points, facings and slant ok")


def test_near_clip_is_per_obstacle():
    """A surface inside NEAR_CLIP is cut away, and the ray goes on to what lies behind it."""
    # A 0.1 m cylinder whose face is 0.05 m from the front camera covers every front ray.
    c, _ = _both((0, 0), 0.0, circles=[(0.40, 0, 0.1), (4.0, 0, 0.25)])
    assert abs(c[0] - 3.5) < 5e-3, c
    c, _ = _both((0, 0), 0.0, circles=[(0.40, 0, 0.1)])
    assert c[0] == 10.0, c
    print("  near clip: clipped surface reveals the obstacle behind it")


def _isaac_cameras():
    """{prim name: (pos, width, height, focal, aperture, clipping range)} from TiledCameraCfg calls."""
    with open(ISAAC_ENV, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    cams = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "TiledCameraCfg":
            kw = {k.arg: k.value for k in node.keywords}
            offset = {k.arg: k.value for k in kw["offset"].keywords}
            spawn = {k.arg: ast.literal_eval(k.value) for k in kw["spawn"].keywords}
            name = ast.literal_eval(kw["prim_path"]).rsplit("/", 1)[-1]
            cams[name] = (ast.literal_eval(offset["pos"]), ast.literal_eval(kw["width"]), ast.literal_eval(kw["height"]),
                          spawn["focal_length"], spawn["horizontal_aperture"], spawn["clipping_range"])
    return cams


def test_constants_match_isaac():
    """The camera table must keep matching the Isaac env, or the two simulations feed the
    network different readings with every other test still green."""
    isaac = _isaac_cameras()
    for prim, camera in zip(("depth_camera", "depth_camera_left", "depth_camera_right"), cam.CAMERAS):
        pos, width, height, focal, aperture, clip = isaac[prim]
        assert pos[:2] == (camera.mount_x, camera.mount_y), (prim, pos)
        assert width == height == camera.columns, (prim, width, height)
        assert math.isclose(aperture / (2.0 * focal), cam.TAN_HALF_FOV), prim
        assert clip == (cam.NEAR_CLIP, cam.FAR_CLIP), (prim, clip)
    # camera_readings assumes no camera is inside an obstacle before a crash.
    assert max(math.hypot(c.mount_x, c.mount_y) for c in cam.CAMERAS) < E.DRONE_RADIUS
    print("  camera table matches the Isaac TiledCameraCfg calls")


def _random_scene(n, gen):
    pos = (torch.rand(n, 2, generator=gen) * 2 - 1) * 2.0
    yaw = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
    circle_xy = (torch.rand(n, 3, 2, generator=gen) * 2 - 1) * 8.0
    box_xy = (torch.rand(n, 2, 2, generator=gen) * 2 - 1) * 8.0
    return pos, yaw, circle_xy, f([0.3, 0.6, 1.0]), box_xy, f([[0.5, 0.5], [0.9, 0.3]])


def _exact_from_cameras(pos, yaw, circle_xy, circle_r, box_xy, box_half):
    """Exact nearest return inside each camera's full FOV, seen from that camera's mount point."""
    out = []
    for camera in cam.CAMERAS:
        c, s = torch.cos(yaw), torch.sin(yaw)
        origin = pos + torch.stack([camera.mount_x * c - camera.mount_y * s,
                                    camera.mount_x * s + camera.mount_y * c], dim=-1)
        sector = og.obstacle_sector_distances(origin, yaw + camera.yaw_offset, circle_xy, circle_r, box_xy, box_half,
                                              math.atan(cam.TAN_HALF_FOV), cam.FAR_CLIP)
        out.append(sector[:, 0])
    return torch.stack(out, dim=-1)


def test_never_closer_than_exact():
    """A camera samples a subset of its FOV's directions and slant only lengthens a return,
    so a reading can never be shorter than the exact distance from the same mount point."""
    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        scene = _random_scene(400, gen)
        camera = cam.camera_readings(*scene)
        exact = _exact_from_cameras(*scene)
        visible = exact >= cam.NEAR_CLIP  # the near clip can hide a return entirely
        assert (camera[visible] >= exact[visible] - 1e-4).all()
    print("  camera reading >= exact distance from the same mount point: ok")


def test_dense_camera_converges_to_exact():
    """With thousands of columns the gaps between rays vanish, and only the slant and the
    half-pixel margin at the FOV edge separate the camera from the exact reading."""
    gen = torch.Generator().manual_seed(1)
    dense = tuple(c._replace(columns=2001) for c in cam.CAMERAS)
    scene = _random_scene(300, gen)
    camera = cam.camera_readings(*scene, cameras=dense)
    exact = _exact_from_cameras(*scene)
    visible = exact >= cam.NEAR_CLIP
    gap = (camera - exact)[visible].abs()
    assert torch.quantile(gap, 0.99) < 5e-3, f"p99 gap {torch.quantile(gap, 0.99):.4f}"
    print(f"  2001-column cameras vs exact: median gap {gap.median():.1e} m, p99 {torch.quantile(gap, 0.99):.1e} m")


def test_env_camera_mode():
    """Stage 0 shortcut and obstacle wiring through DroneNav2DEnv in camera mode."""
    torch.manual_seed(0)
    env = E.DroneNav2DEnv(256, DQNConfig(), "cpu", num_obstacles=0, distance_mode="camera")
    obs, _ = env.reset()
    obstacles = (env.obstacle_xy[:, :E.NUM_CYLINDERS], f(E.CYLINDER_RADII),
                 env.obstacle_xy[:, E.NUM_CYLINDERS:], f(E.BOX_HALF_EXTENTS))
    assert torch.equal(obs["dqn_state"][:, :3], cam.camera_readings(env.pos, env.yaw, *obstacles))
    for k in (1, 2):
        env = E.DroneNav2DEnv(256, DQNConfig(), "cpu", num_obstacles=k, distance_mode="camera")
        obs, _ = env.reset()
        obstacles = (env.obstacle_xy[:, :E.NUM_CYLINDERS], f(E.CYLINDER_RADII),
                     env.obstacle_xy[:, E.NUM_CYLINDERS:], f(E.BOX_HALF_EXTENTS))
        assert torch.equal(obs["dqn_state"][:, :3], cam.camera_readings(env.pos, env.yaw, *obstacles))
    try:
        E.DroneNav2DEnv(1, DQNConfig(), "cpu", distance_mode="lidar")
        raise AssertionError("an unknown distance_mode must be rejected")
    except ValueError:
        pass
    print("  env camera mode: Stage 0 shortcut and obstacle readings ok")


if __name__ == "__main__":
    for test in (test_hand_checked_cases, test_planar_walls, test_near_clip_is_per_obstacle,
                 test_constants_match_isaac, test_never_closer_than_exact,
                 test_dense_camera_converges_to_exact, test_env_camera_mode):
        print(test.__name__)
        test()
    print("all tests passed")
