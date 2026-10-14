"""
Camera-geometry emulation of the Isaac env's three depth cameras, the
sensor counterpart of the exact readings in obstacle_geometry.py.

Each camera casts one ray per pixel column of a pinhole camera and reports
the nearest hit among them, as the Isaac env takes the minimum over a
camera's ground-filtered pixels. Everything the 2D world can reproduce is
copied from the Isaac env's _setup_scene, and test_camera_emulation.py
fails if these values drift from it:
- the 73 deg HFOV pinhole model (24 mm focal length, 35.5 mm aperture),
- 84 columns on the front camera and 8 on each lateral one, so a thin
  obstacle can sit between two rays and go unseen,
- mount points 0.25 m off the drone's center, so distances are measured
  from the camera and not from the center,
- the 0.1 to 10 m clipping range.
Rendering noise, ground-filter errors, attitude tilt and self-occlusion by
the airframe have no 2D counterpart and are not modeled.

Assumptions not checked against Isaac's renderer:
- one ray through each pixel center, with no sub-pixel coverage or
  anti-aliasing that could reveal an obstacle between two centers,
- a surface closer than NEAR_CLIP is cut away and the ray continues past it,
  so a drone about to crash can read 10 m straight ahead,
- the lateral cameras face +90 (left) and -90 deg (right), the intended
  mounting, which the Isaac env's comments say no smoke test has confirmed.

Obstacles are vertical prisms spanning the flight altitude, so each column's
nearest return comes from its row closest to the horizon. Those rows sit
half a pixel off the horizon, which lengthens a return by a slant factor of
at most 1.00004 on the front camera and 1.0043 on the 8-pixel lateral ones.
"""
import math
from typing import NamedTuple

import torch

from obstacle_geometry import _ray_hit_box, _ray_hit_circle

TAN_HALF_FOV = 35.5 / (2.0 * 24.0)
NEAR_CLIP = 0.1
FAR_CLIP = 10.0


class Camera(NamedTuple):
    mount_x: float      # body frame, forward
    mount_y: float      # body frame, left
    yaw_offset: float   # relative to the heading
    columns: int        # pixel columns of a square image


# In [d_front, d_left, d_right] order.
CAMERAS = (
    Camera(0.25, 0.0, 0.0, 84),
    Camera(0.0, 0.25, math.pi / 2.0, 8),
    Camera(0.0, -0.25, -math.pi / 2.0, 8),
)


def _pixel_rays(columns, device):
    """Horizontal angle of each column's ray and the slant factor of its
    row nearest the horizon, both (columns,)."""
    u = ((torch.arange(columns, device=device, dtype=torch.float32) + 0.5) - columns / 2.0) / (columns / 2.0) * TAN_HALF_FOV
    v = 0.5 / (columns / 2.0) * TAN_HALF_FOV
    return torch.atan(u), torch.sqrt(1.0 + v * v / (1.0 + u * u))


def camera_readings(pos, yaw, circle_xy, circle_r, box_xy, box_half, cameras=CAMERAS):
    """[d_front, d_left, d_right] as the three cameras would report them, (N, 3).

    Arguments as in obstacle_sector_distances. Assumes no camera sits inside
    an obstacle, which holds before the drone's center reaches DRONE_RADIUS
    of a surface because every mount point is closer to the center than that.
    """
    n = pos.shape[0]
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    readings = []
    for camera in cameras:
        origin = pos + torch.stack([camera.mount_x * cos_yaw - camera.mount_y * sin_yaw,
                                    camera.mount_x * sin_yaw + camera.mount_y * cos_yaw], dim=-1)
        angle, slant = _pixel_rays(camera.columns, pos.device)
        ray_yaw = (yaw + camera.yaw_offset).unsqueeze(1) + angle  # (N, columns)
        # Every ray becomes a row of its own, so one call tests all columns at once.
        direction = torch.stack([torch.cos(ray_yaw), torch.sin(ray_yaw)], dim=-1).reshape(-1, 2)
        o = origin.repeat_interleave(camera.columns, dim=0)
        hits = torch.cat([
            _ray_hit_circle(o, direction, circle_xy.repeat_interleave(camera.columns, dim=0), circle_r),
            _ray_hit_box(o, direction, box_xy.repeat_interleave(camera.columns, dim=0), box_half),
        ], dim=1) * slant.repeat(n).unsqueeze(1)  # (N * columns, K)
        # Clip per obstacle, before taking the nearest, so a clipped surface
        # does not hide an obstacle farther along the same ray.
        hits = torch.where(hits < NEAR_CLIP, torch.full_like(hits, float("inf")), hits)
        readings.append(hits.amin(dim=1).view(n, camera.columns).amin(dim=1).clamp(max=FAR_CLIP))
    return torch.stack(readings, dim=-1)
