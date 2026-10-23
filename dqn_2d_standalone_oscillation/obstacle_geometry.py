"""
Exact ground-truth [d_front, d_left, d_right] for the DQN state, computed
from obstacle geometry instead of read from a rendered depth camera.

Each reading is the distance from the drone's center to the nearest
obstacle surface inside a horizontal sector of half-angle `half_angle`
centered on the heading (front), heading + 90 deg (left) or heading - 90 deg
(right), capped at `max_range`. With the 73 deg sectors used here, bearings
between 36.5 and 53.5 deg off the front axis fall in no sector, the same
blind spots the three-camera layout has. Obstacles are vertical prisms
(upright cylinders and axis-aligned boxes) spanning the flight altitude, so
the horizontal slice at the drone's height gives the true 3D nearest point.
In the Isaac env that holds only below the 3 m top of the shortest one (Box0).

Why the result is exact rather than sampled: every obstacle footprint and
the sector (half-angle < 90 deg) are convex. If an obstacle's nearest point
to the drone lies inside the sector, it is the answer. Otherwise the
constrained minimum lies on one of the sector's two boundary rays, where it
is that ray's first hit.

The Isaac env does not use these functions. Its DQN reads the same quantity
from its depth cameras, so these exact values are the noise-free version of
those readings. camera_emulation.py also imports _ray_hit_circle and
_ray_hit_box.
"""
import math

import torch


def _ray_hit_circle(origin, direction, centers, radii):
    """Distance along each ray to its first hit on each circle, inf on a miss.

    origin (N, 2), direction (N, 2) unit vectors, centers (N, K, 2),
    radii (K,). Returns (N, K). Assumes origin lies outside every circle.
    """
    rel = centers - origin.unsqueeze(1)
    b = (rel * direction.unsqueeze(1)).sum(dim=-1)
    c = (rel * rel).sum(dim=-1) - radii * radii
    disc = b * b - c
    t = b - torch.sqrt(disc.clamp(min=0.0))
    hit = (disc >= 0.0) & (t >= 0.0)
    return torch.where(hit, t, torch.full_like(t, float("inf")))


def _ray_hit_box(origin, direction, centers, half_extents):
    """Distance along each ray to its first hit on each axis-aligned box,
    inf on a miss (slab test).

    origin (N, 2), direction (N, 2) unit vectors, centers (N, K, 2),
    half_extents (K, 2). Returns (N, K). Assumes origin lies outside every box.
    """
    d = direction.unsqueeze(1)
    # A zero component turns a ray lying on a slab face into 0/0. Only the
    # replacement's magnitude matters to the slab interval, not its sign.
    d = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    o = origin.unsqueeze(1)
    t1 = (centers - half_extents - o) / d
    t2 = (centers + half_extents - o) / d
    t_near = torch.minimum(t1, t2).amax(dim=-1)
    t_far = torch.maximum(t1, t2).amin(dim=-1)
    hit = (t_far >= t_near) & (t_near >= 0.0)
    return torch.where(hit, t_near, torch.full_like(t_near, float("inf")))


def _nearest_surface(pos, circle_xy, circle_r, box_xy, box_half):
    """Distance and bearing from pos to each obstacle's nearest surface
    point, circles first and then boxes, as two (N, Kc + Kb) tensors.
    Distance is 0 when pos lies inside the obstacle.
    """
    rel_c = circle_xy - pos.unsqueeze(1)
    dist_c = (rel_c.norm(dim=-1) - circle_r).clamp(min=0.0)
    p = pos.unsqueeze(1).expand_as(box_xy)
    rel_b = torch.maximum(torch.minimum(p, box_xy + box_half), box_xy - box_half) - p
    dist_b = rel_b.norm(dim=-1)
    rel = torch.cat([rel_c, rel_b], dim=1)
    bearing = torch.atan2(rel[..., 1], rel[..., 0])
    return torch.cat([dist_c, dist_b], dim=1), bearing


def obstacle_sector_distances(pos, yaw, circle_xy, circle_r, box_xy, box_half, half_angle, max_range, sectors=None):
    """[d_front, d_left, d_right] as an (N, 3) tensor, or one column per entry of sectors.

    pos (N, 2) and yaw (N,) are the drone's position and heading, in the
    same frame as the obstacle centers. circle_xy (N, Kc, 2) with radii
    circle_r (Kc,), box_xy (N, Kb, 2) with half extents box_half (Kb, 2).
    sectors is a sequence of (offset, half width) pairs in radians relative to
    the heading. When given, half_angle is ignored.
    Needs at least one obstacle. A reading of 0 means the drone's center is
    inside an obstacle.
    """
    if sectors is None:
        sectors = ((0.0, half_angle), (math.pi / 2.0, half_angle), (-math.pi / 2.0, half_angle))
    dist, bearing = _nearest_surface(pos, circle_xy, circle_r, box_xy, box_half)
    inside = dist <= 0.0
    readings = []
    for offset, width in sectors:
        center = (yaw + offset).unsqueeze(1)
        off_axis = torch.atan2(torch.sin(bearing - center), torch.cos(bearing - center))
        in_sector = off_axis.abs() <= width
        edge_hits = []
        for edge in (center - width, center + width):
            u = torch.stack([torch.cos(edge[:, 0]), torch.sin(edge[:, 0])], dim=-1)
            edge_hits.append(torch.cat([
                _ray_hit_circle(pos, u, circle_xy, circle_r),
                _ray_hit_box(pos, u, box_xy, box_half),
            ], dim=1))
        d = torch.where(in_sector, dist, torch.minimum(edge_hits[0], edge_hits[1]))
        d = torch.where(inside, torch.zeros_like(d), d)
        # Obstacles parked 1000 m or more away get float32 hits that are imprecise but
        # still far, so this clamp alone keeps them out of every reading.
        readings.append(d.amin(dim=1).clamp(max=max_range))
    return torch.stack(readings, dim=-1)


def obstacle_surface_distance(pos, circle_xy, circle_r, box_xy, box_half):
    """(N,) distance from pos to the nearest obstacle surface, 0 inside one."""
    dist, _ = _nearest_surface(pos, circle_xy, circle_r, box_xy, box_half)
    return dist.amin(dim=1)
