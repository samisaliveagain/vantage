"""World model: spherical obstacle field + 3D occupancy grid.

Spheres are a cheap but expressive proxy for clutter (trees, poles, drones).
The occupancy grid voxelizes them (inflated by the drone radius) so grid-based
planners can treat the robot as a point.
"""
from __future__ import annotations

import numpy as np


class World:
    def __init__(self, bounds=((0, 10), (0, 10), (0, 5)), obstacles=None):
        self.bounds = np.asarray(bounds, dtype=float)  # (3,2)
        # each obstacle: (cx, cy, cz, radius)
        self.obstacles = list(obstacles) if obstacles is not None else []

    # --------------------------------------------------------------- sampling
    @classmethod
    def random(cls, n_obstacles=12, bounds=((0, 10), (0, 10), (0, 5)),
               r_range=(0.4, 0.9), seed=None, keep_clear=((0, 0, 1.5), (10, 10, 1.5)),
               clear_radius=1.2):
        rng = np.random.default_rng(seed)
        b = np.asarray(bounds, float)
        obs = []
        tries = 0
        while len(obs) < n_obstacles and tries < n_obstacles * 200:
            tries += 1
            c = rng.uniform(b[:, 0], b[:, 1])
            r = rng.uniform(*r_range)
            # keep start & goal regions clear
            ok = True
            for kc in keep_clear:
                if np.linalg.norm(c - np.asarray(kc)) < r + clear_radius:
                    ok = False
                    break
            if ok:
                obs.append((float(c[0]), float(c[1]), float(c[2]), float(r)))
        return cls(bounds=bounds, obstacles=obs)

    # ------------------------------------------------------------- collision
    def is_free(self, p, robot_radius=0.2) -> bool:
        p = np.asarray(p, float)
        if np.any(p < self.bounds[:, 0]) or np.any(p > self.bounds[:, 1]):
            return False
        for (cx, cy, cz, r) in self.obstacles:
            if np.linalg.norm(p - np.array([cx, cy, cz])) <= r + robot_radius:
                return False
        return True

    def segment_free(self, a, b, robot_radius=0.2, step=0.1) -> bool:
        a = np.asarray(a, float); b = np.asarray(b, float)
        d = np.linalg.norm(b - a)
        n = max(2, int(d / step))
        for t in np.linspace(0, 1, n):
            if not self.is_free(a + t * (b - a), robot_radius):
                return False
        return True

    def clearance(self, p) -> float:
        p = np.asarray(p, float)
        if not self.obstacles:
            return np.inf
        return min(np.linalg.norm(p - np.array([cx, cy, cz])) - r
                   for (cx, cy, cz, r) in self.obstacles)

    # ----------------------------------------------------------------- grid
    def occupancy_grid(self, resolution=0.25, robot_radius=0.2):
        b = self.bounds
        dims = np.ceil((b[:, 1] - b[:, 0]) / resolution).astype(int) + 1
        grid = np.zeros(dims, dtype=bool)  # True = occupied
        # voxel centers
        xs = b[0, 0] + np.arange(dims[0]) * resolution
        ys = b[1, 0] + np.arange(dims[1]) * resolution
        zs = b[2, 0] + np.arange(dims[2]) * resolution
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        for (cx, cy, cz, r) in self.obstacles:
            d2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
            grid |= d2 <= (r + robot_radius) ** 2
        return grid, resolution, b[:, 0].copy()

    def world_to_grid(self, p, origin, resolution):
        return tuple(np.round((np.asarray(p, float) - origin) / resolution).astype(int))

    def grid_to_world(self, idx, origin, resolution):
        return origin + np.asarray(idx, float) * resolution
