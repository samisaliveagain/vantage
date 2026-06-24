"""Reactive collision-avoidance environment (Gymnasium API).

A planar drone (point-mass with velocity-command dynamics, a reasonable proxy
for the inner-loop-stabilized quadrotor) must reach a goal through a field of
circular obstacles using only local lidar-style range sensing -- i.e. a
*reactive* policy with no global map, the complement to the A*/RRT* planners.

Observation (size 4 + n_rays):
    [goal_dx/scale, goal_dy/scale, vx, vy,  r_0 ... r_{n-1}]   (ranges normalised)
Action (size 2): desired velocity command in [-1, 1] (scaled to v_max).
Reward: progress toward goal - time penalty - proximity penalty
        (+goal bonus, -collision penalty terminally).
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _BASE = gym.Env
except Exception:  # pragma: no cover - allows import without gymnasium
    gym = None
    _BASE = object


class AvoidanceEnv(_BASE):
    metadata = {"render_modes": []}

    def __init__(self, size=10.0, n_obstacles=8, n_rays=16, v_max=1.5,
                 dt=0.1, max_steps=200, robot_radius=0.25, sensor_range=3.0,
                 seed=None):
        super().__init__()
        self.size = size
        self.n_obstacles = n_obstacles
        self.n_rays = n_rays
        self.v_max = v_max
        self.dt = dt
        self.max_steps = max_steps
        self.rr = robot_radius
        self.sensor_range = sensor_range
        self.rng = np.random.default_rng(seed)
        self.ray_angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

        self.obs_dim = 4 + n_rays
        self.act_dim = 2
        if gym is not None:
            self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)

        self.start = np.array([1.0, size / 2])
        self.goal = np.array([size - 1.0, size / 2])
        self._reset_internal()

    # ------------------------------------------------------------------ setup
    def _reset_internal(self):
        self.pos = self.start.copy()
        self.vel = np.zeros(2)
        self.t = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        self.obstacles = self._sample_obstacles()

    def _sample_obstacles(self):
        obs = []
        tries = 0
        while len(obs) < self.n_obstacles and tries < self.n_obstacles * 100:
            tries += 1
            c = self.rng.uniform([2.0, 0.5], [self.size - 2.0, self.size - 0.5])
            r = self.rng.uniform(0.4, 0.8)
            if np.linalg.norm(c - self.start) < r + 1.0:
                continue
            if np.linalg.norm(c - self.goal) < r + 1.0:
                continue
            obs.append((c[0], c[1], r))
        return obs

    # ---------------------------------------------------------------- sensing
    def _lidar(self):
        ranges = np.full(self.n_rays, self.sensor_range)
        for i, a in enumerate(self.ray_angles):
            d = np.array([np.cos(a), np.sin(a)])
            best = self.sensor_range
            for (cx, cy, r) in self.obstacles:
                # ray-circle intersection (ray from self.pos along d)
                oc = self.pos - np.array([cx, cy])
                b = 2 * oc @ d
                c = oc @ oc - r * r
                disc = b * b - 4 * c
                if disc < 0:
                    continue
                sq = np.sqrt(disc)
                for t in ((-b - sq) / 2, (-b + sq) / 2):
                    if 0 <= t < best:
                        best = t
            # walls
            ranges[i] = best
        return ranges

    def _obs(self):
        rel = (self.goal - self.pos) / self.size
        ranges = self._lidar() / self.sensor_range
        return np.concatenate([rel, self.vel / self.v_max, ranges]).astype(np.float32)

    def _min_clearance(self):
        if not self.obstacles:
            return np.inf
        return min(np.linalg.norm(self.pos - np.array([cx, cy])) - r
                   for (cx, cy, r) in self.obstacles)

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, float), -1.0, 1.0)
        self.vel = action * self.v_max
        self.pos = self.pos + self.vel * self.dt
        self.t += 1

        dist = np.linalg.norm(self.goal - self.pos)
        clearance = self._min_clearance()

        reward = 2.0 * (self.prev_dist - dist)      # progress
        reward -= 0.01                              # time penalty
        if clearance < 0.5:                         # proximity shaping
            reward -= 0.5 * (0.5 - clearance)
        self.prev_dist = dist

        terminated = False
        truncated = False
        info = {"clearance": clearance, "dist": dist}

        out_of_bounds = np.any(self.pos < 0) or np.any(self.pos > self.size)
        if clearance < 0 or out_of_bounds:
            reward -= 10.0
            terminated = True
            info["event"] = "collision"
        elif dist < 0.4:
            reward += 20.0
            terminated = True
            info["event"] = "goal"
        elif self.t >= self.max_steps:
            truncated = True
            info["event"] = "timeout"
        return self._obs(), float(reward), terminated, truncated, info
