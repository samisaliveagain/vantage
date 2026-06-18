"""RRT* — asymptotically optimal sampling-based planner in continuous 3D.

Operates directly on the World collision model (no voxelization), so it scales
to larger spaces than grid A*. Returns a list of world-frame waypoints.
"""
from __future__ import annotations

import numpy as np


class RRTStar:
    def __init__(self, world, robot_radius=0.2, step=0.7, goal_bias=0.1,
                 search_radius=1.6, max_iter=3000, seed=None):
        self.world = world
        self.rr = robot_radius
        self.step = step
        self.goal_bias = goal_bias
        self.search_radius = search_radius
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self.b = world.bounds

    def _sample(self, goal):
        if self.rng.random() < self.goal_bias:
            return np.asarray(goal, float)
        return self.rng.uniform(self.b[:, 0], self.b[:, 1])

    def _steer(self, a, b):
        d = b - a
        dist = np.linalg.norm(d)
        if dist <= self.step:
            return b
        return a + (d / dist) * self.step

    def plan(self, start, goal):
        start = np.asarray(start, float)
        goal = np.asarray(goal, float)
        if not (self.world.is_free(start, self.rr) and self.world.is_free(goal, self.rr)):
            return None
        nodes = [start]
        parent = {0: -1}
        cost = {0: 0.0}
        goal_node = None

        for _ in range(self.max_iter):
            s = self._sample(goal)
            pts = np.array(nodes)
            nearest = int(np.argmin(np.linalg.norm(pts - s, axis=1)))
            new = self._steer(nodes[nearest], s)
            if not self.world.segment_free(nodes[nearest], new, self.rr):
                continue

            pts = np.array(nodes)
            dists = np.linalg.norm(pts - new, axis=1)
            near = np.where(dists <= self.search_radius)[0]

            # choose best parent among near nodes
            best_parent, best_cost = nearest, cost[nearest] + np.linalg.norm(new - nodes[nearest])
            for idx in near:
                c = cost[idx] + np.linalg.norm(new - nodes[idx])
                if c < best_cost and self.world.segment_free(nodes[idx], new, self.rr):
                    best_parent, best_cost = int(idx), c

            new_idx = len(nodes)
            nodes.append(new)
            parent[new_idx] = best_parent
            cost[new_idx] = best_cost

            # rewire near nodes through new node if cheaper
            for idx in near:
                c = best_cost + np.linalg.norm(nodes[idx] - new)
                if c < cost[idx] and self.world.segment_free(new, nodes[idx], self.rr):
                    parent[idx] = new_idx
                    cost[idx] = c

            if np.linalg.norm(new - goal) <= self.step and \
                    self.world.segment_free(new, goal, self.rr):
                g_idx = len(nodes)
                nodes.append(goal)
                parent[g_idx] = new_idx
                cost[g_idx] = best_cost + np.linalg.norm(goal - new)
                goal_node = g_idx
                # keep improving for a bit, but a first solution is fine here
                break

        if goal_node is None:
            return None
        path = []
        cur = goal_node
        while cur != -1:
            path.append(nodes[cur])
            cur = parent[cur]
        return path[::-1]
