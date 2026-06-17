"""A* path planning on a 3D occupancy grid (26-connected)."""
from __future__ import annotations

import heapq
import numpy as np

_NEIGHBORS = [(dx, dy, dz)
              for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
              if not (dx == 0 and dy == 0 and dz == 0)]


def _h(a, b):
    return float(np.linalg.norm(np.subtract(a, b)))


def astar_grid(grid: np.ndarray, start_idx, goal_idx):
    """Return list of grid indices from start to goal, or None if unreachable."""
    dims = grid.shape
    start = tuple(int(i) for i in start_idx)
    goal = tuple(int(i) for i in goal_idx)

    def valid(c):
        return all(0 <= c[i] < dims[i] for i in range(3)) and not grid[c]

    if not valid(start) or not valid(goal):
        return None

    open_heap = [(0.0, start)]
    came = {}
    g = {start: 0.0}
    closed = set()
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        for d in _NEIGHBORS:
            nb = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
            if not valid(nb) or nb in closed:
                continue
            tentative = g[cur] + _h(cur, nb)
            if tentative < g.get(nb, np.inf):
                g[nb] = tentative
                came[nb] = cur
                heapq.heappush(open_heap, (tentative + _h(nb, goal), nb))
    return None


def shortcut_path(path_world, world, robot_radius=0.2):
    """Greedy line-of-sight shortcutting to remove grid staircasing."""
    if not path_world or len(path_world) < 3:
        return path_world
    out = [path_world[0]]
    i = 0
    while i < len(path_world) - 1:
        j = len(path_world) - 1
        while j > i + 1:
            if world.segment_free(path_world[i], path_world[j], robot_radius):
                break
            j -= 1
        out.append(path_world[j])
        i = j
    return out
