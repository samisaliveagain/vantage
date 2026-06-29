"""Mission layer: sequence behaviors into a full autonomous flight.

A Mission is an ordered list of waypoints the drone must visit (e.g. takeoff ->
inspect points -> return-to-home). For each leg we plan a global path with A*,
concatenate, and fly the whole route through the closed-loop quadrotor sim with
pure-pursuit. The runner returns a structured report with per-leg and overall
metrics -- the same numbers a real autonomy stack would log.
"""
from __future__ import annotations

import numpy as np

from vantage.planning.astar import astar_grid, shortcut_path
from vantage.planning.follow import fly_path
from vantage.utils.metrics import path_length


def plan_leg(world, start, goal, resolution=0.25, robot_radius=0.2):
    grid, res, origin = world.occupancy_grid(resolution=resolution, robot_radius=robot_radius)
    s = world.world_to_grid(start, origin, res)
    g = world.world_to_grid(goal, origin, res)
    idx = astar_grid(grid, s, g)
    if idx is None:
        return None
    wp = [world.grid_to_world(i, origin, res) for i in idx]
    return shortcut_path(wp, world, robot_radius)


class Mission:
    def __init__(self, name, waypoints):
        self.name = name
        self.waypoints = [np.asarray(w, float) for w in waypoints]

    def run(self, world, robot_radius=0.2):
        legs = []
        full_path = [self.waypoints[0]]
        ok = True
        for a, b in zip(self.waypoints[:-1], self.waypoints[1:]):
            path = plan_leg(world, a, b, robot_radius=robot_radius)
            if path is None:
                legs.append({"from": a.tolist(), "to": b.tolist(), "planned": False})
                ok = False
                break
            legs.append({"from": a.tolist(), "to": b.tolist(), "planned": True,
                         "plan_length": path_length(np.array(path))})
            full_path.extend(path[1:])

        report = {"mission": self.name, "planned_ok": ok, "legs": legs}
        if not ok:
            report["success"] = False
            return report, None

        result = fly_path(world, full_path, robot_radius=robot_radius, max_time=120.0)
        report.update({
            "success": result["success"],
            "collided": result["collided"],
            "min_clearance_m": result["min_clearance"],
            "flown_length_m": result["flown_length"],
            "plan_length_m": path_length(np.array(full_path)),
            "flight_time_s": result["time_s"],
            "n_waypoints": len(self.waypoints),
        })
        return report, result["trajectory"]
