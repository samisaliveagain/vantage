"""Fly a planned path through the closed-loop quadrotor + controller.

Uses pure-pursuit: the path is densified into a fine polyline and the
controller chases a "carrot" point a fixed arc-length ahead of the drone's
closest point on the path, with deceleration near the goal. This yields smooth,
near-shortest flown trajectories instead of waypoint-to-waypoint overshoot.
"""
from __future__ import annotations

import numpy as np

from vantage.sim.quadrotor import Quadrotor
from vantage.control.geometric import CascadedController


def _densify(path, ds=0.1):
    path = [np.asarray(p, float) for p in path]
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        d = np.linalg.norm(b - a)
        n = max(1, int(d / ds))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return np.array(out)


def fly_path(world, path, robot_radius=0.2, dt=0.01, lookahead=0.8,
             cruise_speed=1.6, max_time=40.0):
    """Return dict with flown trajectory, success flag, clearance, time, length."""
    pts = _densify(path, ds=0.1)
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seglen)])
    total = arc[-1]
    goal = pts[-1]

    quad = Quadrotor(dt=dt)
    ctrl = CascadedController()
    quad.reset(position=pts[0])

    flown = [quad.position]
    t = 0.0
    collided = False
    min_clear = np.inf
    closest = 0            # monotonic progress index (never moves backward)
    window = 40            # only look a bounded distance ahead on the path

    while t < max_time:
        pos = quad.position
        # advance the progress index forward only, within a local window --
        # robust to self-intersecting paths (e.g. return-to-home loops)
        hi = min(closest + window, len(pts))
        local = int(np.argmin(np.linalg.norm(pts[closest:hi] - pos, axis=1)))
        closest = closest + local
        target_arc = min(arc[closest] + lookahead, total)
        ci = int(np.searchsorted(arc, target_arc))
        ci = min(ci, len(pts) - 1)
        carrot = pts[ci]

        tan = pts[min(ci + 1, len(pts) - 1)] - pts[max(ci - 1, 0)]
        tn = np.linalg.norm(tan)
        tan = tan / tn if tn > 1e-6 else np.zeros(3)

        dist_goal = total - arc[closest]
        speed = min(cruise_speed, max(0.3, dist_goal))
        vel_sp = tan * speed

        thrust, torque = ctrl.compute(quad.state, pos_sp=carrot, vel_sp=vel_sp)
        quad.step(thrust, torque)
        t += dt

        p = quad.position
        flown.append(p)
        c = world.clearance(p)
        min_clear = min(min_clear, c)
        if c < 0:
            collided = True
        # require the drone to have traversed ~the whole path before finishing
        # (otherwise closed-loop missions that return home would end instantly)
        if (arc[closest] > 0.9 * total and np.linalg.norm(p - goal) < 0.25
                and np.linalg.norm(quad.velocity) < 0.5):
            break

    flown = np.array(flown)
    reached = np.linalg.norm(flown[-1] - goal) < 0.4
    return {
        "trajectory": flown,
        "success": bool(reached and not collided),
        "reached": bool(reached),
        "collided": bool(collided),
        "min_clearance": float(min_clear),
        "time_s": float(t),
        "flown_length": float(np.sum(np.linalg.norm(np.diff(flown, axis=0), axis=1))),
        "path_length": float(total),
    }
