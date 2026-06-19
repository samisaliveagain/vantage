"""Fly a planned path through the closed-loop quadrotor + controller.

First pass: drive the position setpoint to the next waypoint and advance when
within `lookahead`. Simple, but the controller overshoots fast waypoints and
cuts corners -- see Phase 2 benchmark.
"""
from __future__ import annotations

import numpy as np

from vantage.sim.quadrotor import Quadrotor
from vantage.control.geometric import CascadedController


def fly_path(world, path, robot_radius=0.2, dt=0.01, lookahead=0.6,
             cruise_speed=2.0, max_time=40.0):
    path = [np.asarray(p, float) for p in path]
    quad = Quadrotor(dt=dt)
    ctrl = CascadedController()
    quad.reset(position=path[0])

    flown = [quad.position]
    seg = 0
    t = 0.0
    collided = False
    min_clear = np.inf
    goal = path[-1]

    while t < max_time:
        pos = quad.position
        while seg < len(path) - 1 and np.linalg.norm(pos - path[seg]) < lookahead:
            seg += 1
        target = path[seg]
        dirv = target - pos
        nrm = np.linalg.norm(dirv)
        vel_sp = (dirv / nrm) * cruise_speed if nrm > 1e-6 else np.zeros(3)
        thrust, torque = ctrl.compute(quad.state, pos_sp=target, vel_sp=vel_sp)
        quad.step(thrust, torque)
        t += dt
        p = quad.position
        flown.append(p)
        c = world.clearance(p)
        min_clear = min(min_clear, c)
        if c < 0:
            collided = True
        if np.linalg.norm(p - goal) < 0.25 and np.linalg.norm(quad.velocity) < 0.6:
            break

    flown = np.array(flown)
    reached = np.linalg.norm(flown[-1] - goal) < 0.4
    return {
        "trajectory": flown, "success": bool(reached and not collided),
        "reached": bool(reached), "collided": bool(collided),
        "min_clearance": float(min_clear), "time_s": float(t),
        "flown_length": float(np.sum(np.linalg.norm(np.diff(flown, axis=0), axis=1))),
        "path_length": float(0.0),
    }
