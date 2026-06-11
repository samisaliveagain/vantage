"""Reusable evaluation metrics shared across phases."""
from __future__ import annotations

import numpy as np


def rmse(traj: np.ndarray, reference: np.ndarray) -> float:
    """Root-mean-square position error between trajectory and reference."""
    traj = np.asarray(traj)
    reference = np.asarray(reference)
    err = np.linalg.norm(traj - reference, axis=-1)
    return float(np.sqrt(np.mean(err ** 2)))


def settling_time(err: np.ndarray, dt: float, tol: float = 0.05) -> float:
    """Time after which error stays within `tol` (m). Returns np.inf if never."""
    err = np.asarray(err)
    within = err <= tol
    for i in range(len(within)):
        if within[i:].all():
            return i * dt
    return float("inf")


def path_length(path: np.ndarray) -> float:
    path = np.asarray(path)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def min_obstacle_clearance(path: np.ndarray, obstacles) -> float:
    """Minimum distance from any path point to any spherical obstacle surface."""
    path = np.asarray(path)
    worst = np.inf
    for (cx, cy, cz, r) in obstacles:
        d = np.linalg.norm(path - np.array([cx, cy, cz]), axis=1) - r
        worst = min(worst, float(d.min()))
    return worst


def summarize(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "n": int(v.size),
    }
