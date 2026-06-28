"""Phase 3 head-to-head: learned reactive PPO policy vs classical A* planner.

Same randomized obstacle layouts for both. A* gets a full occupancy map and
plans a global path; PPO sees only local lidar and reacts. We report the
classic trade-off: planning quality/safety vs. map-free reactivity and the
per-decision compute cost.
"""
from __future__ import annotations

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vantage.rl.avoidance_env import AvoidanceEnv
from vantage.rl.ppo import PPOAgent
from vantage.sim.world import World
from vantage.planning.astar import astar_grid, shortcut_path
from vantage.utils.metrics import path_length
from vantage.utils.report import write_csv, write_markdown_table

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
CKPT = os.path.join(RESULTS, "phase3_policy.npz")


def world_from_env(env):
    obs3d = [(cx, cy, 0.5, r) for (cx, cy, r) in env.obstacles]
    return World(bounds=((0, env.size), (0, env.size), (0, 1)), obstacles=obs3d)


def run_ppo(agent, env):
    o, _ = env.reset(seed=env._seed)
    path = [env.pos.copy()]
    t_inf = 0.0
    event = "timeout"
    for _ in range(env.max_steps):
        t0 = time.perf_counter()
        a, _ = agent.act(o, deterministic=True)
        t_inf += time.perf_counter() - t0
        o, r, term, trunc, info = env.step(a)
        path.append(env.pos.copy())
        if term or trunc:
            event = info.get("event", "timeout")
            break
    return dict(event=event, success=event == "goal", collided=event == "collision",
                length=path_length(np.array(path)), compute_ms=t_inf * 1e3,
                path=np.array(path))


def run_astar(env):
    w = world_from_env(env)
    start = np.array([env.start[0], env.start[1], 0.5])
    goal = np.array([env.goal[0], env.goal[1], 0.5])
    t0 = time.perf_counter()
    grid, res, origin = w.occupancy_grid(resolution=0.2, robot_radius=env.rr)
    s = w.world_to_grid(start, origin, res)
    g = w.world_to_grid(goal, origin, res)
    idx = astar_grid(grid, s, g)
    compute_ms = (time.perf_counter() - t0) * 1e3
    if idx is None:
        return dict(event="no_path", success=False, collided=False,
                    length=np.nan, compute_ms=compute_ms, path=None)
    wp = [w.grid_to_world(i, origin, res) for i in idx]
    wp = shortcut_path(wp, w, env.rr)
    p2d = np.array([[x, y] for (x, y, z) in wp])
    return dict(event="goal", success=True, collided=False,
                length=path_length(p2d), compute_ms=compute_ms, path=p2d)


def main(n_trials=40):
    agent = PPOAgent.__new__(PPOAgent)
    PPOAgent.__init__(agent, 20, 2)
    agent.load(CKPT)

    rows = {"ppo": [], "astar": []}
    example = None
    for k in range(n_trials):
        env = AvoidanceEnv(n_obstacles=6, seed=1000 + k)
        env._seed = 1000 + k
        rp = run_ppo(agent, env)
        ra = run_astar(env)
        rows["ppo"].append(rp); rows["astar"].append(ra)
        if example is None and rp["success"] and ra["success"]:
            example = (env, rp["path"], ra["path"])

    def agg(rs):
        succ = np.mean([r["success"] for r in rs])
        coll = np.mean([r["collided"] for r in rs])
        lens = [r["length"] for r in rs if r["success"] and not np.isnan(r["length"])]
        comp = [r["compute_ms"] for r in rs]
        return dict(success_rate=succ, collision_rate=coll,
                    mean_length=np.mean(lens) if lens else float("nan"),
                    mean_compute_ms=np.mean(comp))

    A = {k: agg(v) for k, v in rows.items()}
    print(f"Phase 3 — PPO (reactive) vs A* (global) over {n_trials} worlds:")
    hdr = f"{'method':7s} {'success':>8s} {'collision':>10s} {'len':>7s} {'compute(ms)':>12s}"
    print(hdr); print("-" * len(hdr))
    for n, a in A.items():
        print(f"{n:7s} {a['success_rate']:8.2f} {a['collision_rate']:10.2f} "
              f"{a['mean_length']:7.2f} {a['mean_compute_ms']:12.3f}")

    write_csv(os.path.join(RESULTS, "phase3_metrics.csv"),
              [{"method": n, **{k: round(v, 4) for k, v in a.items()}} for n, a in A.items()],
              ["method", "success_rate", "collision_rate", "mean_length", "mean_compute_ms"])
    write_markdown_table(
        os.path.join(RESULTS, "phase3_metrics.md"),
        f"Phase 3 — PPO (reactive, map-free) vs A* (global planner), {n_trials} worlds",
        ["method", "success", "collision", "mean_len", "compute_ms/decision"],
        [(n.upper(), f"{a['success_rate']:.2f}", f"{a['collision_rate']:.2f}",
          f"{a['mean_length']:.2f}", f"{a['mean_compute_ms']:.3f}") for n, a in A.items()],
        notes="A* plans once over a full occupancy map (safe by construction but needs the map). "
              "PPO uses only 16 lidar rays + goal bearing, reacts per-step. compute_ms is per "
              "planning call (A*) vs per policy forward pass (PPO).")

    if example is not None:
        env, pp, pa = example
        fig, ax = plt.subplots(figsize=(6.5, 6))
        th = np.linspace(0, 2 * np.pi, 40)
        for (cx, cy, r) in env.obstacles:
            ax.fill(cx + r * np.cos(th), cy + r * np.sin(th), color="0.6", alpha=0.5)
        ax.plot(pa[:, 0], pa[:, 1], "b--", lw=2, label="A* global plan")
        ax.plot(pp[:, 0], pp[:, 1], "r-", lw=2, label="PPO reactive")
        ax.plot(*env.start, "g^", ms=12, label="start")
        ax.plot(*env.goal, "r*", ms=16, label="goal")
        ax.set_title("Phase 3 — reactive PPO vs global A*"); ax.legend(); ax.axis("equal")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "phase3_compare.png"), dpi=120)
        print("saved comparison plot")


if __name__ == "__main__":
    main()
