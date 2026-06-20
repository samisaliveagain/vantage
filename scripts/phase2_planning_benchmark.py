"""Phase 2 self-evaluation: A* vs RRT* over random cluttered worlds.

For each random world we plan with both planners, fly the path through the
closed-loop quadrotor sim, and record success / collision / path-length /
plan-time. Aggregates a metrics table and saves a comparison figure.
"""
from __future__ import annotations

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vantage.sim.world import World
from vantage.planning.astar import astar_grid, shortcut_path
from vantage.planning.rrt_star import RRTStar
from vantage.planning.follow import fly_path
from vantage.utils.metrics import summarize
from vantage.utils.report import write_csv, write_markdown_table

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
BOUNDS = ((0, 10), (0, 10), (1, 4))
START = np.array([0.5, 0.5, 1.5])
GOAL = np.array([9.5, 9.5, 2.5])
ROBOT_R = 0.2


def plan_astar(world):
    grid, res, origin = world.occupancy_grid(resolution=0.25, robot_radius=ROBOT_R)
    s = world.world_to_grid(START, origin, res)
    g = world.world_to_grid(GOAL, origin, res)
    idx_path = astar_grid(grid, s, g)
    if idx_path is None:
        return None
    wp = [world.grid_to_world(i, origin, res) for i in idx_path]
    return shortcut_path(wp, world, ROBOT_R)


def plan_rrt(world, seed):
    return RRTStar(world, robot_radius=ROBOT_R, seed=seed).plan(START, GOAL)


def run(n_trials=15, seed0=0):
    rows = {"astar": [], "rrt": []}
    example = None
    for k in range(n_trials):
        world = World.random(n_obstacles=14, bounds=BOUNDS, seed=seed0 + k,
                             keep_clear=(START, GOAL), clear_radius=1.0)
        for name, planner in (("astar", lambda w: plan_astar(w)),
                              ("rrt", lambda w: plan_rrt(w, seed0 + k))):
            t0 = time.perf_counter()
            path = planner(world)
            plan_t = time.perf_counter() - t0
            if path is None:
                rows[name].append(dict(success=0, reached=0, collided=0,
                                       length=np.nan, clear=np.nan,
                                       plan_ms=plan_t * 1e3, found=0))
                continue
            res = fly_path(world, path, robot_radius=ROBOT_R)
            rows[name].append(dict(
                success=int(res["success"]), reached=int(res["reached"]),
                collided=int(res["collided"]), length=res["flown_length"],
                clear=res["min_clearance"], plan_ms=plan_t * 1e3, found=1))
            if example is None and name == "astar" and res["success"]:
                example = (world, path, res["trajectory"])
    return rows, example


def aggregate(rows):
    out = {}
    for name, r in rows.items():
        found = [x for x in r if x["found"]]
        succ = np.mean([x["success"] for x in r]) if r else 0
        coll = np.mean([x["collided"] for x in found]) if found else 0
        lens = [x["length"] for x in found if not np.isnan(x["length"])]
        clears = [x["clear"] for x in found if not np.isnan(x["clear"])]
        ptimes = [x["plan_ms"] for x in r]
        out[name] = dict(
            success_rate=succ, found_rate=np.mean([x["found"] for x in r]),
            collision_rate=coll,
            mean_length=np.mean(lens) if lens else float("nan"),
            mean_clearance=np.mean(clears) if clears else float("nan"),
            mean_plan_ms=np.mean(ptimes))
    return out


def plot_example(example, agg):
    os.makedirs(RESULTS, exist_ok=True)
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    world, path, traj = example
    th = np.linspace(0, 2 * np.pi, 40)
    for (cx, cy, cz, r) in world.obstacles:
        ax.fill(cx + r * np.cos(th), cy + r * np.sin(th), color="0.6", alpha=0.5)
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "o--", color="tab:blue", label="A* plan (shortcut)")
    ax.plot(traj[:, 0], traj[:, 1], "-", color="crimson", lw=2, label="flown trajectory")
    ax.plot(*START[:2], "g^", ms=12, label="start")
    ax.plot(*GOAL[:2], "r*", ms=16, label="goal")
    ax.set_title("Phase 2 — plan + flown trajectory (top view)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.legend(fontsize=8); ax.axis("equal")

    ax2 = fig.add_subplot(1, 2, 2)
    names = list(agg.keys())
    metrics = ["success_rate", "collision_rate", "found_rate"]
    x = np.arange(len(metrics)); w = 0.35
    for i, n in enumerate(names):
        ax2.bar(x + i * w, [agg[n][m] for m in metrics], w, label=n.upper())
    ax2.set_xticks(x + w / 2); ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_ylim(0, 1.05); ax2.set_title("A* vs RRT* (rates)"); ax2.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS, "phase2_planning.png")
    fig.savefig(out, dpi=120)
    return os.path.abspath(out)


if __name__ == "__main__":
    rows, example = run(n_trials=15)
    agg = aggregate(rows)
    print("Phase 2 — A* vs RRT* over random worlds:")
    hdr = f"{'planner':8s} {'success':>8s} {'found':>7s} {'collision':>10s} {'len(m)':>8s} {'clear(m)':>9s} {'plan(ms)':>9s}"
    print(hdr); print("-" * len(hdr))
    for n, a in agg.items():
        print(f"{n:8s} {a['success_rate']:8.2f} {a['found_rate']:7.2f} "
              f"{a['collision_rate']:10.2f} {a['mean_length']:8.2f} "
              f"{a['mean_clearance']:9.2f} {a['mean_plan_ms']:9.1f}")
    csv_rows, md_rows = [], []
    for n, a in agg.items():
        csv_rows.append({"planner": n, **{k: round(v, 4) for k, v in a.items()}})
        md_rows.append((n.upper(), f"{a['success_rate']:.2f}", f"{a['found_rate']:.2f}",
                        f"{a['collision_rate']:.2f}", f"{a['mean_length']:.2f}",
                        f"{a['mean_clearance']:.2f}", f"{a['mean_plan_ms']:.1f}"))
    write_csv(os.path.join(RESULTS, "phase2_metrics.csv"), csv_rows,
              ["planner", "success_rate", "found_rate", "collision_rate",
               "mean_length", "mean_clearance", "mean_plan_ms"])
    write_markdown_table(
        os.path.join(RESULTS, "phase2_metrics.md"),
        "Phase 2 — A* vs RRT* (15 random worlds, 14 obstacles each)",
        ["planner", "success", "found", "collision", "len(m)", "clear(m)", "plan(ms)"],
        md_rows,
        notes="Start (0.5,0.5,1.5) -> goal (9.5,9.5,2.5). Paths flown through the "
              "closed-loop quadrotor sim (pure-pursuit). success = reached AND no collision.")
    if example is not None:
        print("Plot saved:", plot_example(example, agg))
