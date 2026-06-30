"""Phase 4 — run a full inspection mission and render a demo (3D plot + GIF)."""
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio

from vantage.sim.world import World
from vantage.missions.mission import Mission

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def build_world():
    obs = [(3, 3, 2.0, 0.7), (5, 6, 2.5, 0.8), (7, 3.5, 2.2, 0.6),
           (4.5, 8, 2.0, 0.7), (8, 7, 2.5, 0.6), (2.5, 6.5, 2.3, 0.6),
           (6.5, 5, 1.8, 0.5)]
    return World(bounds=((0, 10), (0, 10), (0, 4)), obstacles=obs)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    world = build_world()
    mission = Mission("search-inspect-return", [
        (0.5, 0.5, 1.5),   # home / takeoff
        (9.0, 2.0, 2.5),   # inspect point A
        (8.5, 9.0, 2.0),   # inspect point B
        (2.0, 9.0, 3.0),   # inspect point C
        (0.5, 0.5, 1.5),   # return to home
    ])
    report, traj = mission.run(world)
    print("Mission report:")
    print(json.dumps({k: v for k, v in report.items() if k != "legs"}, indent=2))
    with open(os.path.join(RESULTS, "phase4_mission_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    if traj is None:
        print("mission failed to plan"); return

    # ---- 3D trajectory plot
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 16); v = np.linspace(0, np.pi, 8)
    for (cx, cy, cz, r) in world.obstacles:
        xs = cx + r * np.outer(np.cos(u), np.sin(v))
        ys = cy + r * np.outer(np.sin(u), np.sin(v))
        zs = cz + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color="0.6", alpha=0.3, linewidth=0)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="crimson", lw=2, label="flown")
    wp = np.array(mission.waypoints)
    ax.scatter(wp[:, 0], wp[:, 1], wp[:, 2], color="tab:blue", s=40, label="waypoints")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("VANTAGE Phase 4 — search/inspect/return mission")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "phase4_mission_3d.png"), dpi=120)

    # ---- top-down animated GIF
    frames = []
    th = np.linspace(0, 2 * np.pi, 40)
    step = max(1, len(traj) // 60)
    for i in range(1, len(traj), step):
        f, a = plt.subplots(figsize=(5, 5))
        for (cx, cy, cz, r) in world.obstacles:
            a.fill(cx + r * np.cos(th), cy + r * np.sin(th), color="0.6", alpha=0.5)
        a.plot(traj[:i, 0], traj[:i, 1], "-", color="crimson", lw=2)
        a.plot(traj[i - 1, 0], traj[i - 1, 1], "o", color="black", ms=7)
        a.plot(wp[:, 0], wp[:, 1], "b^", ms=8)
        a.set_xlim(0, 10); a.set_ylim(0, 10); a.set_title("VANTAGE mission (top view)")
        a.set_xticks([]); a.set_yticks([])
        f.tight_layout()
        f.canvas.draw()
        img = np.frombuffer(f.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(f.canvas.get_width_height()[::-1] + (4,))[..., :3]
        frames.append(img)
        plt.close(f)
    imageio.mimsave(os.path.join(RESULTS, "phase4_mission.gif"), frames, fps=15, loop=0)
    print(f"saved 3D plot + GIF ({len(frames)} frames)")


if __name__ == "__main__":
    main()
