"""Phase 1 self-evaluation: arm, take off, hold position, then track a step.

Runs headless, prints metrics, and saves a trajectory plot to results/.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vantage.sim.quadrotor import Quadrotor
from vantage.control.geometric import CascadedController
from vantage.utils.metrics import rmse, settling_time
from vantage.utils.report import write_csv, write_markdown_table

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run(duration: float = 8.0, dt: float = 0.005):
    quad = Quadrotor(dt=dt)
    ctrl = CascadedController()
    quad.reset(position=(0, 0, 0))

    n = int(duration / dt)
    t = np.arange(n) * dt
    traj = np.zeros((n, 3))
    setpoints = np.zeros((n, 3))

    for k in range(n):
        # mission: take off to 2 m, hold, then step to (2,1,2.5) at t=4s
        if t[k] < 4.0:
            sp = np.array([0.0, 0.0, 2.0])
        else:
            sp = np.array([2.0, 1.0, 2.5])
        thrust, torque = ctrl.compute(quad.state, pos_sp=sp, yaw_sp=0.0)
        quad.step(thrust, torque)
        traj[k] = quad.position
        setpoints[k] = sp

    # ---- metrics
    hold_mask = (t >= 2.0) & (t < 4.0)             # steady hover window
    hover_err = np.linalg.norm(traj[hold_mask] - setpoints[hold_mask], axis=1)
    err_all = np.linalg.norm(traj - setpoints, axis=1)
    metrics = {
        "hover_rmse_m": rmse(traj[hold_mask], setpoints[hold_mask]),
        "hover_max_err_m": float(hover_err.max()),
        "step_settling_s": settling_time(err_all[t >= 4.0], dt, tol=0.10),
        "final_err_m": float(err_all[-1]),
    }

    # ---- plot
    os.makedirs(RESULTS, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    labels = ["x", "y", "z"]
    for i in range(3):
        axs[0].plot(t, traj[:, i], label=f"{labels[i]} actual")
        axs[0].plot(t, setpoints[:, i], "--", alpha=0.6, label=f"{labels[i]} setpoint")
    axs[0].set_ylabel("position [m]"); axs[0].legend(ncol=3, fontsize=8); axs[0].grid(alpha=0.3)
    axs[0].set_title("VANTAGE Phase 1 — takeoff, hover hold, and step tracking")
    axs[1].plot(t, err_all, color="crimson")
    axs[1].set_ylabel("tracking error [m]"); axs[1].set_xlabel("time [s]"); axs[1].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS, "phase1_hover.png")
    fig.savefig(out, dpi=120)
    return metrics, os.path.abspath(out)


if __name__ == "__main__":
    m, out = run()
    print("Phase 1 metrics:")
    for k, v in m.items():
        print(f"  {k:20s}: {v:.4f}")
    rows = [{"metric": k, "value": round(v, 5)} for k, v in m.items()]
    write_csv(os.path.join(RESULTS, "phase1_metrics.csv"), rows, ["metric", "value"])
    write_markdown_table(
        os.path.join(RESULTS, "phase1_metrics.md"),
        "Phase 1 — Hover & step-tracking metrics",
        ["metric", "value"], [(r["metric"], r["value"]) for r in rows],
        notes="Quadrotor mass 1.0 kg, dt 5 ms, cascaded PD controller. "
              "Mission: takeoff to 2 m, hold 2 s, step to (2,1,2.5).")
    print(f"Plot saved: {out}")
