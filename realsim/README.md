# realsim/ — PX4 SITL + Gazebo (the real 3D simulator)

The top-level `vantage` package is a fast, headless **physics** simulator that
produces metrics, plots, and the demo GIF. This folder is the **3D simulator**
path: PX4 flight stack + Gazebo, run in WSL2, where you watch the quadrotor fly
in a real-time window.

Start here: **[00_RUNBOOK.md](00_RUNBOOK.md)**

| file | what |
|------|------|
| `check_wsl.sh` | environment readiness check |
| `install_px4_gazebo.sh` | one-shot PX4 + Gazebo installer (Ubuntu 22.04 / WSL) |
| `run_sim.sh` | launch PX4 SITL + Gazebo x500 (3D window) |
| `fly_mission.py` | MAVSDK offboard script — flies the VANTAGE mission in the sim |

> These scripts use the standard upstream PX4 commands. They have **not** been
> executed inside this build environment (no GPU/WSL here), so treat them as the
> tested-per-upstream-docs starting point and report any errors back.
