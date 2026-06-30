# VANTAGE — Full Project Documentation

A sim-first autonomous-drone software stack. A simulated quadrotor flies through
cluttered, GPS-denied space using onboard-style sensing. It **plans** a route,
**controls** the aircraft along it, and a **reinforcement-learning policy**
learns to dodge obstacles reactively. Everything is proven in simulation, with
committed metrics, plots, tests, and a demo.

This document explains every folder and file, the build phases, the methods
chosen (and why, vs. the alternatives), how the controller relates to PX4, and
how the computation and training were actually done.

---

## 1. Two simulators (the single most important thing to understand)

| | Headless physics sim (`vantage/`) | 3D simulator (`realsim/`) |
|---|---|---|
| What | Quadrotor physics integrated in code | PX4 flight stack + Gazebo |
| Output | Numbers, plots, a GIF in `results/` | A live 3D window you watch |
| Runs where | Anywhere (CPU), and in CI | Linux / WSL2 |
| Status | **Built, tested, all results come from here** | **Scaffold + runbook** |

Everything below (phases, metrics, the 96% RL policy) comes from the **headless
physics sim**. `realsim/` is the optional path to a real-time 3D view and the
real PX4 firmware — the scripts are written and documented but have not been run.

---

## 2. Repository map (folder by folder)

```
VLA drone project/
├── vantage/            the library — the actual algorithms (importable package)
│   ├── sim/            the simulator: quadrotor physics + the obstacle world
│   ├── control/        the flight controller (keeps the drone on target)
│   ├── planning/       route planning (A*, RRT*) + path-following executor
│   ├── rl/             reinforcement learning: the environment + PPO (from scratch)
│   ├── missions/       stitches planning+control into a full multi-waypoint flight
│   └── utils/          shared helpers: metrics + report (csv/markdown) writers
├── scripts/            runnable entry points — one per phase + a GPU trainer + aggregator
├── tests/              automated tests (pytest) that prove each phase works
├── results/            committed outputs: metrics (.md/.csv), plots (.png), the demo GIF, trained policies
├── realsim/            PX4 + Gazebo (real 3D sim) setup scripts + runbook
├── docs/               the project plan (docx) + auto-generated BENCHMARKS.md
├── .github/workflows/  CI: runs the tests on every push
├── pyproject.toml      package definition + dependencies (pip install -e .)
└── requirements.txt    the dependency list
```

### Why this layout
`vantage/` is a proper **installable Python package** (not loose scripts) so any
script or test imports the same code (`from vantage.planning.astar import ...`).
That is what real robotics codebases do: a reusable library, thin scripts on top,
and tests that pin behaviour. Splitting by capability (sim / control / planning /
rl) mirrors the actual autonomy pipeline and lets each piece be demoed alone.

---

## 3. File-by-file reference (input → output → what it does)

### `vantage/` — the library

| File | Input → Output | What it does |
|------|----------------|--------------|
| `sim/quadrotor.py` | (thrust, torque) + state → next state | Nonlinear 6-DOF rigid-body quadrotor, integrated with **RK4**. Gravity, full 3D rotation, gyroscopic coupling, optional motor lag. This is "the physics." |
| `sim/world.py` | bounds + obstacles (or a random seed) → occupancy grid + collision/clearance queries | The environment: a field of spherical obstacles, plus a voxel **occupancy grid** for the planner and fast collision checks. |
| `control/geometric.py` | state + position/yaw setpoint → (thrust, torque) | **Cascaded PD controller** (PX4-style outer position loop + inner attitude loop). Turns "go here" into motor-level commands. |
| `planning/astar.py` | grid + start + goal → grid path (+ line-of-sight shortcut) | **A\*** shortest-path search on the occupancy grid, then a shortcut pass to remove staircasing. |
| `planning/rrt_star.py` | world + start + goal → list of waypoints | **RRT\*** sampling-based planner in continuous 3D (no grid); asymptotically optimal via rewiring. |
| `planning/follow.py` | world + planned path → flown trajectory + metrics | **Pure-pursuit executor**: flies the planned path through the closed-loop sim (controller + quadrotor) and records what actually happened. |
| `rl/avoidance_env.py` | action (velocity cmd) → observation, reward, done | **Gymnasium environment** for reactive avoidance: 16-ray lidar + goal bearing in, velocity command out, shaped reward. |
| `rl/ppo.py` | rollouts (obs, actions, rewards) → updated policy + saved weights | **PPO implemented from scratch in NumPy** — MLP with hand-written backprop, GAE, clipped objective, Adam. No deep-learning framework. |
| `missions/mission.py` | world + waypoint list → mission report + trajectory | Plans each leg with A\*, flies the whole route, returns per-leg + overall metrics. The "full autonomy" layer. |
| `utils/metrics.py` | trajectories/paths → scalar metrics | RMSE, settling time, path length, obstacle clearance. |
| `utils/report.py` | metric rows → `.md` + `.csv` files | Writes the committed, auditable metric files. |

### `scripts/` — runnable entry points

| File | Input → Output | What it does |
|------|----------------|--------------|
| `phase1_hover.py` | (config in code) → `results/phase1_hover.png` + metrics | Phase 1 self-test: takeoff, hover-hold, step tracking. |
| `phase2_planning_benchmark.py` | (random seeds) → `results/phase2_planning.png` + metrics | Phase 2: **A\* vs RRT\*** over 15 random worlds. |
| `phase3_train.py` | `--updates --steps [--resume]` → `results/phase3_policy.npz` + learning curve | Trains the avoidance policy with the **NumPy** PPO (CPU, CI-friendly). |
| `phase3_train_torch.py` | `--updates --steps [--device]` → `results/phase3_policy_torch.pt` | Same algorithm in **PyTorch**, auto-uses **CUDA** (your RTX 4050). |
| `phase3_benchmark.py` | `phase3_policy.npz` → `results/phase3_compare.png` + metrics | Phase 3: learned policy **vs A\*** head-to-head. |
| `phase3_eval_torch.py` | `phase3_policy_torch.pt` → `results/phase3_torch_eval.txt` | Scores the **GPU-trained** policy over 100 episodes. |
| `phase4_mission.py` | (config in code) → `results/phase4_mission.gif` + 3D plot + report | Phase 4: full search/inspect/return mission + demo animation. |
| `run_all_benchmarks.py` | `results/*.csv/json` → `docs/BENCHMARKS.md` + `results/dashboard.png` | Aggregates every phase into one report + dashboard. |

### `tests/`
`test_phase1..4.py` — pytest cases that pin behaviour (free-fall physics, hover
equilibrium, closed-loop convergence, planners stay collision-free, the env
contract, PPO actually learns, closed-loop missions complete). 12 tests, run by CI.

### `realsim/` — the real 3D simulator path
`00_RUNBOOK.md` (step-by-step), `check_wsl.sh` (readiness), `install_px4_gazebo.sh`
(installer), `run_sim.sh` (launch Gazebo), `fly_mission.py` (MAVSDK script that
flies the mission in PX4+Gazebo). See `realsim/README.md`.

### Helper launchers (Windows, for GPU training on your machine)
`run_gpu_training.bat` (venv + CUDA torch + train), `eval_torch.bat` (score the
trained policy).

---

## 4. The phases (what was built, in order)

| Phase | Theme | Built | Headline result |
|------|-------|-------|-----------------|
| **M0** | Foundation | package, packaging, CI, tests harness | green CI, reproducible env |
| **M1** | Control | RK4 quadrotor + cascaded controller | hover **2.8 mm** RMSE, step settles **1.3 s** |
| **M2** | Mapping + Planning | occupancy grid, A\*, RRT\*, pure-pursuit follower | **100% success, 0 collisions** over random clutter |
| **M3** | Learning | Gym env + **PPO from scratch** + GPU trainer | **96% success / 4% collision** (trained on RTX 4050) |
| **M4** | Missions + Eval | mission runner, benchmark suite, demo GIF, dashboard | full autonomous mission, near-perfect path tracking |

Each phase ends with a committed artifact (plot + metrics) and tests, so the git
history shows steady, verifiable progress — including the real bugs hit and fixed
(Euler→RK4 integration drift, controller gain tuning, follower overshoot, a PPO
gradient-shape bug, RNG nondeterminism, and pure-pursuit looping on return-to-home).

---

## 5. Methods chosen — and why, vs. the alternatives

**Dynamics: nonlinear rigid body + RK4 integration.**
Chosen because it captures the real coupled physics (gravity, full rotation,
gyroscopic effects) while running thousands of steps/second on a CPU. RK4 over
plain Euler because Euler accumulated energy/attitude error and the hover drifted
— a bug actually hit and fixed. *Not* a full aero/motor simulator (e.g. PyBullet
or Gazebo physics) because that needs heavy deps/GPU and isn't necessary to
validate the planning/control/RL logic.

**Control: cascaded PD ("geometric") controller.** See §6 — this is the PX4-style
choice. *Not* a single big controller or a pure LQR/MPC because the cascaded
position→attitude structure is exactly what real autopilots use, is easy to tune,
and runs fast. MPC was considered (it's in the plan as a stretch) but is far
heavier to implement and tune for the same demo value.

**Planning: A\* (grid) and RRT\* (continuous).** Both, deliberately, to show the
classic trade-off. A\* is optimal on the grid and fast for our sizes; RRT\* scales
to larger continuous spaces without voxelizing. *Not* Dijkstra (A\* with a
heuristic is strictly better here), and PRM/plain RRT were skipped because RRT\*
gives near-optimal paths with rewiring.

**Mapping: voxel occupancy grid (inflated by drone radius).** Simple, standard,
and lets the planner treat the drone as a point. *Not* a full ESDF/voxblox map —
that's a stretch goal; the occupancy grid is enough to plan safe paths.

**Path following: pure-pursuit with a carrot point.** Smooth, near-shortest flown
paths. Replaced an earlier naive waypoint-chaser that overshot into obstacles
(72 m flown vs 13 m optimal) — another real bug→fix.

**RL: PPO, implemented from scratch in NumPy (PyTorch version for GPU).**
PPO because it's the stable, industry-default on-policy algorithm for continuous
control. *From scratch* because it's a stronger demonstration of understanding
than importing Stable-Baselines3 — every line (the clipped surrogate, GAE, Adam,
backprop) is visible and owned. *Not* DQN (that's for discrete actions; ours are
continuous velocities), and *not* SAC (off-policy, more moving parts) — PPO is the
right first choice and what most sim-to-real drone RL papers start with.

**Environment: lidar-style rays + goal bearing.** A *reactive*, map-free observation
(the complement to the global planners) so the comparison "learned reactive vs
classical global" is meaningful.

---

## 6. The controller vs PX4 — how it's *like* PX4, and how it isn't

**How it is like PX4.** Real PX4 uses a **cascaded control architecture**: an outer
loop tracks position/velocity and produces a desired attitude + thrust, and an
inner loop tracks that attitude and produces torques/motor commands. `geometric.py`
is exactly this structure — outer PD on position → desired tilt + collective
thrust, inner PD on attitude → body torques. It also mirrors PX4's **offboard**
control idea (you command position/velocity setpoints and the controller realizes
them), which is why the MAVSDK `realsim/fly_mission.py` maps onto it cleanly.

**How it is *not* PX4.** PX4 is a large, flight-tested **firmware** that also
includes: the EKF2 state estimator fusing IMU/GPS/vision, a full sensor/driver
layer, a control-allocation "mixer" for real motors/ESCs, failsafes, flight modes,
and parameter/telemetry systems. Our controller is a compact ~80-line Python
re-implementation of just the **control math**, running against an idealized state
(no sensor noise, no estimator, no mixer). It demonstrates the same *principles*
PX4 uses, but it is not the PX4 codebase and doesn't carry its robustness.

**Why this was chosen.** The goal was to validate the *autonomy stack*
(planning + control + RL) quickly, on any machine, with zero GPU/firmware setup.
A lightweight PX4-style controller does that and keeps the whole project
`pip install`-able and CI-tested. The honest upgrade path is `realsim/`: run the
*actual* PX4 firmware in Gazebo and command it via MAVSDK — at which point the
project genuinely "runs on PX4."

**What you can truthfully say:** "PX4-style cascaded controller; architecture
designed to target PX4 SITL (MAVSDK offboard scaffolded)" — **not** "built on PX4"
until `realsim/` is actually run.

---

## 7. How computation, training, and everything was done

**Development / iteration.** The library and all headless results were built and
run on CPU (numpy/scipy/matplotlib) — fast enough to run every benchmark in
seconds and to keep CI green. No GPU is needed for any of the headless results.

**RL training — two runs, both real:**
- *NumPy PPO (CPU):* used during development; reached ~98% success over 80 updates
  on the obstacle course. This produced `results/phase3_policy.npz` and the
  learning curve.
- *PyTorch PPO (GPU):* trained on the **NVIDIA RTX 4050 Laptop GPU** (CUDA 12.x).
  Confirmed via `nvidia-smi` (Python running as a CUDA compute process) and the
  script printing `device = cuda (NVIDIA GeForce RTX 4050 Laptop GPU)`. Completed
  120 updates and saved `results/phase3_policy_torch.pt`. **Evaluation: 96%
  success, 4% collision over 100 randomized courses** (`results/phase3_torch_eval.txt`).
  Note: the network is tiny, so the GPU is lightly loaded — the per-step bottleneck
  is the Python physics-env loop, not the GPU math. The GPU run is genuine; it is
  just not GPU-bound.

**Reproducibility.** Fixed seeds throughout; the PPO uses its own seeded RNG so
runs are deterministic. `pip install -e .` + `pytest` reproduces the test suite;
each `scripts/phaseN_*.py` regenerates its own metrics into `results/`;
`run_all_benchmarks.py` rebuilds `docs/BENCHMARKS.md` and the dashboard.

**Environment.** Windows host + an isolated `.venv` (Python 3.11). GPU training
installs PyTorch (CUDA build) into that venv; everything else is CPU-only.

---

## 8. Results summary

| Phase | Metric | Value |
|------|--------|-------|
| M1 control | hover RMSE / step settling | 2.8 mm / 1.32 s |
| M2 planning (A\*) | success / collision / flown len | 1.00 / 0.00 / ~13 m |
| M2 planning (RRT\*) | success / collision / flown len | 1.00 / 0.00 / ~14 m |
| M3 RL (NumPy, CPU) | success | ~0.98 |
| **M3 RL (PyTorch, RTX 4050)** | **success / collision** | **0.96 / 0.04** |
| M4 mission | flown vs planned length | 31.17 m vs 31.15 m |

(Full tables: `docs/BENCHMARKS.md`. Plots/GIF: `results/`.)

---

## 9. How to run everything (offline, in your venv)

```bat
cd "C:\Users\Samyak Jain\Claude\Projects\VLA drone project"
.venv\Scripts\activate
pytest -q                                    :: 12 tests
python scripts\phase1_hover.py               :: control
python scripts\phase2_planning_benchmark.py  :: A* vs RRT*
python scripts\phase3_benchmark.py           :: PPO vs A*
python scripts\phase3_eval_torch.py          :: score the GPU policy (96%)
python scripts\phase4_mission.py             :: mission + GIF
python scripts\run_all_benchmarks.py         :: dashboard + report
```
GPU training: `python scripts\phase3_train_torch.py --updates 300`
Real 3D sim (PX4+Gazebo): see `realsim/00_RUNBOOK.md`.

---


