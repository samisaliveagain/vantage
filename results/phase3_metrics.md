# Phase 3 — PPO (reactive, map-free) vs A* (global planner), 40 worlds

_Generated 2026-06-30 10:06 by the eval script — do not edit by hand._

| method | success | collision | mean_len | compute_ms/decision |
| --- | --- | --- | --- | --- |
| PPO | 0.97 | 0.00 | 8.58 | 0.675 |
| ASTAR | 1.00 | 0.00 | 8.18 | 52.563 |

## Notes

A* plans once over a full occupancy map (safe by construction but needs the map). PPO uses only 16 lidar rays + goal bearing, reacts per-step. compute_ms is per planning call (A*) vs per policy forward pass (PPO).
