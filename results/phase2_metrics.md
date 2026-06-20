# Phase 2 — A* vs RRT* (15 random worlds, 14 obstacles each)

_First benchmark run. Something is wrong: flown paths are ~5x longer than the
straight-line distance (~13 m) and A* collides 20% of the time even though the
planned paths are collision-free. Suspect the follower is overshooting._

| planner | success | found | collision | len(m) | clear(m) | plan(ms) |
| --- | --- | --- | --- | --- | --- | --- |
| ASTAR | 0.67 | 1.00 | 0.20 | 71.99 | 0.35 | 69.5 |
| RRT | 0.60 | 1.00 | 0.00 | 67.56 | 0.40 | 50.2 |

## Notes

Start (0.5,0.5,1.5) -> goal (9.5,9.5,2.5). TODO: fix follower overshoot.
