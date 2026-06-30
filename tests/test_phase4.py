"""Phase 4 tests: mission planning + closed-loop execution incl. return-to-home."""
import numpy as np

from vantage.sim.world import World
from vantage.missions.mission import Mission, plan_leg


def _world():
    obs = [(3, 3, 2.0, 0.7), (5, 6, 2.5, 0.8), (7, 3.5, 2.2, 0.6)]
    return World(bounds=((0, 10), (0, 10), (0, 4)), obstacles=obs)


def test_plan_leg_collision_free():
    w = _world()
    path = plan_leg(w, (0.5, 0.5, 1.5), (9.0, 9.0, 2.5))
    assert path is not None
    for a, b in zip(path[:-1], path[1:]):
        assert w.segment_free(a, b, 0.2)


def test_closed_loop_mission_returns_home():
    # a mission that returns to its start must still complete (regression for the
    # follower bug where return-to-home terminated instantly / looped forever)
    w = _world()
    m = Mission("loop", [(0.5, 0.5, 1.5), (9.0, 2.0, 2.5),
                         (8.5, 9.0, 2.0), (0.5, 0.5, 1.5)])
    report, traj = m.run(w)
    assert report["planned_ok"]
    assert report["success"], report
    assert not report["collided"]
    # flown length should track the planned length closely (no wild looping)
    assert report["flown_length_m"] < 1.5 * report["plan_length_m"]
    assert np.linalg.norm(traj[-1] - m.waypoints[-1]) < 0.5
