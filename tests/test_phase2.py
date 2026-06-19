"""Phase 2 tests: world collision model + both planners on a known map."""
import numpy as np

from vantage.sim.world import World
from vantage.planning.astar import astar_grid, shortcut_path
from vantage.planning.rrt_star import RRTStar


def _wall_world():
    # vertical wall of spheres across the middle with a gap near y=8
    obs = [(5.0, y, 2.5, 0.6) for y in np.arange(0.5, 7.0, 0.7)]
    return World(bounds=((0, 10), (0, 10), (1, 4)), obstacles=obs)


def test_collision_model():
    w = _wall_world()
    assert not w.is_free((5.0, 0.5, 2.5))      # inside an obstacle
    assert w.is_free((0.5, 9.0, 2.0))          # open space
    assert not w.segment_free((0.5, 0.5, 2.5), (9.5, 0.5, 2.5))  # crosses wall


def test_astar_finds_path_around_wall():
    w = _wall_world()
    grid, res, origin = w.occupancy_grid(resolution=0.25, robot_radius=0.2)
    s = w.world_to_grid((0.5, 0.5, 2.5), origin, res)
    g = w.world_to_grid((9.5, 0.5, 2.5), origin, res)
    path = astar_grid(grid, s, g)
    assert path is not None and len(path) > 2
    wp = [w.grid_to_world(i, origin, res) for i in path]
    sc = shortcut_path(wp, w, 0.2)
    for a, b in zip(sc[:-1], sc[1:]):
        assert w.segment_free(a, b, 0.2)       # shortcut stays collision-free


def test_rrt_star_finds_valid_path():
    w = _wall_world()
    path = RRTStar(w, robot_radius=0.2, seed=1, max_iter=4000).plan(
        (0.5, 0.5, 2.5), (9.5, 0.5, 2.5))
    assert path is not None
    for a, b in zip(path[:-1], path[1:]):
        assert w.segment_free(a, b, 0.2)
