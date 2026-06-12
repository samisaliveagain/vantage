"""Phase 1 tests: dynamics sanity + closed-loop hover stability."""
import numpy as np

from vantage.sim.quadrotor import Quadrotor, rotation_matrix, GRAVITY
from vantage.control.geometric import CascadedController


def test_rotation_matrix_orthonormal():
    R = rotation_matrix(0.3, -0.2, 1.1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_freefall_without_thrust():
    q = Quadrotor(dt=0.01)
    q.reset(position=(0, 0, 10))
    for _ in range(100):  # 1 s
        q.step(thrust=0.0, torque=(0, 0, 0))
    # after 1 s of free fall: drop ~ 0.5 g t^2
    expected_drop = 0.5 * GRAVITY * 1.0 ** 2
    assert np.isclose(10 - q.position[2], expected_drop, atol=0.3)


def test_hover_equilibrium_holds():
    q = Quadrotor(dt=0.005)
    q.reset(position=(0, 0, 2))
    for _ in range(200):  # exact weight thrust, no torque
        q.step(thrust=q.m * GRAVITY, torque=(0, 0, 0))
    assert abs(q.position[2] - 2.0) < 1e-3


def test_closed_loop_converges():
    q = Quadrotor(dt=0.005)
    ctrl = CascadedController()
    q.reset(position=(0, 0, 0))
    sp = np.array([1.0, -1.0, 2.0])
    for _ in range(2000):  # 10 s
        thrust, torque = ctrl.compute(q.state, pos_sp=sp)
        q.step(thrust, torque)
    err = np.linalg.norm(q.position - sp)
    assert err < 0.1, f"did not converge, err={err:.3f}"
