"""6-DOF quadrotor dynamics (rigid body, Euler-angle attitude).

State vector (length 12):
    [x, y, z,  vx, vy, vz,  roll, pitch, yaw,  p, q, r]
where (p,q,r) are body angular rates.

Inputs:
    thrust  : total collective thrust along body +z   [N]
    torque  : body torques (tau_x, tau_y, tau_z)       [N*m]

The model is deliberately lightweight (no aero drag tables, no motor lag by
default) so it runs thousands of steps per second on CPU, yet it is a faithful
nonlinear rigid-body model: gravity, full SO(3) rotation, and gyroscopic
coupling are all present. A first-order motor model can be enabled.
"""
from __future__ import annotations

import numpy as np

GRAVITY = 9.81


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body->world rotation matrix from ZYX (yaw-pitch-roll) Euler angles."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def euler_rate_matrix(roll: float, pitch: float) -> np.ndarray:
    """Maps body rates (p,q,r) -> Euler angle rates (roll,pitch,yaw)."""
    cr, sr = np.cos(roll), np.sin(roll)
    tp = np.tan(pitch)
    cp = np.cos(pitch)
    cp = cp if abs(cp) > 1e-6 else 1e-6  # avoid gimbal singularity blowup
    return np.array([
        [1.0, sr * tp,      cr * tp],
        [0.0, cr,          -sr],
        [0.0, sr / cp,      cr / cp],
    ])


class Quadrotor:
    """Nonlinear quadrotor with optional first-order motor dynamics."""

    def __init__(
        self,
        mass: float = 1.0,
        inertia=(0.0085, 0.0085, 0.0165),
        arm_length: float = 0.17,
        max_thrust: float = 4.0 * 9.81,   # ~4:1 thrust-to-weight at hover mass=1
        motor_tau: float = 0.0,           # 0 = ideal actuators
        dt: float = 0.005,
    ):
        self.m = float(mass)
        self.I = np.diag(inertia).astype(float)
        self.I_inv = np.linalg.inv(self.I)
        self.L = arm_length
        self.max_thrust = max_thrust
        self.motor_tau = motor_tau
        self.dt = dt
        self.state = np.zeros(12)
        self._thrust_cmd = self.m * GRAVITY
        self._thrust_act = self.m * GRAVITY

    # ------------------------------------------------------------------ utils
    @property
    def position(self) -> np.ndarray:
        return self.state[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[3:6].copy()

    @property
    def euler(self) -> np.ndarray:
        return self.state[6:9].copy()

    def reset(self, position=(0, 0, 0), yaw: float = 0.0):
        self.state = np.zeros(12)
        self.state[0:3] = np.asarray(position, dtype=float)
        self.state[8] = yaw
        self._thrust_cmd = self._thrust_act = self.m * GRAVITY
        return self.state.copy()

    # --------------------------------------------------------------- dynamics
    def _deriv(self, s: np.ndarray, thrust: float, torque: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = s[6], s[7], s[8]
        omega = s[9:12]
        R = rotation_matrix(roll, pitch, yaw)

        acc = np.array([0.0, 0.0, -GRAVITY]) + R @ np.array([0.0, 0.0, thrust]) / self.m
        euler_dot = euler_rate_matrix(roll, pitch) @ omega
        omega_dot = self.I_inv @ (torque - np.cross(omega, self.I @ omega))

        ds = np.zeros(12)
        ds[0:3] = s[3:6]
        ds[3:6] = acc
        ds[6:9] = euler_dot
        ds[9:12] = omega_dot
        return ds

    def step(self, thrust: float, torque) -> np.ndarray:
        """Advance one dt with RK4. Returns new state."""
        torque = np.asarray(torque, dtype=float)
        thrust = float(np.clip(thrust, 0.0, self.max_thrust))

        # optional first-order motor lag on collective thrust
        if self.motor_tau > 0:
            alpha = self.dt / (self.motor_tau + self.dt)
            self._thrust_act += alpha * (thrust - self._thrust_act)
            thrust = self._thrust_act

        dt = self.dt
        s = self.state
        k1 = self._deriv(s, thrust, torque)
        k2 = self._deriv(s + 0.5 * dt * k1, thrust, torque)
        k3 = self._deriv(s + 0.5 * dt * k2, thrust, torque)
        k4 = self._deriv(s + dt * k3, thrust, torque)
        self.state = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.state.copy()
