"""Cascaded position + attitude controller for the quadrotor.

Outer loop: PD on position/velocity -> desired acceleration -> collective
thrust and desired roll/pitch (small-angle near-hover mapping, valid for the
moderate manoeuvres used here). Inner loop: PD on attitude error -> body
torques. This is the classic structure used on PX4-class autopilots, written
compactly for simulation and benchmarking.
"""
from __future__ import annotations

import numpy as np

from vantage.sim.quadrotor import GRAVITY, rotation_matrix


class CascadedController:
    def __init__(
        self,
        mass: float = 1.0,
        inertia=(0.0085, 0.0085, 0.0165),
        kp_pos=(3.5, 3.5, 5.0),
        kd_pos=(2.5, 2.5, 3.5),
        kp_att=(90.0, 90.0, 40.0),
        kd_att=(10.0, 10.0, 8.0),
        max_tilt: float = np.deg2rad(35.0),
    ):
        self.m = mass
        self.I = np.diag(inertia).astype(float)
        self.kp_pos = np.asarray(kp_pos, float)
        self.kd_pos = np.asarray(kd_pos, float)
        self.kp_att = np.asarray(kp_att, float)
        self.kd_att = np.asarray(kd_att, float)
        self.max_tilt = max_tilt

    def compute(self, state: np.ndarray, pos_sp, vel_sp=(0, 0, 0), yaw_sp: float = 0.0,
                acc_ff=(0, 0, 0)):
        """Return (thrust, torque) for a position+yaw setpoint."""
        p = state[0:3]
        v = state[3:6]
        roll, pitch, yaw = state[6], state[7], state[8]
        omega = state[9:12]

        pos_sp = np.asarray(pos_sp, float)
        vel_sp = np.asarray(vel_sp, float)
        acc_ff = np.asarray(acc_ff, float)

        # ---- outer loop: desired acceleration (world frame)
        e_p = pos_sp - p
        e_v = vel_sp - v
        acc_des = self.kp_pos * e_p + self.kd_pos * e_v + acc_ff
        acc_des[2] += GRAVITY  # gravity compensation

        # collective thrust = projection of desired force onto current body z
        R = rotation_matrix(roll, pitch, yaw)
        body_z = R[:, 2]
        thrust = self.m * float(acc_des @ body_z)
        thrust = max(thrust, 0.0)

        # ---- desired attitude from desired acceleration direction
        a = acc_des
        a_norm = np.linalg.norm(a)
        if a_norm < 1e-6:
            roll_des = pitch_des = 0.0
        else:
            zb_des = a / a_norm
            # desired roll/pitch that align thrust with zb_des at the given yaw
            cy, sy = np.cos(yaw_sp), np.sin(yaw_sp)
            pitch_des = np.arctan2(zb_des[0] * cy + zb_des[1] * sy, zb_des[2])
            roll_des = np.arctan2(zb_des[0] * sy - zb_des[1] * cy,
                                  zb_des[2] / max(np.cos(pitch_des), 1e-3))
            roll_des = np.clip(roll_des, -self.max_tilt, self.max_tilt)
            pitch_des = np.clip(pitch_des, -self.max_tilt, self.max_tilt)

        # ---- inner loop: attitude PD -> torques
        att = np.array([roll, pitch, yaw])
        att_des = np.array([roll_des, pitch_des, yaw_sp])
        e_att = att_des - att
        e_att[2] = (e_att[2] + np.pi) % (2 * np.pi) - np.pi  # wrap yaw error
        torque = self.I @ (self.kp_att * e_att - self.kd_att * omega)
        return thrust, torque
