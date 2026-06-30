#!/usr/bin/env python3
"""Fly a VANTAGE-style waypoint mission in PX4 SITL + Gazebo via MAVSDK.

Run this in a SECOND terminal while `make px4_sitl gz_x500` is running. You'll
see the x500 quadrotor take off and fly the waypoint pattern in the Gazebo 3D
window -- i.e. the real simulator, commanded by Python.

    python3 realsim/fly_mission.py

The waypoints mirror the Phase-4 'search / inspect / return' mission. To fly a
path produced by the VANTAGE A* planner, replace WAYPOINTS with planner output
(x, y, z) -> here expressed in local NED (north, east, down; down is negative up).
"""
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw

# (north, east, down, yaw_deg) -- down is negative for altitude above ground
WAYPOINTS = [
    (0.0,  0.0, -2.5,   0.0),   # take off / hold
    (8.0,  0.0, -2.5,   0.0),   # inspect A
    (8.0,  8.0, -2.0,  90.0),   # inspect B
    (0.0,  8.0, -3.0, 180.0),   # inspect C
    (0.0,  0.0, -2.5,   0.0),   # return home
]
HOLD_S = 5.0          # seconds to settle at each waypoint
REACH_TOL = 0.6       # metres


async def run():
    drone = System()
    print("connecting to PX4 SITL (udp://:14540)...")
    await drone.connect(system_address="udp://:14540")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("connected.")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok or health.is_local_position_ok:
            print("position estimate OK.")
            break

    print("arming..."); await drone.action.arm()

    # set an initial setpoint, then start offboard
    await drone.offboard.set_position_ned(PositionNedYaw(0, 0, -2.5, 0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print("offboard start failed:", e._result.result); await drone.action.disarm(); return

    for i, (n, e, d, yaw) in enumerate(WAYPOINTS):
        print(f"-> waypoint {i}: N={n} E={e} D={d} yaw={yaw}")
        await drone.offboard.set_position_ned(PositionNedYaw(n, e, d, yaw))
        # wait until close or timeout
        for _ in range(int(HOLD_S * 5)):
            await asyncio.sleep(0.2)

    print("returning / landing..."); await drone.action.land()
    await asyncio.sleep(8)
    print("mission complete.")


if __name__ == "__main__":
    asyncio.run(run())
