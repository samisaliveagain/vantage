#!/usr/bin/env bash
# Launch PX4 SITL + Gazebo with the x500 quadrotor. A 3D window opens (WSLg).
# In the PX4 console (pxh>) you can type:  commander takeoff   /   commander land
cd ~/PX4-Autopilot
make px4_sitl gz_x500
