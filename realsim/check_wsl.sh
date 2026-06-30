#!/usr/bin/env bash
# VANTAGE real-sim — environment readiness check. Run inside WSL Ubuntu 22.04.
echo "=== OS ==="; lsb_release -a 2>/dev/null
echo "=== CPU cores ==="; nproc
echo "=== RAM ==="; free -h | awk '/Mem/{print $2" total, "$7" available"}'
echo "=== disk (need ~15 GB free) ==="; df -h ~ | tail -1
echo "=== GPU visible in WSL? ==="; nvidia-smi -L 2>/dev/null || echo "nvidia-smi not in WSL (Gazebo will use llvmpipe/software GL — still works, just slower)"
echo "=== GL renderer (WSLg) ==="; (glxinfo 2>/dev/null | grep -i "OpenGL renderer") || echo "glxinfo not installed (sudo apt install mesa-utils)"
echo "=== python ==="; python3 --version
echo "If RAM>=8GB, disk>=15GB free, and a window can open (WSLg), you're good for PX4+Gazebo."
