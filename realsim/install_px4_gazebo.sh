#!/usr/bin/env bash
# VANTAGE real-sim — one-shot installer for PX4 SITL + Gazebo on Ubuntu 22.04 / WSL2.
# Installs the PX4 toolchain (which pulls in Gazebo Harmonic) and MAVSDK-Python.
# Safe to re-run. Expect 20-40 min and ~10 GB download the first time.
set -e

echo ">>> [1/5] base packages"
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv mesa-utils

echo ">>> [2/5] clone PX4-Autopilot (recursive)"
cd ~
if [ ! -d ~/PX4-Autopilot ]; then
  git clone https://github.com/PX4/PX4-Autopilot.git --recursive
else
  (cd ~/PX4-Autopilot && git pull && git submodule update --init --recursive)
fi

echo ">>> [3/5] run PX4's official Ubuntu setup (installs sim deps incl. Gazebo)"
bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh

echo ">>> [4/5] MAVSDK-Python (for our offboard mission script)"
python3 -m pip install --user mavsdk

echo ">>> [5/5] first build of PX4 SITL with Gazebo x500 (this compiles PX4)"
cd ~/PX4-Autopilot
echo "Building... (first build is slow). Close the Gazebo window when it opens to finish."
make px4_sitl gz_x500 || true

echo ">>> DONE. To launch the sim again:   cd ~/PX4-Autopilot && make px4_sitl gz_x500"
