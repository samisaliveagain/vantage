# VANTAGE — Real Simulator Runbook (PX4 SITL + Gazebo, in WSL)

This gets you a **true 3D simulator window** where you watch the drone fly,
running on your RTX 4050 via WSL2 Ubuntu 22.04. You run the commands; Claude
wrote and will help debug them. (Isaac Sim is heavier and marginal on 6 GB VRAM,
so this uses Gazebo — the fallback named in the project plan.)

## 0. Open Ubuntu (WSL)
Open the **Ubuntu** app from the Start menu (you already have Ubuntu 22.04.5 LTS).
Everything below is typed in that Ubuntu terminal.

## 1. Copy this folder into WSL (once)
```bash
mkdir -p ~/vantage && cp -r "/mnt/c/Users/Samyak Jain/Claude/Projects/VLA drone project/realsim" ~/vantage/
cd ~/vantage/realsim && chmod +x *.sh
```

## 2. Check readiness
```bash
./check_wsl.sh
```
Want: RAM ≥ 8 GB, ~15 GB free disk, and WSLg able to open a window.

## 3. Install PX4 + Gazebo  (20–40 min, ~10 GB, first time only)
```bash
./install_px4_gazebo.sh
```
A Gazebo window with the x500 quadrotor should open at the end. Close it to finish.

## 4. Launch the simulator
```bash
./run_sim.sh        # opens Gazebo 3D with the drone; PX4 console shows  pxh>
```
Quick manual test — in the `pxh>` console type:
```
commander takeoff
commander land
```
You'll see the drone lift off and land in the 3D window.

## 5. Fly the VANTAGE mission from code
Leave step 4 running. Open a **second** Ubuntu terminal:
```bash
cd ~/vantage/realsim && python3 fly_mission.py
```
The drone flies the search/inspect/return waypoints in Gazebo — commanded by Python.

## Troubleshooting
- **No window appears**: update WSL — in Windows PowerShell run `wsl --update`, then `wsl --shutdown` and reopen Ubuntu. WSLg needs a recent build.
- **Gazebo black / slow**: it's using software GL. To use the GPU, install the NVIDIA CUDA-on-WSL driver (you already have driver 566.07 on Windows; usually GPU GL works in WSLg automatically).
- **`make` errors**: run `bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh` again, then `cd ~/PX4-Autopilot && make distclean && make px4_sitl gz_x500`.
- **MAVSDK can't connect**: make sure step 4 is still running; it exposes `udp://:14540`.

## Next: hook in the planners
`fly_mission.py` flies fixed waypoints. To fly a path from the VANTAGE A* planner,
import `vantage.planning` (install the Python package in WSL with `pip install -e ..`)
and convert planner (x,y,z) world points to local NED setpoints.
