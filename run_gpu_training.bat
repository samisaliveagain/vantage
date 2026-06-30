@echo off
REM VANTAGE - GPU training launcher (RTX 4050). Double-click to run.
REM Sets up an isolated venv, installs CUDA PyTorch, trains the avoidance policy.
cd /d "%~dp0"
set PY="C:\Program Files\Python311\python.exe"
set LOG=results\gpu_train_log.txt
if not exist results mkdir results

echo [%TIME%] [setup] creating virtual env (.venv)...> "%LOG%" 2>&1
if not exist .venv (%PY% -m venv .venv>> "%LOG%" 2>&1)
call .venv\Scripts\activate.bat

echo [%TIME%] [setup] upgrading pip...>> "%LOG%" 2>&1
python -m pip install --upgrade pip>> "%LOG%" 2>&1

echo [%TIME%] [setup] installing vantage package + deps...>> "%LOG%" 2>&1
python -m pip install -e .>> "%LOG%" 2>&1

echo [%TIME%] [setup] installing PyTorch (CUDA 12.1 build) - large download, please wait...>> "%LOG%" 2>&1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121>> "%LOG%" 2>&1

echo [%TIME%] [check] torch / CUDA visibility:>> "%LOG%" 2>&1
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO-CUDA')">> "%LOG%" 2>&1

echo [%TIME%] [train] starting GPU training (120 updates)...>> "%LOG%" 2>&1
python -u scripts\phase3_train_torch.py --updates 120 --steps 3000>> "%LOG%" 2>&1

echo [%TIME%] [done] training complete. checkpoint: results\phase3_policy_torch.pt>> "%LOG%" 2>&1
echo.
echo ================================================================
echo  Training finished. Full log: results\gpu_train_log.txt
echo ================================================================
pause
