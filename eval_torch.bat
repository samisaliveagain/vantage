@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -u scripts\phase3_eval_torch.py > results\phase3_torch_eval_run.txt 2>&1
