@echo off
chcp 65001 > nul
cls
echo ==============================================================================
echo   RUNNING ADVERSARIAL QA BENCHMARK (6 SOMATOTYPES)...
echo ==============================================================================
python run_simulation.py
echo.
pause
