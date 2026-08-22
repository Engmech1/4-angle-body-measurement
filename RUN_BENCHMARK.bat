@echo off
setlocal
cd /d "%~dp0"
title Exercise App - Adversarial QA Benchmark

:: Find Python Executable
set "PY_EXE=C:\Users\thana\.pyenv\pyenv-win\versions\3.13.2\python.exe"

if not exist "%PY_EXE%" (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_EXE=py -3"
    ) else (
        set "PY_EXE=python"
    )
)

echo ==============================================================================
echo   STARTING ADVERSARIAL QA BENCHMARK (6 SOMATOTYPES)...
echo ==============================================================================
"%PY_EXE%" run_simulation.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with code %errorlevel%
)
echo.
pause
