@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: Find Python Executable
set "PYTHON_EXE="
if exist "C:\Users\thana\.pyenv\pyenv-win\versions\3.13.2\python.exe" (
    set "PYTHON_EXE=C:\Users\thana\.pyenv\pyenv-win\versions\3.13.2\python.exe"
) else (
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=py -3"
    ) else (
        where python >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=python"
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python not found on system!
    pause
    exit /b 1
)

cls
echo ==============================================================================
echo   RUNNING ADVERSARIAL QA BENCHMARK (6 SOMATOTYPES)...
echo ==============================================================================
%PYTHON_EXE% run_simulation.py
echo.
pause
