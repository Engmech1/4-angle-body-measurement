@echo off
setlocal
cd /d "%~dp0"
title Exercise App - Live Camera Capture and ML Engine

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
echo   STARTING LIVE COMPUTER WEBCAM CAPTURE AND ML ENGINE...
echo ==============================================================================
"%PY_EXE%" live_camera_capture.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with code %errorlevel%
    pause
)
