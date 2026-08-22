@echo off
setlocal
cd /d "%~dp0"
title Exercise App - Visual Diagnostic Report Generator

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
echo   GENERATING VISUAL DIAGNOSTIC REPORT...
echo ==============================================================================
"%PY_EXE%" visualize_pipeline.py
if exist body_measurement_visual_report.png (
    echo.
    echo [INFO] Opening visual diagnostic plot (body_measurement_visual_report.png)...
    start body_measurement_visual_report.png
)
echo.
pause
