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
echo   STARTING VISUAL DIAGNOSTIC REPORT GENERATOR...
echo ==============================================================================
%PYTHON_EXE% visualize_pipeline.py
if exist body_measurement_visual_report.png (
    echo.
    echo [INFO] เปิดรูปภาพกราฟวิเคราะห์ (body_measurement_visual_report.png)...
    start body_measurement_visual_report.png
)
echo.
pause
