@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion
title Exercise App - 4-Angle Guided Capture & ML Body Measurement Engine

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

:MENU
cls
echo ==============================================================================
echo       EXERCISE APP: 4-ANGLE GUIDED CAPTURE + ML MEASUREMENT SYSTEM
echo                  High-Precision Computer Vision & ML Engine
echo ==============================================================================
echo  Python Path: %PYTHON_EXE%
echo ==============================================================================
echo.
echo  กรุณาเลือกเมนูที่ต้องการรัน (Select an option):
echo.
echo   [1] Run Multi-View Calibration Dashboard (calibration_dashboard.py)
echo       - แสดงหน้าจอวิเคราะห์ 4 ช่องแบบ Real-time (ArUco, Skeleton, Oscilloscope, Cross-Section)
echo       - รองรับทั้งกล้องเว็บแคมคอมพิวเตอร์ และกล้องมือถือผ่าน IP Camera / DroidCam
echo.
echo   [2] Run Live Computer Webcam Capture + ML Engine (live_camera_capture.py)
echo       - ถ่ายวัดสรีระ 4 มุมอัตโนมัติ พร้อมระบบ Online ML Active Learning
echo.
echo   [3] Run Interactive Guided Capture Demo (demo_capture.py)
echo       - จำลองขั้นตอนการวัดสรีระ 4 มุม (0, 90, 180, 270 องศา) ในหน่วยความจำ
echo.
echo   [4] Run Adversarial QA Benchmark (run_simulation.py)
echo       - ทดสอบความแม่นยำ Error ^< 0.5 cm บน 6 สรีระมนุษย์ พร้อม Sway และ Noise
echo.
echo   [5] Generate Visual Diagnostic Report & Plots (visualize_pipeline.py)
echo       - สร้างรูปภาพกราฟวิเคราะห์ 4 มิติ (Cross-Section, DoG Profile, Sway, Error)
echo.
echo   [6] Train / Retrain ML Continual Learning Model (train_ml_model.py)
echo       - ฝึกโมเดล ML เรียนรู้ข้อผิดพลาดทางสรีระและชดเชยมิติอัตโนมัติ
echo.
echo   [7] Run Pytest Unit & Integration Tests (pytest -v)
echo       - ตรวจสอบความถูกต้องของทุกโมดูล (Sub-Pixel, Scaling, MAD, Spline, ML, SMPL)
echo.
echo   [8] View README.md Documentation
echo.
echo   [0] Exit (ออกจากโปรแกรม)
echo.
echo ==============================================================================
set /p choice="ป้อนหมายเลขเมนู (0-8): "

if "%choice%"=="1" goto DASHBOARD
if "%choice%"=="2" goto CAMERA
if "%choice%"=="3" goto DEMO
if "%choice%"=="4" goto BENCHMARK
if "%choice%"=="5" goto VISUAL
if "%choice%"=="6" goto TRAIN_ML
if "%choice%"=="7" goto TESTS
if "%choice%"=="8" goto README
if "%choice%"=="0" goto EXIT

echo.
echo [ERROR] หมายเลขเมนูไม่ถูกต้อง กรุณาลองใหม่อีกครั้ง...
timeout /t 2 > nul
goto MENU

:DASHBOARD
cls
echo ==============================================================================
echo   STARTING MULTI-VIEW CALIBRATION DASHBOARD & WORKBENCH...
echo ==============================================================================
%PYTHON_EXE% calibration_dashboard.py
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:CAMERA
cls
echo ==============================================================================
echo   STARTING LIVE COMPUTER WEBCAM CAPTURE + ML ENGINE...
echo ==============================================================================
%PYTHON_EXE% live_camera_capture.py
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:DEMO
cls
echo ==============================================================================
echo   RUNNING INTERACTIVE GUIDED CAPTURE DEMO...
echo ==============================================================================
%PYTHON_EXE% demo_capture.py
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:BENCHMARK
cls
echo ==============================================================================
echo   RUNNING ADVERSARIAL QA BENCHMARK (6 SOMATOTYPES)...
echo ==============================================================================
%PYTHON_EXE% run_simulation.py
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:VISUAL
cls
echo ==============================================================================
echo   GENERATING VISUAL DIAGNOSTIC REPORT...
echo ==============================================================================
%PYTHON_EXE% visualize_pipeline.py
if exist body_measurement_visual_report.png (
    echo.
    echo [INFO] เปิดรูปภาพกราฟวิเคราะห์ (body_measurement_visual_report.png)...
    start body_measurement_visual_report.png
)
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:TRAIN_ML
cls
echo ==============================================================================
echo   TRAINING CONTINUAL LEARNING ML MODEL...
echo ==============================================================================
%PYTHON_EXE% train_ml_model.py
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:TESTS
cls
echo ==============================================================================
echo   RUNNING PYTEST TEST SUITE...
echo ==============================================================================
%PYTHON_EXE% -m pytest -v
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:README
cls
type README.md | more
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:EXIT
cls
echo ขอบคุณที่ใช้งาน Exercise App Body Measurement Engine!
timeout /t 2 > nul
exit
