@echo off
chcp 65001 > nul
cls
title Exercise App - 4-Angle Guided Capture Body Measurement System

:MENU
cls
echo ==============================================================================
echo       EXERCISE APP: 4-ANGLE GUIDED CAPTURE BODY MEASUREMENT SYSTEM
echo                  High-Precision Computer Vision Engine
echo ==============================================================================
echo.
echo  กรุณาเลือกเมนูที่ต้องการรัน (Select an option):
echo.
echo   [1] Run Interactive Guided Capture Demo (demo_capture.py)
echo       - จำลองขั้นตอนการวัดสรีระ 4 มุม (0, 90, 180, 270 องศา)
echo.
echo   [2] Run Adversarial QA Benchmark (run_simulation.py)
echo       - ทดสอบความแม่นยำ Error ^< 0.5 cm บน 6 สรีระมนุษย์ พร้อม Sway และ Noise
echo.
echo   [3] Generate Visual Diagnostic Report & Plots (visualize_pipeline.py)
echo       - สร้างรูปภาพกราฟวิเคราะห์ 4 มิติ (Cross-Section, DoG Profile, Sway, Error)
echo.
echo   [4] Run Pytest Unit & Integration Tests (pytest -v)
echo       - ตรวจสอบความถูกต้องของทุกโมดูล (Sub-Pixel, Scaling, MAD, Spline)
echo.
echo   [5] View README.md Documentation
echo.
echo   [0] Exit (ออกจากโปรแกรม)
echo.
echo ==============================================================================
set /p choice="ป้อนหมายเลขเมนู (0-5): "

if "%choice%"=="1" goto DEMO
if "%choice%"=="2" goto BENCHMARK
if "%choice%"=="3" goto VISUAL
if "%choice%"=="4" goto TESTS
if "%choice%"=="5" goto README
if "%choice%"=="0" goto EXIT

echo.
echo [ERROR] หมายเลขเมนูไม่ถูกต้อง กรุณาลองใหม่อีกครั้ง...
timeout /t 2 > nul
goto MENU

:DEMO
cls
echo ==============================================================================
echo   RUNNING INTERACTIVE GUIDED CAPTURE DEMO...
echo ==============================================================================
python demo_capture.py
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:BENCHMARK
cls
echo ==============================================================================
echo   RUNNING ADVERSARIAL QA BENCHMARK (6 SOMATOTYPES)...
echo ==============================================================================
python run_simulation.py
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:VISUAL
cls
echo ==============================================================================
echo   GENERATING VISUAL DIAGNOSTIC REPORT...
echo ==============================================================================
python visualize_pipeline.py
if exist body_measurement_visual_report.png (
    echo.
    echo [INFO] เปิดรูปภาพกราฟวิเคราะห์ (body_measurement_visual_report.png)...
    start body_measurement_visual_report.png
)
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
goto MENU

:TESTS
cls
echo ==============================================================================
echo   RUNNING PYTEST TEST SUITE...
echo ==============================================================================
pytest -v
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
