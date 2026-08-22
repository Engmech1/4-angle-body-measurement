@echo off
chcp 65001 > nul
cls
title Exercise App - 4-Angle Guided Capture & ML Body Measurement Engine

:MENU
cls
echo ==============================================================================
echo       EXERCISE APP: 4-ANGLE GUIDED CAPTURE + ML MEASUREMENT SYSTEM
echo                  High-Precision Computer Vision & ML Engine
echo ==============================================================================
echo.
echo  กรุณาเลือกเมนูที่ต้องการรัน (Select an option):
echo.
echo   [1] Run Live Computer Webcam Capture + ML Engine (live_camera_capture.py)
echo       - เชื่อมต่อกล้องคอมพิวเตอร์แบบ Real-time, ตรวจจับ ArUco และ Skeleton Pose
echo       - มีระบบ Online Active Learning อัปเดต ML Model เมื่อป้อนค่าสายวัดจริง
echo.
echo   [2] Run Interactive Guided Capture Demo (demo_capture.py)
echo       - จำลองขั้นตอนการวัดสรีระ 4 มุม (0, 90, 180, 270 องศา) ในหน่วยความจำ
echo.
echo   [3] Run Adversarial QA Benchmark (run_simulation.py)
echo       - ทดสอบความแม่นยำ Error ^< 0.5 cm บน 6 สรีระมนุษย์ พร้อม Sway และ Noise
echo.
echo   [4] Generate Visual Diagnostic Report & Plots (visualize_pipeline.py)
echo       - สร้างรูปภาพกราฟวิเคราะห์ 4 มิติ (Cross-Section, DoG Profile, Sway, Error)
echo.
echo   [5] Train / Retrain ML Continual Learning Model (train_ml_model.py)
echo       - ฝึกโมเดล ML เรียนรู้ข้อผิดพลาดทางสรีระและชดเชยมิติอัตโนมัติ
echo.
echo   [6] Run Pytest Unit & Integration Tests (pytest -v)
echo       - ตรวจสอบความถูกต้องของทุกโมดูล (Sub-Pixel, Scaling, MAD, Spline, ML)
echo.
echo   [7] View README.md Documentation
echo.
echo   [0] Exit (ออกจากโปรแกรม)
echo.
echo ==============================================================================
set /p choice="ป้อนหมายเลขเมนู (0-7): "

if "%choice%"=="1" goto CAMERA
if "%choice%"=="2" goto DEMO
if "%choice%"=="3" goto BENCHMARK
if "%choice%"=="4" goto VISUAL
if "%choice%"=="5" goto TRAIN_ML
if "%choice%"=="6" goto TESTS
if "%choice%"=="7" goto README
if "%choice%"=="0" goto EXIT

echo.
echo [ERROR] หมายเลขเมนูไม่ถูกต้อง กรุณาลองใหม่อีกครั้ง...
timeout /t 2 > nul
goto MENU

:CAMERA
cls
echo ==============================================================================
echo   STARTING LIVE COMPUTER WEBCAM CAPTURE + ML ENGINE...
echo ==============================================================================
python live_camera_capture.py
echo.
echo กดปุ่มใดๆ เพื่อกลับสู่เมนูหลัก...
pause > nul
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

:TRAIN_ML
cls
echo ==============================================================================
echo   TRAINING CONTINUAL LEARNING ML MODEL...
echo ==============================================================================
python train_ml_model.py
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
