@echo off
chcp 65001 > nul
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
pause
