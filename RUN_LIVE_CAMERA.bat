@echo off
chcp 65001 > nul
cls
echo ==============================================================================
echo   STARTING LIVE COMPUTER WEBCAM CAPTURE + ML ENGINE...
echo ==============================================================================
python live_camera_capture.py
echo.
pause
