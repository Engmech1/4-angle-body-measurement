@echo off
chcp 65001 > nul
cls
echo ==============================================================================
echo   TRAINING CONTINUAL LEARNING ML MODEL...
echo ==============================================================================
python train_ml_model.py
echo.
pause
