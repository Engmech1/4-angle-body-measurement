@echo off
setlocal
cd /d "%~dp0"
title Exercise App - ML Continual Learning Trainer

:: Find Python Executable
set "PY_EXE=C:\Users\thana\.pyenv\pyenv-win\versions\3.13.2\python.exe"

if not exist "%PY_EXE%" (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_EXE=py -3"
    ) else (
        where python >nul 2>&1
        if %errorlevel% equ 0 (
            set "PY_EXE=python"
        )
    )
)

echo ==============================================================================
echo   TRAINING CONTINUAL LEARNING ML MODEL...
echo ==============================================================================
"%PY_EXE%" train_ml_model.py
echo.
pause
