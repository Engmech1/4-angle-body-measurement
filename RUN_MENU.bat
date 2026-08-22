@echo off
setlocal
cd /d "%~dp0"

:: Search for python executable
set "PY_EXE=C:\Users\thana\.pyenv\pyenv-win\versions\3.13.2\python.exe"

if not exist "%PY_EXE%" (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_EXE=py -3"
    ) else (
        set "PY_EXE=python"
    )
)

"%PY_EXE%" run_menu.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
    pause
)
