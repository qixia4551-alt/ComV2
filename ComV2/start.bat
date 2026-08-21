@echo off
chcp 65001 >nul 2>&1
title ComfyUI Console Server

REM ============================================================
REM  ComfyUI Console - Start Script
REM  This script starts the Python backend server.
REM ============================================================

echo.
echo ============================================================
echo    ComfyUI Console Server
echo ============================================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if Python is available
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in PATH.
    echo.
    echo Please install Python 3.6 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2 delims= " %%a in ('python --version 2^>^&1') do set PYVER=%%a
echo [INFO] Python version: %PYVER%
echo.

REM Check if server.py exists
if not exist "server.py" (
    echo [ERROR] server.py not found in current directory.
    echo Current directory: %cd%
    echo.
    pause
    exit /b 1
)

REM Check if index.html exists
if not exist "index.html" (
    echo [WARNING] index.html not found. Frontend may not work properly.
    echo.
)

echo [INFO] Starting ComfyUI Console server...
echo [INFO] Server will run on port 8501
echo [INFO] Open your browser and visit: http://127.0.0.1:8501
echo.
echo ============================================================
echo   Press Ctrl+C to stop the server
echo ============================================================
echo.

REM Start the server
python server.py

REM If server exits unexpectedly, show the error
echo.
echo ============================================================
echo   Server has stopped.
echo ============================================================
echo.
echo If the server crashed, scroll up to see the error message.
echo.
pause
