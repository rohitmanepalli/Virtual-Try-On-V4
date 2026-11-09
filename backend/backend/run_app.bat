@echo off
setlocal enabledelayedexpansion

:: Always run from this script's own folder, regardless of where it's launched from
cd /d "%~dp0"

echo ============================================
echo  Virtual Trial Room - Startup
echo ============================================
echo.

:: 1. Find a working Python. Some installs put "python" on PATH, others only
::    register the "py" launcher (C:\Windows\py.exe) - so check both.
set "PY_CMD="
where python >nul 2>nul
if not errorlevel 1 set "PY_CMD=python"

if not defined PY_CMD (
    where py >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py -3"
)

if not defined PY_CMD (
    echo [ERROR] Python was not found on this computer.
    echo.
    echo Please install Python 3.10 or newer from:
    echo     https://www.python.org/downloads/
    echo.
    echo IMPORTANT: During installation, check the box that says
    echo "Add Python to PATH" before clicking Install.
    echo.
    echo After installing, run this file again.
    pause
    exit /b 1
)

:: 2. Create the virtual environment if it doesn't exist yet
if not exist "venv\Scripts\python.exe" (
    echo [SETUP] Creating virtual environment for first-time setup...
    !PY_CMD! -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [OK] Virtual environment found.
)

:: 3. Install/update required packages
echo [SETUP] Checking required packages, this may take a minute the first time...
venv\Scripts\python.exe -m pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo [ERROR] Failed to install required packages.
    echo Check your internet connection and try again.
    pause
    exit /b 1
)
echo [OK] Required packages are ready.

:: 4. Check for the .env file (holds the GEMINI_API_KEY)
if not exist ".env" (
    echo.
    echo [WARNING] No .env file found in this folder.
    echo The app needs a .env file containing your GEMINI_API_KEY to work.
    echo The app will still start, but image generation will fail until this is added.
    echo.
)

:: 5. Launch the application
echo.
echo [STARTUP] Launching Virtual Trial Room...
echo Your browser will open automatically. Close this window to stop the app.
echo.
venv\Scripts\python.exe main.py

echo.
echo [STOPPED] The application has stopped.
pause
