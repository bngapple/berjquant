@echo off
setlocal

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..\..
set EXEC_DIR=%REPO_ROOT%\tradovate-executor
set PYTHON=%EXEC_DIR%\venv\Scripts\python.exe

cd /d "%EXEC_DIR%"

if not exist "%PYTHON%" (
    echo Python virtual environment not found.
    echo.
    echo From %EXEC_DIR% run:
    echo   py -3 -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    exit /b 1
)

if not exist "%EXEC_DIR%\config.json" (
    echo config.json not found.
    echo Copy deploy\windows_local_bridge\config.windows_local_nt_only.example.json to tradovate-executor\config.json and edit it first.
    exit /b 1
)

echo Testing NinjaTrader bridge connectivity...
"%PYTHON%" scripts\test_nt_connection.py
if errorlevel 1 exit /b 1

echo Starting live Python executor in NinjaTrader mode...
"%PYTHON%" app.py --live
