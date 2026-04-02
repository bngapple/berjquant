@echo off
:: build_app_windows.bat — Builds HTF Executor as a Windows .exe
setlocal EnableDelayedExpansion

echo ===============================================
echo   HTF Executor -- Windows Build
echo ===============================================

:: 1. Build React dashboard
echo.
echo [1/3] Building dashboard...
cd dashboard
call npm run build
if errorlevel 1 (
    echo ERROR: Dashboard build failed
    exit /b 1
)
cd ..
echo    Done: dashboard built

:: 2. Install deps and run PyInstaller
echo.
echo [2/3] Bundling with PyInstaller...
if exist venv (
    call venv\Scripts\activate.bat
) else (
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
)

pip install pyinstaller --quiet

:: Clean old artifacts
if exist build rmdir /s /q build
if exist dist\HTFExecutor rmdir /s /q dist\HTFExecutor

pyinstaller HTFExecutor_windows.spec --clean --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller build failed
    exit /b 1
)
echo    Done: dist\HTFExecutor\HTFExecutor.exe

:: 3. Package as zip
echo.
echo [3/3] Packaging as zip...
if exist "dist\HTFExecutor_windows.zip" del "dist\HTFExecutor_windows.zip"

powershell -Command "Compress-Archive -Path 'dist\HTFExecutor\*' -DestinationPath 'dist\HTFExecutor_windows.zip'"
if errorlevel 1 (
    echo WARNING: Zip failed, but .exe folder is still available at dist\HTFExecutor\
) else (
    echo    Done: dist\HTFExecutor_windows.zip
)

echo.
echo ===============================================
echo   Done!
echo   EXE folder: dist\HTFExecutor\
echo   Zip:        dist\HTFExecutor_windows.zip
echo.
echo   Config/logs: %%APPDATA%%\HTFExecutor\
echo ===============================================
