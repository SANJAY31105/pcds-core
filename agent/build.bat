@echo off
echo ========================================
echo PCDS Agent Build Script
echo ========================================
echo.

cd /d "%~dp0"

echo [1/4] Installing dependencies...
pip install -r requirements.txt pyinstaller --quiet

echo [2/4] Building Tray Agent...
pyinstaller pcds_tray_agent.spec --noconfirm

echo [3/4] Building Installer...
pyinstaller PCDS_Setup.spec --noconfirm

echo [4/4] Copying to frontend/public...
copy /Y "dist\PCDS_Setup.exe" "..\frontend\public\PCDS_Setup.exe"

echo.
echo ========================================
echo Build Complete!
echo Output: dist\PCDS_Setup.exe
echo ========================================
pause
