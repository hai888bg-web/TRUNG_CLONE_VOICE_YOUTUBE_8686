@echo off
REM Double-click de chay tren Windows
cd /d "%~dp0"
set SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0

echo ========================================
echo   NHAN BAN GIONG NOI - VoxCPM2
echo ========================================
echo.

where uv >nul 2>nul
if errorlevel 1 (
  echo Dang cai uv - trinh quan ly Python...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo Dang chuan bi moi truong (lan dau hoi lau)...
uv sync --no-editable --python 3.11

echo.
echo Dang khoi dong giao dien...
echo Trinh duyet se tu mo. Neu khong, hay mo: http://localhost:7860
echo.
uv run python app_cloud.py
pause
