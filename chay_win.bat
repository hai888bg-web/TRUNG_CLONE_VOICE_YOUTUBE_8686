@echo off
REM Double-click de chay tren Windows
cd /d "%~dp0"

echo ========================================
echo   TRUNG_CLONE_VOICE_YOUTUBE_8686
echo ========================================
echo.

where uv >nul 2>nul
if errorlevel 1 (
  echo Dang cai uv - trinh quan ly Python...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo Dang chuan bi moi truong nhe (lan dau hoi lau)...
uv sync --python 3.11

if not exist "engine\src\voxcpm" (
  where git >nul 2>nul
  if errorlevel 1 (
    echo.
    echo [CANH BAO] Chua cai Git - can Git de tai model chay local cho nhanh.
    echo Tai Git tai: https://git-scm.com/download/win  ^(cai xong chay lai file nay^)
    echo Bo qua, se chay o che do cloud binh thuong.
    echo.
    goto :run
  )
  echo.
  echo Dang tai model VoxCPM2 ve may ^(engine\ - chi 1 lan duy nhat, ~vai phut^)...
  git clone --depth 1 https://github.com/OpenBMB/VoxCPM.git engine
)

if exist "engine\src\voxcpm" (
  echo Dang cai them thu vien chay local ^(torch... - lan dau hoi lau^)...
  set SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
  uv pip install -e engine
)

:run
echo.
echo Dang khoi dong giao dien...
echo Trinh duyet se tu mo. Neu khong, hay mo: http://localhost:7860
echo.
uv run python app_cloud.py
pause
