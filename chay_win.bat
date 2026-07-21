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
    where winget >nul 2>nul
    if not errorlevel 1 (
      echo.
      echo Chua co Git - dang tu cai bang winget ^(khong can bam gi, cho chut^)...
      winget install --id Git.Git -e --source winget --silent --accept-package-agreements --accept-source-agreements
      set "PATH=%PATH%;C:\Program Files\Git\cmd;C:\Program Files\Git\bin"
    )
  )

  where git >nul 2>nul
  if errorlevel 1 (
    echo.
    echo [CANH BAO] Chua cai duoc Git tu dong.
    echo Tai thu cong tai: https://git-scm.com/download/win
    echo Cai xong, KHOI DONG LAI MAY, roi chay lai file nay.
    echo Bo qua, se chay o che do cloud binh thuong.
    echo.
    goto :run
  )

  if exist "engine" (
    echo Don dep ban tai do dang truoc do...
    rmdir /s /q engine
  )

  echo.
  echo Dang tai model VoxCPM2 ve may ^(engine\ - chi 1 lan duy nhat, ~vai phut^)...
  git clone --depth 1 https://github.com/OpenBMB/VoxCPM.git engine
  if errorlevel 1 (
    echo.
    echo [LOI] Tai model that bai! Nguyen nhan thuong gap:
    echo   - Mang cong ty / VPN chan Github
    echo   - Chua cai xong Git, can khoi dong lai May Tinh sau khi cai
    echo   - Het dung luong o dia ^(can it nhat 2GB trong^)
    echo Chay lai file nay de thu lai. Tam thoi chay che do cloud ^(van dung duoc, chi cham hon^).
    echo.
    goto :run
  )
  echo Tai model xong!
)

if exist "engine\src\voxcpm" (
  echo.
  echo Dang cai them thu vien chay local ^(torch... - lan dau co the toi 10-15 phut^)...
  set SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
  uv pip install -e engine
  if errorlevel 1 (
    echo.
    echo [CANH BAO] Cai thu vien local that bai - se chay o che do cloud tam thoi.
    echo Chay lai file nay de thu lai buoc nay.
    echo.
  ) else (
    echo Cai xong!
  )
)

:run
echo.
echo Dang khoi dong giao dien...
echo Trinh duyet se tu mo. Neu khong, hay mo: http://localhost:7860
echo.
uv run python app_cloud.py
pause
