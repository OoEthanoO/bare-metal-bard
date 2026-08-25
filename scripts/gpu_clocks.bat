@echo off
REM Windows counterpart to gpu_clocks.sh: lock the SM clock so numbers mean
REM something, and (optionally) let a non-admin process read GPU performance
REM counters so `ncu` stops needing an elevated shell every single time.
REM
REM   scripts\gpu_clocks.bat status
REM   scripts\gpu_clocks.bat lock [MHz]     (default 1200; prompts for UAC)
REM   scripts\gpu_clocks.bat unlock
REM   scripts\gpu_clocks.bat counters       (one-time; needs a reboot after)
REM
REM Why 1200: this is a 55 W mobile RTX 4070. Left alone it boosts to 3105 MHz
REM and then falls back as it hits the power cap. Measured cuBLAS SGEMM at
REM N=2048 swung between 9.0 and 12.3 TFLOP/s across runs in one session purely
REM from thermal state -- a 37% swing in the DENOMINATOR of every "% of cuBLAS"
REM claim in this repo. 1200 MHz is the highest round clock the card holds
REM without deviating under a sustained dense-GEMM load.
REM
REM The lock and the registry key both need administrator rights, so this
REM re-launches itself elevated and Windows shows a UAC prompt. That prompt is
REM the one thing here that cannot be automated away.
setlocal
set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=status"
set "CLOCK=%~2"
if "%CLOCK%"=="" set "CLOCK=1200"

if /i "%ACTION%"=="status" (
  nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit,temperature.gpu --format=csv
  echo.
  reg query "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" /v RmProfilingAdminOnly 2>nul
  if errorlevel 1 echo RmProfilingAdminOnly: not set  ^(ncu needs an elevated shell^)
  exit /b 0
)

net session >nul 2>&1
if errorlevel 1 (
  echo [elevating] a UAC prompt is about to appear
  powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0' -ArgumentList '%ACTION%','%CLOCK%'"
  exit /b 0
)

if /i "%ACTION%"=="lock" (
  nvidia-smi -lgc %CLOCK%,%CLOCK%
  echo SM clock locked to %CLOCK% MHz
  nvidia-smi --query-gpu=clocks.sm --format=csv
  pause
  exit /b 0
)
if /i "%ACTION%"=="unlock" (
  nvidia-smi -rgc
  echo SM clock lock released
  pause
  exit /b 0
)
if /i "%ACTION%"=="counters" (
  REM Lets any user read GPU performance counters. Without it Nsight Compute
  REM refuses with ERR_NVGPUCTRPERM and every profile needs its own UAC click.
  reg add "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" ^
      /v RmProfilingAdminOnly /t REG_DWORD /d 0 /f
  echo.
  echo Set. This takes effect after a reboot.
  pause
  exit /b 0
)
echo usage: %~nx0 {status^|lock [MHz]^|unlock^|counters}
exit /b 1
