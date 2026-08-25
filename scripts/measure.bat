@echo off
REM Run a command with the SM clock verified before AND after it.
REM
REM   scripts\measure.bat <command> [args ...]
REM
REM WHY THIS EXISTS. `nvidia-smi -lgc` does not stay applied on this machine.
REM Nsight Compute resets the application clock when it detaches, and the lock
REM has also been observed to lapse on its own across a driver power event. Both
REM times it happened, the result was a number that looked like a large speedup
REM and was a clock ratio: 59.9 -> 38 ms is 1.58x, and so is 1900/1200.
REM
REM A rule that says "pin the clock first" does not survive that, because the
REM pin coming undone is silent and happens BETWEEN the pinning and the
REM measurement. So every timing run goes through here, and the clock is read
REM on both sides of it. If they differ, or either is not the target, the run
REM is labelled UNPINNED and the number is not a result.
setlocal enabledelayedexpansion
cd /d "%~dp0.."
set "WANT=1200"

for /f %%c in ('nvidia-smi --query-gpu^=clocks.sm --format^=csv^,noheader^,nounits') do set "BEFORE=%%c"
if not "%BEFORE%"=="%WANT%" (
  echo [clock] %BEFORE% MHz, want %WANT% -- re-pinning
  call "%~dp0gpu_clocks.bat" lock %WANT%
  timeout /t 3 >nul
  for /f %%c in ('nvidia-smi --query-gpu^=clocks.sm --format^=csv^,noheader^,nounits') do set "BEFORE=%%c"
)

%*
set "RC=%ERRORLEVEL%"

for /f %%c in ('nvidia-smi --query-gpu^=clocks.sm --format^=csv^,noheader^,nounits') do set "AFTER=%%c"
if "%BEFORE%"=="%WANT%" if "%AFTER%"=="%WANT%" (
  echo [clock] %WANT% MHz before and after -- timings above are comparable
  exit /b %RC%
)
echo [clock] UNPINNED: %BEFORE% MHz before, %AFTER% MHz after. The timings
echo [clock] above are NOT a result -- re-pin and run again.
exit /b %RC%
