@echo off
REM Where does a training step actually spend its time?
REM
REM Every time this project has GUESSED at a bottleneck it has been wrong --
REM occupancy for the fused backward, bandwidth for the tensor-core kernel, the
REM interconnect for multi-GPU. The profile has been right every time. This is
REM the tool that makes asking cheap.
REM
REM Reading GPU performance counters needs administrator on Windows, so this
REM re-launches itself elevated and Windows shows a UAC prompt. The output goes
REM to a file, so the elevated console closing takes nothing with it.
REM
REM   scripts\profile_step.bat            fp32 path
REM   scripts\profile_step.bat --tf32     tensor-core path
REM
REM ncu replays each kernel to collect counters, so the ABSOLUTE times are
REM inflated. Only the shares between categories are meaningful, which is all
REM tools/step_profile.py reports.
setlocal
cd /d "%~dp0.."
if not exist bench\logs mkdir bench\logs

set "TAG=fp32"
if /i "%~1"=="--tf32" set "TAG=tf32"

net session >nul 2>&1
if errorlevel 1 (
  echo [elevating] a UAC prompt is about to appear
  powershell -NoProfile -Command "Start-Process -Verb RunAs -FilePath '%~f0' -ArgumentList '%~1'"
  echo waiting for bench\logs\step_ncu_%TAG%.csv ...
  exit /b 0
)

set "NCU="
for /d %%d in ("C:\Program Files\NVIDIA Corporation\Nsight Compute*") do set "NCU=%%d\ncu.bat"
if not defined NCU (
  echo Nsight Compute not found
  pause
  exit /b 1
)

echo [ncu] %NCU%  (%TAG%)
REM Two steps is enough: the first warms caches and allocators, the second is
REM representative, and every kernel appears many times either way.
call "%NCU%" --csv --metrics gpu__time_duration.sum --target-processes all ^
  bench\train_gpt.exe -n 4 %~1 --eval 0 --eval-batches 0 --sample 99999 --len 0 ^
  > bench\logs\step_ncu_%TAG%.csv 2>&1

REM ncu resets the application clock when it detaches, which silently unpins a
REM clock locked earlier -- and an unpinned clock made a 1.8%% change measure as
REM 37%% in this repo before the profile itself disagreed and gave it away. We
REM are already elevated here, so put it back.
nvidia-smi -lgc 1200,1200 >nul 2>&1
nvidia-smi --query-gpu=clocks.sm --format=csv,noheader

echo wrote bench\logs\step_ncu_%TAG%.csv
