@echo off
REM Where does a training step actually spend its time?
REM
REM Must run elevated: reading GPU performance counters is administrator-only on
REM Windows. ncu replays each kernel to collect counters, so the ABSOLUTE times
REM are inflated -- only the shares between categories are meaningful, which is
REM all this is used for.
REM
REM   scripts\profile_step.bat            fp32 path
REM   scripts\profile_step.bat --tf32     tensor-core path

cd /d "%~dp0.."
if not exist bench\logs mkdir bench\logs

set "NCU="
for /d %%d in ("C:\Program Files\NVIDIA Corporation\Nsight Compute*") do set "NCU=%%d\ncu.bat"
if not defined NCU (
  echo Nsight Compute not found
  exit /b 1
)

set "TAG=fp32"
if /i "%~1"=="--tf32" set "TAG=tf32"

echo [ncu] %NCU%  (%TAG%)
REM Two steps is enough: the first warms caches and allocators, the second is
REM representative, and every kernel appears many times either way.
"%NCU%" --csv --metrics gpu__time_duration.sum --target-processes all ^
  bench\train_gpt.exe -n 2 %~1 --eval 99999 --sample 99999 --len 0 ^
  > bench\logs\step_ncu_%TAG%.csv 2>&1

echo wrote bench\logs\step_ncu_%TAG%.csv
