@echo off
REM Profile the fused attention kernels as they actually run inside a training
REM step, not in a microbenchmark.
REM
REM Must run elevated: reading GPU performance counters needs administrator on
REM Windows (ERR_NVGPUCTRPERM otherwise). The alternative is the driver-wide
REM "allow all users" setting, which is worse -- it is a permanent change to
REM the machine for a measurement that takes a minute.
REM
REM ncu replays each kernel several times to collect counters, so the ABSOLUTE
REM times it reports are inflated and meaningless. Only the counters matter.

cd /d "%~dp0.."
if not exist bench\logs mkdir bench\logs

set "NCU="
for /d %%d in ("C:\Program Files\NVIDIA Corporation\Nsight Compute*") do set "NCU=%%d\ncu.bat"
if not defined NCU (
  echo Nsight Compute not found
  exit /b 1
)

REM Arguments pass through to train_gpt, and --tf32 matters: the tensor-core
REM backward kernels (flash_bwd_kv_mma_k, flash_bwd_q_mma_k) run only on that
REM path, and profiling the fp32 ones says nothing about them.
REM   scripts\profile_flash.bat --tf32
set "TAG=fp32"
if /i "%~1"=="--tf32" set "TAG=tf32"
echo [ncu] %NCU% (%TAG%)
"%NCU%" --kernel-name regex:flash_ --launch-count 24 ^
  --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis ^
  --section LaunchStats --section WarpStateStats ^
  bench\train_gpt.exe -n 2 --eval 10000 --sample 10000 --len 0 %* > bench\logs\flash_ncu_%TAG%.txt 2>&1

echo wrote bench\logs\flash_ncu_%TAG%.txt
