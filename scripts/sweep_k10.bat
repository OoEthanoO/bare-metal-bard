@echo off
REM Sweep kernel 10's tile configs. Each one is a separate compile, because the
REM tile shape decides the register allocation and the launch bounds, so it
REM cannot be a runtime switch without changing what is being measured.
REM
REM   scripts\sweep_k10.bat [size] [cfg ...]     (default 4096, all configs)
REM
REM Repeat a config to replicate it -- two of these land within 1% of each
REM other and a single run cannot separate them.
setlocal enabledelayedexpansion
cd /d "%~dp0.."
if not defined VSCMD_ARG_HOST_ARCH (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
set "FLAGS=-arch=sm_89 -DBMB_TF32=1 -O3 -lineinfo -std=c++17 -allow-unsupported-compiler -Xcompiler /wd4819 -diag-suppress 177"
set "SIZE=%~1"
if "%SIZE%"=="" set "SIZE=4096"
shift
set "CFGS=%1 %2 %3 %4 %5 %6 %7 %8"
if "%CFGS%"==" " set "CFGS=0 1 2 3 4 5 6 7"

set "KERNELS="
for %%f in (src\kernels\*.cu) do set "KERNELS=!KERNELS! %%f"

for %%c in (%CFGS%) do (
  "%NVCC%" %FLAGS% -DK10_CFG=%%c src\bench.cu src\registry.cu !KERNELS! ^
      -o bench\sgemm_k10.exe -lcublas >nul 2>&1
  if errorlevel 1 (
    echo cfg %%c: build failed
  ) else (
    for /f "tokens=*" %%l in ('bench\sgemm_k10.exe -k 10 -s %SIZE% -i 20 -w 5 ^| findstr /C:"mma"') do echo cfg %%c  %%l
  )
)
