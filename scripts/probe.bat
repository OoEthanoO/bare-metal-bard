@echo off
REM Build and run a one-off probe: scripts\probe.bat <file.cu>
setlocal
cd /d "%~dp0.."
if not defined VSCMD_ARG_HOST_ARCH (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
if not exist bench mkdir bench
"%NVCC%" -arch=sm_89 -DBMB_TF32=1 -O2 -std=c++17 -allow-unsupported-compiler "%~1" -o bench\probe.exe || exit /b 1
bench\probe.exe %2 %3 %4 %5
