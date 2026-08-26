@echo off
REM Register / spill / shared-memory report for one .cu: scripts\regs.bat <file>
setlocal
cd /d "%~dp0.."
if not defined VSCMD_ARG_HOST_ARCH (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
"%NVCC%" -arch=sm_89 -DBMB_TF32=1 -O3 -std=c++17 -allow-unsupported-compiler -Xptxas -v -lineinfo -cubin "%~1" -o "%TEMP%\regs.cubin"
