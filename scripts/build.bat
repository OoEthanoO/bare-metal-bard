@echo off
setlocal enabledelayedexpansion
REM Windows build, mirroring the Makefile target for target.
REM
REM Usage:  scripts\build.bat [target ...]      (no args = all)
REM Targets: sgemm test_gemm train_gpt test_grad test_flash device_query
REM
REM Two Windows-specific notes:
REM   * nvcc needs cl.exe and the Windows SDK on PATH, which is what vcvars64
REM     does. Calling it here keeps the build a single command.
REM   * CUDA 12.5 was released against MSVC 19.40 and refuses anything newer.
REM     The Build Tools ship 19.44, and the guard is a version compare in
REM     host_config.h, not a real incompatibility -- hence the override flag.

cd /d "%~dp0.."

if not defined VSCMD_ARG_HOST_ARCH (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

REM Toolkits install side by side. CUDA_PATH points at the newest one the
REM installer saw; override it to pin a specific toolkit, which matters because
REM nvcc version changes the generated SASS and cuBLAS version changes the
REM baseline every percentage in the README is measured against.
REM   set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
if not defined CUDA_PATH set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not exist "%NVCC%" (
  echo no nvcc at "%NVCC%" -- set CUDA_PATH to an installed toolkit
  exit /b 1
)
set "FLAGS=-arch=sm_89 -O3 -lineinfo -std=c++17 -allow-unsupported-compiler"
set "FLAGS=%FLAGS% -Xcompiler /wd4819 -diag-suppress 177"

if not exist bench mkdir bench
echo [toolkit] %CUDA_PATH%

set "KERNELS="
for %%f in (src\kernels\*.cu) do set "KERNELS=!KERNELS! %%f"

set "GPT_SRC=src\gpt.cu src\gemm.cu src\bgemm.cu src\nn.cu src\attention.cu src\flash.cu"

set "TARGETS=%*"
if "%TARGETS%"=="" set "TARGETS=sgemm test_gemm train_gpt test_grad test_flash device_query"

for %%t in (%TARGETS%) do (
  echo [build] %%t
  if "%%t"=="sgemm"        "%NVCC%" %FLAGS% src\bench.cu src\registry.cu !KERNELS! -o bench\sgemm.exe -lcublas
  if "%%t"=="test_gemm"    "%NVCC%" %FLAGS% tools\test_gemm.cu src\gemm.cu src\bgemm.cu -o bench\test_gemm.exe -lcublas
  if "%%t"=="train_gpt"    "%NVCC%" %FLAGS% src\train_gpt.cu %GPT_SRC% -o bench\train_gpt.exe
  if "%%t"=="test_grad"    "%NVCC%" %FLAGS% tools\test_grad.cu %GPT_SRC% -o bench\test_grad.exe
  if "%%t"=="test_flash"   "%NVCC%" %FLAGS% tools\test_flash.cu src\flash.cu src\attention.cu src\bgemm.cu src\gemm.cu -o bench\test_flash.exe
  if "%%t"=="device_query" "%NVCC%" -arch=sm_89 -O2 -std=c++17 -allow-unsupported-compiler tools\device_query.cu -o bench\device_query.exe
  if errorlevel 1 exit /b 1
)
echo [build] ok
