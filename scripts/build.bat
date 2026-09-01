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

REM The toolkit, host compiler and architecture all come from env.bat, which is
REM the one place any of them is decided. To pin an older toolkit on purpose:
REM   scripts\build.bat --cuda 12.5 sgemm
if /i "%~1"=="--cuda" (
  set "CUDA_PIN=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%~2"
  shift
  shift
)
call "%~dp0env.bat" || exit /b 1

if not exist bench mkdir bench
echo [toolkit] %CUDA_PATH%
echo [arch]    %ARCH% (BMB_TF32=%TF32%)

set "KERNELS="
for %%f in (src\kernels\*.cu) do set "KERNELS=!KERNELS! %%f"

set "GPT_SRC=src\gpt.cu src\gemm.cu src\bgemm.cu src\nn.cu src\attention.cu src\flash.cu src\ddp.cu"

REM Collected by walking the arguments rather than using %*, because %* still
REM contains the --cuda pair that shift already consumed.
set "TARGETS="
:argloop
if "%~1"=="" goto argdone
set "TARGETS=!TARGETS! %~1"
shift
goto argloop
:argdone
if "%TARGETS%"=="" set "TARGETS=sgemm test_gemm train_gpt test_grad test_flash test_ddp bench_nn device_query"

for %%t in (%TARGETS%) do (
  echo [build] %%t
  if "%%t"=="sgemm"        "%NVCC%" %FLAGS% src\bench.cu src\registry.cu !KERNELS! -o bench\sgemm.exe -lcublas
  if "%%t"=="test_gemm"    "%NVCC%" %FLAGS% tools\test_gemm.cu src\gemm.cu src\bgemm.cu -o bench\test_gemm.exe -lcublas
  if "%%t"=="train_gpt"    "%NVCC%" %FLAGS% src\train_gpt.cu %GPT_SRC% -o bench\train_gpt.exe
  if "%%t"=="test_grad"    "%NVCC%" %FLAGS% tools\test_grad.cu %GPT_SRC% -o bench\test_grad.exe
  if "%%t"=="test_flash"   "%NVCC%" %FLAGS% tools\test_flash.cu src\flash.cu src\attention.cu src\bgemm.cu src\gemm.cu -o bench\test_flash.exe
  if "%%t"=="test_ddp"     "%NVCC%" %FLAGS% tools\test_ddp.cu src\ddp.cu -o bench\test_ddp.exe
  if "%%t"=="bench_nn"     "%NVCC%" %FLAGS% tools\bench_nn.cu src\nn.cu -o bench\bench_nn.exe
  if "%%t"=="device_query" "%NVCC%" -arch=%ARCH% -O2 -std=c++17 -allow-unsupported-compiler tools\device_query.cu -o bench\device_query.exe
  if errorlevel 1 exit /b 1
)
echo [build] ok
