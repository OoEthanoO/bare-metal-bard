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

REM Toolkits install side by side and this builds with the NEWEST one present.
REM That matters because nvcc's version determines the generated SASS and the
REM cuBLAS beside it is the denominator of every percentage in the README, so
REM which toolkit built a binary is part of the measurement, not a detail.
REM
REM To pin an older one on purpose:
REM   scripts\build.bat --cuda 12.5 sgemm
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
if /i "%~1"=="--cuda" (
  set "CUDA_PIN=%CUDA_ROOT%\v%~2"
  shift
  shift
)

REM Default to the newest COMPLETE toolkit installed, compared numerically so
REM v10 beats v9 rather than losing a string compare.
REM
REM The ambient CUDA_PATH is deliberately IGNORED. It records whichever
REM installer ran last, not the newest toolkit present -- installing 13.3
REM alongside 12.5 left it pointing at 12.5 -- and trusting it is what caused
REM this repo to benchmark on the old toolkit for a whole session without
REM anyone noticing. Pass --cuda <version> to pin one on purpose.
REM
REM "Complete" is checked rather than assumed, because a toolkit can be
REM installed and still not compile: CUDA 13 ships crt and nvvm as separate
REM installer components, and a partial install has an nvcc that fails on the
REM first #include. Silently defaulting to that is worse than not finding it.
if not defined CUDA_PIN (
  set /a BEST_NUM=0
  for /d %%d in ("%CUDA_ROOT%\v*") do (
    set "VER=%%~nxd"
    set "VER=!VER:~1!"
    for /f "tokens=1,2 delims=." %%a in ("!VER!.0") do (
      set /a NUM=%%a*1000+%%b
      if exist "%%~fd\bin\nvcc.exe" if exist "%%~fd\include\crt\host_config.h" if exist "%%~fd\nvvm" (
        if !NUM! GTR !BEST_NUM! (
          set /a BEST_NUM=!NUM!
          set "CUDA_PIN=%%~fd"
        )
      )
    )
  )
)
if not defined CUDA_PIN (
  echo no complete CUDA toolkit found under "%CUDA_ROOT%"
  exit /b 1
)
set "CUDA_PATH=%CUDA_PIN%"
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not exist "%NVCC%" (
  echo no nvcc at "%NVCC%" -- pass --cuda ^<version^> for an installed toolkit
  exit /b 1
)
set "FLAGS=-arch=sm_89 -O3 -lineinfo -std=c++17 -allow-unsupported-compiler"
set "FLAGS=%FLAGS% -Xcompiler /wd4819 -diag-suppress 177"

if not exist bench mkdir bench
echo [toolkit] %CUDA_PATH%

set "KERNELS="
for %%f in (src\kernels\*.cu) do set "KERNELS=!KERNELS! %%f"

set "GPT_SRC=src\gpt.cu src\gemm.cu src\bgemm.cu src\nn.cu src\attention.cu src\flash.cu"

REM Collected by walking the arguments rather than using %*, because %* still
REM contains the --cuda pair that shift already consumed.
set "TARGETS="
:argloop
if "%~1"=="" goto argdone
set "TARGETS=!TARGETS! %~1"
shift
goto argloop
:argdone
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
