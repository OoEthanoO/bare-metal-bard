@echo off
REM Everything every other script here needs to invoke nvcc, worked out once.
REM
REM   call "%~dp0env.bat" || exit /b 1
REM
REM Exports: NVCC, CUDA_PATH, ARCH (sm_NN), TF32 (0/1), FLAGS.
REM
REM WHY THIS EXISTS. Every build script in this directory used to carry its own
REM copy of three facts: the path to vcvars64, the path to nvcc, and
REM `-arch=sm_89 -DBMB_TF32=1`. That was fine while the repo only ever built on
REM the one laptop it was written on. Moving to a different machine -- a
REM 5070 Ti Laptop, sm_120, with a different toolkit version in the path --
REM broke eight scripts in the same three ways at once, and the failure mode of
REM the arch one is the bad kind: `-arch=sm_89` on an sm_120 card still
REM compiles, and then the driver either refuses to load the cubin or silently
REM JITs it, which is a measurement of the JIT and not of the kernel.
REM
REM So the arch is DETECTED, like the Makefile has always done, and there is
REM exactly one place to fix when the next machine differs again.

setlocal enabledelayedexpansion

REM --- host compiler ------------------------------------------------------
REM nvcc needs cl.exe and the Windows SDK on PATH. vswhere knows where the
REM installer put them; the fixed BuildTools path is the fallback for when it
REM does not (older installs shipped no vswhere).
REM
REM Only LOCATED here. Calling it inside this setlocal would put the PATH it
REM sets inside the scope that `endlocal` below throws away, and the symptom of
REM that is nvcc reporting "Cannot find compiler 'cl.exe' in PATH" from a shell
REM where vcvars had, in every visible sense, just run.
set "VSWHERE=!ProgramFiles(x86)!\Microsoft Visual Studio\Installer\vswhere.exe"
set "VCVARS="
if exist "!VSWHERE!" (
  for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
  )
)
if not defined VCVARS (
  set "VCVARS=!ProgramFiles(x86)!\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
if not exist "!VCVARS!" (
  echo [env] no MSVC found -- install the "Desktop development with C++" workload
  exit /b 1
)
REM --- toolkit ------------------------------------------------------------
REM Toolkits install side by side and this builds with the NEWEST one present.
REM That matters because nvcc's version determines the generated SASS and the
REM cuBLAS beside it is the denominator of every "% of cuBLAS" in the README,
REM so which toolkit built a binary is part of the measurement, not a detail.
REM
REM The ambient CUDA_PATH is deliberately IGNORED. It records whichever
REM installer ran last, not the newest toolkit present -- installing 13.3
REM alongside 12.5 left it pointing at 12.5 -- and trusting it is what caused
REM this repo to benchmark on the old toolkit for a whole session without
REM anyone noticing. Set CUDA_PIN to a full toolkit path to override.
REM
REM "Complete" is checked rather than assumed, because a toolkit can be
REM installed and still not compile: CUDA 13 ships crt and nvvm as separate
REM installer components, and a partial install has an nvcc that fails on the
REM first #include. Silently defaulting to that is worse than not finding it.
set "CUDA_ROOT=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA"
if not defined CUDA_PIN (
  set /a BEST_NUM=0
  for /d %%d in ("!CUDA_ROOT!\v*") do (
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
  echo [env] no complete CUDA toolkit found under "!CUDA_ROOT!"
  exit /b 1
)

REM --- architecture -------------------------------------------------------
REM Detected, not assumed. 8.9 is the Ada laptop this project was written on,
REM 12.0 the Blackwell laptop it moved to; an A40 is 8.6, an A100 8.0, a T4 7.5.
set "ARCH="
REM BMB_ARCH=sm_NN bypasses detection. Needed when nvidia-smi refuses to
REM answer -- it did, mid-session, with "insufficient permissions" -- and the
REM detection below then produced the architecture "sm_NVIDIA-SMI".
if defined BMB_ARCH (
  set "ARCH=%BMB_ARCH%"
  set /a ARCH_MAJOR=%BMB_ARCH:~3,-1%
)
if not defined ARCH for /f "tokens=1,2 delims=." %%a in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do (
  if not defined ARCH set "ARCH=sm_%%a%%b"
  if not defined ARCH_MAJOR set /a ARCH_MAJOR=%%a
)
if not defined ARCH (
  echo [env] nvidia-smi did not report a compute capability -- is a GPU present?
  echo [env] ^(set BMB_ARCH=sm_NN to build without asking it^)
  exit /b 1
)
if not "%ARCH:~0,3%"=="sm_" (
  echo [env] arch detection returned "%ARCH%" -- nvidia-smi printed a message, not a number
  echo [env] set BMB_ARCH=sm_NN to build without asking it
  exit /b 1
)
REM TF32 tensor cores are Ampere and newer. A clone still builds on a T4 or a
REM V100; it just builds without the tensor-core half of the ladder.
set "TF32=0"
if !ARCH_MAJOR! GEQ 8 set "TF32=1"

set "FB=-arch=!ARCH! -O3 -lineinfo -std=c++17 -allow-unsupported-compiler"
set "FB=!FB! -Xcompiler /wd4819 -diag-suppress 177"
set "F=!FB! -DBMB_TF32=!TF32!"

endlocal & (
  set "CUDA_PATH=%CUDA_PIN%"
  set "NVCC=%CUDA_PIN%\bin\nvcc.exe"
  set "ARCH=%ARCH%"
  set "TF32=%TF32%"
  set "FLAGS=%F%"
  set "FLAGS_BASE=%FB%"
  set "VCVARS=%VCVARS%"
)

REM CUDA 13 moved the runtime DLLs from bin\ to bin\x64, and a shell opened
REM before the toolkit was installed has neither. Both go on PATH here so a
REM freshly built binary runs from the same shell that built it. The symptom
REM otherwise is exit code 0xC0000135 and not one byte of output.
set "PATH=%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin;%PATH%"

if not defined VSCMD_ARG_HOST_ARCH call "%VCVARS%" >nul 2>&1
exit /b 0
