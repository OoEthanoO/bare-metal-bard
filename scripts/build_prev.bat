@echo off
REM Build train_gpt from a git worktree of an older commit into this repo's
REM bench/ directory, so two versions can be A/B'd in one session on one clock.
REM
REM   git worktree add %TEMP%\bmb_prev <sha>
REM   scripts\build_prev.bat %TEMP%\bmb_prev  bench\train_gpt_prev.exe
setlocal
cd /d "%~dp0.."
if not defined VSCMD_ARG_HOST_ARCH (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
set "FLAGS=-arch=sm_89 -DBMB_TF32=1 -O3 -lineinfo -std=c++17 -allow-unsupported-compiler -Xcompiler /wd4819 -diag-suppress 177"
set "SRC=%~1"
set "OUT=%~2"
"%NVCC%" %FLAGS% "%SRC%\src\train_gpt.cu" "%SRC%\src\gpt.cu" "%SRC%\src\gemm.cu" ^
  "%SRC%\src\bgemm.cu" "%SRC%\src\nn.cu" "%SRC%\src\attention.cu" ^
  "%SRC%\src\flash.cu" "%SRC%\src\ddp.cu" -o "%OUT%" || exit /b 1
echo [build_prev] %OUT%
