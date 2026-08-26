@echo off
REM Build the WMMA comparison binary for the model GEMM, so the two tensor-core
REM paths can be A/B'd on the same clock in the same session:
REM
REM   scripts\build_ab.bat  &&  bench\test_gemm_wmma.exe --bench
REM
REM GEMM_USE_WMMA picks kernel 9's path in src/gemm.cu instead of kernel 10's.
setlocal
cd /d "%~dp0.."
if not defined VSCMD_ARG_HOST_ARCH (
  call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
set "FLAGS=-arch=sm_89 -DBMB_TF32=1 -O3 -lineinfo -std=c++17 -allow-unsupported-compiler -Xcompiler /wd4819 -diag-suppress 177"
"%NVCC%" %FLAGS% -DGEMM_USE_WMMA tools\test_gemm.cu src\gemm.cu src\bgemm.cu -o bench\test_gemm_wmma.exe -lcublas || exit /b 1
"%NVCC%" %FLAGS% -DGEMM_USE_WMMA src\train_gpt.cu src\gpt.cu src\gemm.cu src\bgemm.cu src\nn.cu src\attention.cu src\flash.cu src\ddp.cu -o bench\train_gpt_wmma.exe || exit /b 1
echo [build_ab] ok
