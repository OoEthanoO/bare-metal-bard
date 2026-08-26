@echo off
REM Build the pre-Ampere configuration on Ampere hardware, to check that a
REM clone still compiles and runs where TF32 tensor cores do not exist.
REM
REM The brief this project came from suggests a free Colab or Kaggle T4, which
REM is sm_75. Until BMB_TF32 existed, this repo would not build there at all --
REM a portability bug nobody would have reported, because anyone who hit it
REM would just have closed the tab.
REM
REM   scripts\build_notf32.bat
setlocal enabledelayedexpansion
cd /d "%~dp0.."
call "%~dp0env.bat" || exit /b 1
set "FLAGS=%FLAGS_BASE% -DBMB_TF32=0"
set "GPT_SRC=src\gpt.cu src\gemm.cu src\bgemm.cu src\nn.cu src\attention.cu src\flash.cu src\ddp.cu"
set "KERNELS="
for %%f in (src\kernels\*.cu) do set "KERNELS=!KERNELS! %%f"

echo [notf32] sgemm
"%NVCC%" %FLAGS% src\bench.cu src\registry.cu !KERNELS! -o bench\sgemm_notf32.exe -lcublas || exit /b 1
echo [notf32] test_gemm
"%NVCC%" %FLAGS% tools\test_gemm.cu src\gemm.cu src\bgemm.cu -o bench\test_gemm_notf32.exe -lcublas || exit /b 1
echo [notf32] test_grad
"%NVCC%" %FLAGS% tools\test_grad.cu %GPT_SRC% -o bench\test_grad_notf32.exe || exit /b 1
echo [notf32] train_gpt
"%NVCC%" %FLAGS% src\train_gpt.cu %GPT_SRC% -o bench\train_gpt_notf32.exe || exit /b 1
echo [notf32] ok
