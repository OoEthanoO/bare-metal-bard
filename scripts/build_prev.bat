@echo off
REM Build train_gpt from a git worktree of an older commit into this repo's
REM bench/ directory, so two versions can be A/B'd in one session on one clock.
REM
REM   git worktree add %TEMP%\bmb_prev <sha>
REM   scripts\build_prev.bat %TEMP%\bmb_prev  bench\train_gpt_prev.exe
setlocal
cd /d "%~dp0.."
call "%~dp0env.bat" || exit /b 1
set "SRC=%~1"
set "OUT=%~2"
"%NVCC%" %FLAGS% "%SRC%\src\train_gpt.cu" "%SRC%\src\gpt.cu" "%SRC%\src\gemm.cu" ^
  "%SRC%\src\bgemm.cu" "%SRC%\src\nn.cu" "%SRC%\src\attention.cu" ^
  "%SRC%\src\flash.cu" "%SRC%\src\ddp.cu" -o "%OUT%" || exit /b 1
echo [build_prev] %OUT%
