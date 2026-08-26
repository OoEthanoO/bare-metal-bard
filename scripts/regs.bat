@echo off
REM Register / spill / shared-memory report for one .cu: scripts\regs.bat <file>
setlocal
cd /d "%~dp0.."
call "%~dp0env.bat" || exit /b 1
"%NVCC%" %FLAGS% -Xptxas -v -cubin "%~1" -o "%TEMP%\regs.cubin"
