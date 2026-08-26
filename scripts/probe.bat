@echo off
REM Build and run a one-off probe: scripts\probe.bat <file.cu>
setlocal
cd /d "%~dp0.."
call "%~dp0env.bat" || exit /b 1
if not exist bench mkdir bench
"%NVCC%" %FLAGS% "%~1" -o bench\probe.exe || exit /b 1
bench\probe.exe %2 %3 %4 %5
