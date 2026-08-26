@echo off
REM Run a command with the SM clock watched for the whole time it runs.
REM
REM   scripts\measure.bat <command> [args ...]
REM
REM The work is in measure.ps1, which explains why this checks the clock during
REM the run rather than on either side of it. Batch has no sub-second sleep and
REM no way to poll a child process, so this is a two-line shim.
setlocal
cd /d "%~dp0.."
call "%~dp0env.bat" >nul || exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0measure.ps1" %*
exit /b %ERRORLEVEL%
