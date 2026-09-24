@echo off
REM Run the test suite and the coverage gate. The work is in test.ps1, which
REM says why the gate exists; the suite is scripts\test_suite.txt.
REM
REM   scripts\test.bat              build the test targets, then run
REM   scripts\test.bat --no-build   run what is already in bench\
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test.ps1" %*
exit /b %ERRORLEVEL%
