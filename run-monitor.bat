@echo off
setlocal

REM === EDIT THESE THREE LINES TO MATCH YOUR SYSTEM ===
set "PYTHON_EXE=C:\Users\abl96\AppData\Local\Microsoft\WindowsApps\python.exe"
set "SCRIPT_DIR=C:\Users\abl96\Downloads\project cft"
set "SCRIPT_NAME=monitor-sheets.py"
REM ===================================================

cd /d "%SCRIPT_DIR%"

echo Running %SCRIPT_NAME% with %PYTHON_EXE%
echo.

"%PYTHON_EXE%" "%SCRIPT_NAME%"
set "ERR=%ERRORLEVEL%"

echo.
echo Script exited with errorlevel %ERR%.
echo (If there was a traceback or error above, that's what Task Scheduler sees too.)
echo.
pause
endlocal