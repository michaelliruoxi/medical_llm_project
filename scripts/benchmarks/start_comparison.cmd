@echo off
setlocal

set "PROJECT_ROOT=%~dp0..\.."
set "PYTHON_EXE=C:\Users\Owner\AppData\Local\Programs\Python\Python311\python.exe"
set "OUTPUT_DIR=%PROJECT_ROOT%\data\outputs\comparison"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "STAMP=%%i"

set "STDOUT_LOG=%OUTPUT_DIR%\run_%STAMP%.log"
set "STDERR_LOG=%OUTPUT_DIR%\run_%STAMP%_stderr.log"

echo Starting model comparison...
echo STDOUT=%STDOUT_LOG%
echo STDERR=%STDERR_LOG%

cd /d "%PROJECT_ROOT%"
"%PYTHON_EXE%" "scripts\benchmarks\run_comparison.py" 1>>"%STDOUT_LOG%" 2>>"%STDERR_LOG%"
