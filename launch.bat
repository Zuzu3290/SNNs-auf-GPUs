@echo off
setlocal

set MODEL=%1
if "%MODEL%"=="" set MODEL=norse

if not "%MODEL%"=="norse" if not "%MODEL%"=="torch" if not "%MODEL%"=="sj" (
    echo Unknown model: %MODEL%
    echo Valid options: norse ^| torch ^| sj
    exit /b 1
)

echo ==========================================
echo   Model backend  : %MODEL%
echo ==========================================

python "%~dp0src\learning\main.py" --model %MODEL%
