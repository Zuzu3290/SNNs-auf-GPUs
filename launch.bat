@echo off
setlocal

echo ==========================================
echo   SNN Training Launcher
echo   Framework is set via:
echo   configuration\SNN_module.yaml
echo ==========================================

python "%~dp0src\learning\main.py"
