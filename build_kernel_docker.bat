@echo off
REM SNN Kernel Build with Docker
REM This script builds the Dockerfile and runs the kernel build

setlocal enabledelayedexpansion

set IMAGE_NAME=snn-cuda-kernel-builder
set CONTAINER_NAME=snn-kernel-build
set REPO_PATH=%cd%

echo ============================================
echo SNN CUDA Kernel - Docker Build Environment
echo ============================================
echo.
echo Step 1: Building Docker image...
echo Image: %IMAGE_NAME%
echo.

docker build -t %IMAGE_NAME%:latest .

if errorlevel 1 (
    echo ERROR: Docker build failed
    exit /b 1
)

echo.
echo Step 2: Docker image built successfully
echo.
echo Step 3: Running kernel build inside container...
echo.
echo Container command:
echo   cd /workspace
echo   python src/learning/setup.py build_ext --inplace
echo.

REM Run the container with mounted volume
docker run --rm -it ^
    -v "%REPO_PATH%":/workspace ^
    --gpus all ^
    --name %CONTAINER_NAME% ^
    %IMAGE_NAME%:latest ^
    bash -c "cd /workspace && python src/learning/setup.py build_ext --inplace && echo. && echo Build complete! && bash"

echo.
echo ============================================
echo Kernel build finished
echo ============================================
echo.
echo Next steps:
echo 1. Enable kernel in configuration/SNN_module.yaml (set kernel: ON)
echo 2. Run: python src/learning/main.py
echo.

endlocal
