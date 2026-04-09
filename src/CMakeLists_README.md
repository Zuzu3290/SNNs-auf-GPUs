# Building the SNN Compiler and Acceleration Modules

This README guides you through building the C++/CUDA components in the `src/` directory using CMake. These modules provide the core compiler infrastructure, GPU acceleration, and Python bindings for the SNN project.

## Prerequisites

Before building, ensure you have the following installed:

- **CMake** (version 3.18 or later): Download from [cmake.org](https://cmake.org/) or install via package manager (e.g., `sudo apt install cmake` on Ubuntu).
- **CUDA Toolkit** (version 10.1 or later): Required for compiling CUDA kernels. Download from [NVIDIA's site](https://developer.nvidia.com/cuda-toolkit). Ensure `nvcc` is in your PATH.
- **PyTorch** (with C++ frontend): Install via `pip install torch` (includes LibTorch). CMake will locate it automatically.
- **Python** (3.7+): With development headers (e.g., `python3-dev` on Ubuntu).
- **pybind11**: For Python bindings. Install via `pip install pybind11` or clone from [GitHub](https://github.com/pybind/pybind11) and place in `src/pybind11/` (if using as submodule).
- **C++ Compiler**: GCC/Clang with C++17 support, or MSVC on Windows.

Verify installations:
```bash
cmake --version  # Should be 3.18+
nvcc --version   # CUDA compiler
python -c "import torch; print(torch.__version__)"  # PyTorch
```

## Build Steps

1. **Navigate to the src directory**:
   ```bash
   cd src/
   ```

2. **Create a build directory** (keeps source clean):
   ```bash
   mkdir build
   cd build
   ```

3. **Configure with CMake**:
   ```bash
   cmake ..
   ```
   - This detects dependencies and generates build files (e.g., Makefiles).
   - If PyTorch/LibTorch isn't found, specify its path: `cmake .. -DTorch_DIR=/path/to/libtorch/share/cmake/Torch`.
   - For custom CUDA architectures, edit `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` (e.g., for your GPU: check with `nvidia-smi`).

4. **Build the project**:
   ```bash
   make -j$(nproc)  # Use all CPU cores for faster build
   ```
   - On Windows: `cmake --build . --config Release`.
   - This compiles libraries and the PyBind module.

5. **Verify build output**:
   - Check `build/` for generated files:
     - `libsnn_compiler.so` (or `.dll`/`.dylib`): Compiler library.
     - `libsnn_acceleration.so`: Acceleration library.
     - `libsnn_crsc.so`: CRSC runtime library.
     - `snn_backend.so` (or `.pyd`): Python module for integration.

## Using the Built Modules

- **Python Integration**: After building, the `snn_backend` module can be imported in Python scripts (e.g., in `../learning/main.py`). Ensure `build/` is in your Python path or copy the `.so` files to your project directory.
- **Testing**: Run Python tests in `../src/compiler/tests/` or `../src/learning/test_scripts/` to verify integration.
- **Example**: In a Python script:
  ```python
  import snn_backend  # From build/
  # Use functions bound via pybind11
  ```

## Troubleshooting

- **CMake Errors**:
  - "CUDA not found": Ensure CUDA is installed and `CUDA_HOME` is set.
  - "Torch not found": Install PyTorch or set `Torch_DIR`.
  - "pybind11 not found": Install pybind11 or add it as a submodule.
- **Build Errors**:
  - If stub files (e.g., empty `.cpp`) fail, comment them out in `CMakeLists.txt` until implemented.
  - CUDA arch mismatch: Update `CMAKE_CUDA_ARCHITECTURES` for your GPU (e.g., 86 for RTX 30-series).
- **Runtime Issues**:
  - "Library not found": Add `build/` to `LD_LIBRARY_PATH` (Linux) or `PATH` (Windows).
  - GPU errors: Ensure NVIDIA drivers are up-to-date.
- **Clean Rebuild**: Delete `build/` and restart from step 2.

## Project Structure

- `CMakeLists.txt`: Build configuration.
- `compiler/`: IR, lowering, backends.
- `acceleration/`: GPU kernels and management.
- `crsc/`: Custom runtime and kernels.
- `pybind11/`: (Optional) pybind11 submodule.

For more on the overall project, see `../README.md`. If issues persist, check the main README or open an issue.