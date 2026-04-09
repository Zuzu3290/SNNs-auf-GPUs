# CMakeLists.txt for SNN Compiler and Acceleration Modules

# Require a modern CMake for CUDA support
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project name and languages
project(SNNCompiler LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(CUDA REQUIRED)  # For CUDA compilation
find_package(Torch REQUIRED)  # PyTorch for integration
find_package(Python REQUIRED COMPONENTS Interpreter Development)  # For pybind11

# Include PyTorch and CUDA directories
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Enable CUDA language and set architecture (adjust for your GPU, e.g., 70 for V100, 80 for A100)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)  # Common architectures; add more as needed
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")  # Optimizations

# Include directories for headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/compiler/include
    ${CMAKE_CURRENT_SOURCE_DIR}/acceleration/N_H
    ${CMAKE_CURRENT_SOURCE_DIR}/crsc
)

# ------------------------------------------------------------------
# Compiler Library (IR, lowering, etc.)
# ------------------------------------------------------------------
add_library(snn_compiler SHARED
    compiler/src/ir.cpp
    compiler/src/lowering.cpp
    compiler/src/planner.cpp
    compiler/src/runtime.cpp
    compiler/src/scheduler.cpp
    compiler/src/diagnostics.cpp
    compiler/passes/device_annotation.cpp
    compiler/passes/fusion.cpp
    compiler/passes/op_rewrite.cpp
    compiler/backends/backend_cuda.cu
    compiler/backends/ptx_backend.cu
)
target_link_libraries(snn_compiler ${TORCH_LIBRARIES} CUDA::cudart)

# ------------------------------------------------------------------
# Acceleration Library (GPU kernels and wrappers)
# ------------------------------------------------------------------
add_library(snn_acceleration SHARED
    acceleration/N_H/Kernel.cu
    acceleration/N_H/SNN_mapping.cpp
    acceleration/N_H/backend/launch_wrappers.cu
    acceleration/N_H/backend/ptx_loader.cu
    acceleration/N_H/GPU_attributes/energy_management.cu
    acceleration/N_H/GPU_attributes/memory_management.cu
    acceleration/N_H/GPU_attributes/throughput_optimiation.cu
)
target_link_libraries(snn_acceleration CUDA::cudart)

# ------------------------------------------------------------------
# CRSC Library (Custom runtime kernels)
# ------------------------------------------------------------------
add_library(snn_crsc SHARED
    crsc/binding.cpp
    crsc/compiler_bridge.cpp
    crsc/engine.cu
    crsc/kernels/decode_ops.cu
    crsc/kernels/membrane_ops.cu
    crsc/kernels/reset_ops.cu
    crsc/kernels/spike_ops.cu
    crsc/kernels/threshold_ops.cu
)
target_link_libraries(snn_crsc CUDA::cudart)

# ------------------------------------------------------------------
# PyBind Module for Python Integration
# ------------------------------------------------------------------
# Note: Assuming pybind11 is available; adjust if needed
find_package(pybind11 REQUIRED)
pybind11_add_module(snn_backend
    compiler/pybind_module.cpp
)
target_link_libraries(snn_backend PRIVATE snn_compiler snn_acceleration snn_crsc ${TORCH_LIBRARIES})

# ------------------------------------------------------------------
# Optional: Tests (commented out since tests are Python-based)
# ------------------------------------------------------------------
# add_executable(test_ir compiler/tests/test_ir.py)  # Not applicable for Python tests

# ------------------------------------------------------------------
# Installation (optional, for packaging)
# ------------------------------------------------------------------
install(TARGETS snn_compiler snn_acceleration snn_crsc snn_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)