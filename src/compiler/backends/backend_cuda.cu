// src/compiler/backends/cuda_backend.cu

// Main GPU backend.

// This should:

// map IR ops to CUDA kernels
// choose launch dimensions
// call native CUDA execution routines

// This is likely your most important backend.