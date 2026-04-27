// src/compiler/backends/ptx_backend.cu

// This is optional at first, but valid for the future.

// Its role is not to load whole models.
// Its role is to:

// load PTX kernels
// resolve symbols
// launch compiled GPU kernels

// So it is a kernel backend, not a model parser.