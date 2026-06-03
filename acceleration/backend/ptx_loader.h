// Declares the five PTX runtime loader functions: ptx_load_from_file, ptx_load_from_string, ptx_get_kernel, ptx_launch_1d, ptx_unload.
// All functions wrap the CUDA Driver API (cuda.h), enabling kernels compiled to PTX to be loaded and launched at runtime.
// Enables per-SM-arch kernel selection (sm_75/sm_86/sm_89) and hot-swapping without recompiling the host extension.
#pragma once
#include <cuda.h>

// ---------------------------------------------------------------------------
// PTX runtime loader — CUDA Driver API wrapper.
//
// Allows kernel code compiled to PTX (Parallel Thread Execution IR) to be
// loaded and launched at runtime without recompiling the host extension.
// Enables per-SM-arch kernel selection and kernel hot-swapping.
//
// Typical usage:
//   CUmodule   mod = ptx_load_from_file("kernels/lif_sm89.ptx");
//   CUfunction fn  = ptx_get_kernel(mod, "lif_basic");
//   void* args[]   = {&d_input, &d_voltage, &d_spikes, &v_th, &tau_inv,
//                     &B, &N, &T};
//   ptx_launch_1d(fn, total, 256, 0, stream, args);
//   ptx_unload(mod);
// ---------------------------------------------------------------------------

// Load PTX source from a file on disk. Aborts on missing file or Driver error.
CUmodule ptx_load_from_file(const char* ptx_path);

// Load PTX from an in-memory null-terminated string (e.g. embedded at compile
// time via xxd or a raw string literal).
CUmodule ptx_load_from_string(const char* ptx_source);

// Retrieve a kernel function handle by name from a loaded module.
CUfunction ptx_get_kernel(CUmodule module, const char* kernel_name);

// Launch a 1-D kernel via the Driver API.
//   fn              : handle from ptx_get_kernel
//   n_elements      : total work items  (grid = ceil(n / block_size))
//   block_size      : threads per block
//   shared_mem_bytes: dynamic shared memory per block
//   stream          : CUDA stream (nullptr = default stream)
//   kernel_args     : array of void* pointers to each kernel argument
void ptx_launch_1d(CUfunction fn,
                   int n_elements, int block_size,
                   size_t shared_mem_bytes,
                   CUstream stream,
                   void** kernel_args);

// Unload a module and release its device memory.
void ptx_unload(CUmodule module);
