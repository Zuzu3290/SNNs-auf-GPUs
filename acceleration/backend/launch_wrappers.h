// Declares ProfiledLaunchResult (KernelEnergyResult + LaunchConfig in one struct) and LaunchSession.
// LaunchSession::begin computes the optimal block/grid via occupancy API and arms the energy profiler before a kernel launch.
// LaunchSession::end stops the profiler after the launch and returns the combined ProfiledLaunchResult.
#pragma once
#include <cuda_runtime.h>
#include "../GPU_attributes/energy_management.h"
#include "../GPU_attributes/throughput_optimization.h"

// ---------------------------------------------------------------------------
// ProfiledLaunchResult — combined outcome of one kernel invocation.
// Carries both the energy report and the launch geometry that was used.
// ---------------------------------------------------------------------------
struct ProfiledLaunchResult {
    KernelEnergyResult energy;   // elapsed_ms, power, energy_mj, nvml_available
    LaunchConfig       config;   // grid_size, block_size, theoretical_occupancy
};

void print_profiled_result(const ProfiledLaunchResult& r, const char* label);

// ---------------------------------------------------------------------------
// LaunchSession — RAII-style wrapper that sequences:
//   begin() → compute optimal launch config + start energy profiler
//   <caller launches the kernel using session.config>
//   end()   → stop profiler + return ProfiledLaunchResult
//
// Usage:
//   LaunchSession sess;
//   sess.begin((const void*)my_kernel, n_elements, stream);
//   my_kernel<<<sess.config.grid_size, sess.config.block_size, 0, stream>>>(...);
//   ProfiledLaunchResult r = sess.end(stream);
//   print_profiled_result(r, "my_kernel");
// ---------------------------------------------------------------------------
struct LaunchSession {
    LaunchConfig   config;
    EnergyProfiler profiler;

    // Compute optimal block/grid and arm the energy profiler.
    // kernel_fn: pointer to the __global__ function (for occupancy query)
    // n_elements: total work items (used to compute grid)
    void begin(const void* kernel_fn, int n_elements, cudaStream_t stream = nullptr);

    // Stop profiler, collect results. Call after the kernel launch completes.
    ProfiledLaunchResult end(cudaStream_t stream = nullptr);
};
