// Implements LaunchSession::begin (compute_1d_launch + EnergyProfiler.start) and ::end (profiler stop, result assembly).
// Implements print_profiled_result: prints grid, block, occupancy percentage, and the full energy report to stdout.
// Called by dispatch_accelerated in engine.cu to sequence all three GPU-attribute modules around any kernel launch.
#include "launch_wrappers.h"
#include <cstdio>

void print_profiled_result(const ProfiledLaunchResult& r, const char* label) {
    printf("[ProfiledLaunch] %s\n", label);
    printf("  Grid : %d  |  Block : %d  |  Occupancy : %.1f%%\n",
           r.config.grid_size,
           r.config.block_size,
           r.config.theoretical_occupancy * 100.f);
    EnergyProfiler::print_result(r.energy, label);
}

// ---------------------------------------------------------------------------
// LaunchSession
// ---------------------------------------------------------------------------

void LaunchSession::begin(const void* kernel_fn, int n_elements,
                          cudaStream_t stream) {
    config = compute_1d_launch(kernel_fn, n_elements);
    profiler.start(stream);
}

ProfiledLaunchResult LaunchSession::end(cudaStream_t stream) {
    KernelEnergyResult e = profiler.stop(stream);
    return ProfiledLaunchResult{e, config};
}
