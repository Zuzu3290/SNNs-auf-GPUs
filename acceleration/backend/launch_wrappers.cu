#include "../GPU_attributes/energy_management.h"
#include "../GPU_attributes/throughput_optimization.h"
#include <cuda_runtime.h>
#include <cstdio>

// Combined result from a single profiled kernel invocation
struct ProfiledLaunchResult {
    KernelEnergyResult energy;
    LaunchConfig       config;
};

void print_profiled_result(const ProfiledLaunchResult& r, const char* label) {
    printf("[ProfiledLaunch] %s\n", label);
    printf("  Grid : %d  |  Block : %d  |  Occupancy : %.1f%%\n",
           r.config.grid_size,
           r.config.block_size,
           r.config.theoretical_occupancy * 100.f);
    EnergyProfiler::print_result(r.energy, label);
}
