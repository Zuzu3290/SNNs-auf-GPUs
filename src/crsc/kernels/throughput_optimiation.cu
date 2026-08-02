#include "throughput_optimization.h"
#include <cstdio>

LaunchConfig compute_1d_launch(const void* kernel_func,
                                int n_elements,
                                size_t dynamic_shared_mem)
{
    LaunchConfig cfg{};
    int min_grid = 0;

    cudaOccupancyMaxPotentialBlockSize(
        &min_grid, &cfg.block_size,
        kernel_func, dynamic_shared_mem, 0);

    cfg.grid_size = (n_elements + cfg.block_size - 1) / cfg.block_size;

    // Compute theoretical occupancy (active warps / max warps per SM)
    int max_active_blocks = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks, kernel_func, cfg.block_size, dynamic_shared_mem);

    int warps_active = max_active_blocks * (cfg.block_size / prop.warpSize);
    int warps_max    = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    cfg.theoretical_occupancy = (warps_max > 0)
        ? static_cast<float>(warps_active) / warps_max
        : 0.f;

    return cfg;
}

int warp_aligned_block(int desired, int max_threads) {
    const int warp_size = 32;
    int aligned = ((desired + warp_size - 1) / warp_size) * warp_size;
    return aligned > max_threads ? max_threads : aligned;
}
