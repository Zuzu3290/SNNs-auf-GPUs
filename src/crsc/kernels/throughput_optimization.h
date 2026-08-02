#pragma once
#include <cuda_runtime.h>

// Recommended 1-D grid/block layout and SM occupancy for a kernel
struct LaunchConfig {
    int   grid_size;
    int   block_size;
    float theoretical_occupancy;  // fraction [0, 1] of max warp slots occupied
};

// Use cudaOccupancyMaxPotentialBlockSize to find the block size that maximises
// occupancy for kernel_func over n_elements work items.
LaunchConfig compute_1d_launch(const void* kernel_func,
                                int n_elements,
                                size_t dynamic_shared_mem = 0);

// Smallest block size >= desired that is a multiple of the warp size (32),
// clamped to max_threads.
int warp_aligned_block(int desired, int max_threads = 1024);
