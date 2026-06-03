// Declares the SM-saturation warp-oriented LIF kernel.
// Launches exactly sm_count * target_blocks_per_sm blocks regardless of B*N size;
// a grid-stride loop distributes neurons across threads so every SM stays busy.
// Correct sequential LIF (voltage in register), warp ballot per step, energy feedback struct.
#pragma once
#include <torch/extension.h>

// Result from one warp-oriented dispatch — spikes plus runtime metrics
// the energy feedback loop uses to adapt the next launch.
struct WarpOrientedResult {
    torch::Tensor spikes;          // [B, N, T] float32
    float         elapsed_ms;      // CUDA-event kernel time
    int           blocks_launched; // actual grid size used
    float         neurons_per_thread; // B*N / total_threads — workload density
};

// Standard warp-oriented forward: SM-saturating grid, grid-stride loop over neurons.
// Returns WarpOrientedResult; caller uses elapsed_ms to tune next launch via feedback.
WarpOrientedResult lif_warp_oriented_cuda(
    torch::Tensor input,      // [B, N, T] float32 CUDA
    torch::Tensor voltage,    // [B, N]    float32 CUDA — updated in place
    float v_th              = 1.0f,
    float tau_inv           = 0.1f,
    int   target_blocks_per_sm = 8    // tunable: higher = more occupancy pressure
);
