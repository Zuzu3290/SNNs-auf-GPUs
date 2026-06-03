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

// Hot path — fully async, no CPU-GPU sync. elapsed_ms = 0 in returned result.
// Use this in every training forward pass.
WarpOrientedResult lif_warp_oriented_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th              = 1.0f,
    float tau_inv           = 0.1f,
    int   target_blocks_per_sm = 8
);

// Timed variant — syncs GPU once to measure elapsed_ms accurately.
// Call periodically (e.g. every 50 steps) for energy feedback, not every call.
WarpOrientedResult lif_warp_oriented_timed(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th              = 1.0f,
    float tau_inv           = 0.1f,
    int   target_blocks_per_sm = 8
);
