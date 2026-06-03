// Declares the T-in-register LIF kernel: one thread per (b,n) neuron, voltage held in a
// register across all T timesteps, eliminating the race condition in lif_basic and removing
// T global-memory voltage round-trips. Also declares the ballot variant that returns a
// per-warp spike mask alongside spikes for downstream sparse propagation.
#pragma once
#include <torch/extension.h>

// Standard temporal kernel — voltage in register, T loop in-thread, no race condition.
// Returns: spikes [B, N, T] float32
torch::Tensor lif_temporal_cuda(
    torch::Tensor input,    // [B, N, T] float32 CUDA
    torch::Tensor voltage,  // [B, N]    float32 CUDA — updated in place
    float v_th    = 1.0f,
    float tau_inv = 0.1f
);

// Ballot variant — same kernel, additionally returns warp spike masks.
// spike_mask[w, t] = 32-bit ballot of which of warp w's neurons fired at timestep t.
// Shape: [ceil(B*N/32), T] int32
// Use this to drive sparse weight-update or skip zero-spike propagation steps.
std::pair<torch::Tensor, torch::Tensor>
lif_temporal_ballot_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th    = 1.0f,
    float tau_inv = 0.1f
);
