#pragma once
#include <torch/extension.h>
#include <vector>

// Forward: fused membrane_update + threshold + spike_gen + reset in one kernel.
// Returns {spikes, mem_out, mem_integrated}.
// mem_integrated is the membrane value *before* the soft reset — saved for backward.
std::vector<torch::Tensor> lif_fused_forward(
    torch::Tensor input,
    torch::Tensor mem_in,
    float         beta,
    float         threshold
);

// Backward: triangular surrogate gradient through the spike function.
// Returns {grad_input, grad_mem_in}.
std::vector<torch::Tensor> lif_fused_backward(
    torch::Tensor grad_spikes,
    torch::Tensor grad_mem_out,
    torch::Tensor mem_integrated,
    float         beta,
    float         threshold,
    float         slope
);
