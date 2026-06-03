// T-in-register LIF kernel: one thread per (b,n) neuron, voltage held in a register across
// all T timesteps. Fixes the race condition in lif_basic (which launches B*N*T threads and
// has T threads concurrently writing the same voltage[b,n]) and eliminates T global-memory
// round-trips for the voltage state. Warp ballot provides a per-warp spike bitmap per step.
#include "lif_temporal.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel — WriteMask=false: no ballot output (lower overhead, standard path)
//          WriteMask=true:  writes ballot results to spike_mask array
// ---------------------------------------------------------------------------
template <bool WriteMask>
__global__ static void lif_temporal_kernel(
    const float* __restrict__ input,     // [B, N, T]
    float*       __restrict__ voltage,   // [B, N]  — updated in place
    float*       __restrict__ spikes,    // [B, N, T]
    int32_t*                  spike_mask,// [ceil(B*N/32), T]  (nullptr if !WriteMask)
    float v_th, float tau_inv,
    int64_t B, int64_t N, int64_t T
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // All threads in the warp must reach __ballot_sync; determine active set first.
    const bool active = (idx < B * N);
    const uint32_t active_mask = __ballot_sync(0xffffffffu, active);
    if (!active) return;

    const int64_t b = idx / N;
    const int64_t n = idx % N;

    // Voltage held in a register for the full T loop — zero global-mem traffic for state.
    float v = voltage[idx];

    const float*  in_ptr = input  + b * N * T + n * T;  // contiguous over T
    float*        sp_ptr = spikes + b * N * T + n * T;

    const int lane      = threadIdx.x & 31;
    const int warp_glob = static_cast<int>(idx >> 5);  // global warp index

    for (int64_t t = 0; t < T; ++t) {
        // Leaky integration + threshold + hard reset
        v += v * (-tau_inv) + in_ptr[t];   // v = v*(1-tau_inv) + input, fused
        const bool fired = (v >= v_th);
        if (fired) v = 0.0f;
        sp_ptr[t] = fired ? 1.0f : 0.0f;

        // Warp ballot: 32-bit mask of which neurons in this warp fired this step.
        // Active-mask ballot excludes out-of-bounds lanes in the last warp.
        if constexpr (WriteMask) {
            const uint32_t ballot = __ballot_sync(active_mask, fired);
            if (lane == 0)
                spike_mask[warp_glob * T + t] = static_cast<int32_t>(ballot);
        } else {
            // Still call ballot for convergence correctness; discard the result.
            __ballot_sync(active_mask, fired);
        }
    }

    voltage[idx] = v;
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

static void check_inputs(torch::Tensor input, torch::Tensor voltage) {
    TORCH_CHECK(input.is_cuda() && voltage.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(input.dim() == 3,  "input must be [B, N, T]");
    TORCH_CHECK(voltage.dim() == 2, "voltage must be [B, N]");
    TORCH_CHECK(input.size(0) == voltage.size(0) && input.size(1) == voltage.size(1),
                "batch and neuron dims must match between input and voltage");
}

torch::Tensor lif_temporal_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th,
    float tau_inv
) {
    check_inputs(input, voltage);
    input   = input.contiguous();
    voltage = voltage.contiguous();

    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t T = input.size(2);
    const int64_t neurons = B * N;

    auto spikes = torch::zeros_like(input);

    const int block = 256;
    const int grid  = static_cast<int>((neurons + block - 1) / block);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    lif_temporal_kernel<false><<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        voltage.data_ptr<float>(),
        spikes.data_ptr<float>(),
        nullptr,
        v_th, tau_inv, B, N, T
    );
    return spikes;
}

std::pair<torch::Tensor, torch::Tensor>
lif_temporal_ballot_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th,
    float tau_inv
) {
    check_inputs(input, voltage);
    input   = input.contiguous();
    voltage = voltage.contiguous();

    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t T = input.size(2);
    const int64_t neurons = B * N;
    const int64_t n_warps = (neurons + 31) / 32;

    auto spikes     = torch::zeros_like(input);
    auto spike_mask = torch::zeros({n_warps, T},
                                   torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(input.device()));

    const int block = 256;
    const int grid  = static_cast<int>((neurons + block - 1) / block);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    lif_temporal_kernel<true><<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        voltage.data_ptr<float>(),
        spikes.data_ptr<float>(),
        spike_mask.data_ptr<int32_t>(),
        v_th, tau_inv, B, N, T
    );
    return {spikes, spike_mask};
}
