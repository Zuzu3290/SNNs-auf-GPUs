#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// GPU-attribute utilities (include dirs added in setup.py)
#include "energy_management.h"
#include "memory_management.h"
#include "throughput_optimization.h"

// ---------------------------------------------------------------------------
// LIF kernel  —  one thread per (batch, neuron, timestep) work item
//
// Leaky integration:  v_new = v_prev * (1 - tau_inv) + input
// Spike + hard reset: spike = 1 if v_new >= v_th else 0; v_new = 0 on spike
// ---------------------------------------------------------------------------
__global__ void lif_kernel(
    const float* __restrict__ input,   // [B, N, T]
    float*       __restrict__ voltage, // [B, N]  – updated in place
    float*       __restrict__ spikes,  // [B, N, T] – output
    float v_th, float tau_inv,
    int64_t B, int64_t N, int64_t T
) {
    const int64_t total = B * N * T;
    const int64_t idx   = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int64_t b     = idx / (N * T);
    const int64_t rem   = idx % (N * T);
    const int64_t n     = rem / T;

    const int64_t v_idx = b * N + n;

    float v     = voltage[v_idx];
    float exc   = input[idx];
    v           = v * (1.0f - tau_inv) + exc;

    float spike = (v >= v_th) ? 1.0f : 0.0f;
    if (spike > 0.0f) v = 0.0f;    // hard reset

    voltage[v_idx] = v;
    spikes[idx]    = spike;
}

// ---------------------------------------------------------------------------
// Standard forward pass (no energy tracking)
// ---------------------------------------------------------------------------
torch::Tensor snn_forward_cuda(
    torch::Tensor input,    // [B, N, T]  float32, CUDA
    torch::Tensor voltage,  // [B, N]     float32, CUDA  – mutated in place
    float v_th   = 1.0f,
    float tau_inv = 0.1f
) {
    TORCH_CHECK(input.is_cuda(),   "input must be a CUDA tensor");
    TORCH_CHECK(voltage.is_cuda(), "voltage must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3,  "input must be 3-D [B, N, T]");

    input   = input.contiguous();
    voltage = voltage.contiguous();

    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t T = input.size(2);

    auto spikes = torch::zeros_like(input);

    const int64_t total = B * N * T;
    LaunchConfig cfg = compute_1d_launch((const void*)lif_kernel, (int)total);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lif_kernel<<<cfg.grid_size, cfg.block_size, 0, stream>>>(
        input.data_ptr<float>(),
        voltage.data_ptr<float>(),
        spikes.data_ptr<float>(),
        v_th, tau_inv, B, N, T
    );
    return spikes;
}

// ---------------------------------------------------------------------------
// Profiled forward pass — returns spikes plus a full GPU energy report
// Return tuple: (spikes, elapsed_ms, power_before_mw, power_after_mw,
//                energy_mj, nvml_available)
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, float, float, float, float, bool>
snn_forward_profiled(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th    = 1.0f,
    float tau_inv = 0.1f
) {
    TORCH_CHECK(input.is_cuda(),   "input must be a CUDA tensor");
    TORCH_CHECK(voltage.is_cuda(), "voltage must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3,  "input must be 3-D [B, N, T]");

    input   = input.contiguous();
    voltage = voltage.contiguous();

    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t T = input.size(2);

    auto spikes = torch::zeros_like(input);

    const int64_t total = B * N * T;
    LaunchConfig cfg = compute_1d_launch((const void*)lif_kernel, (int)total);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    EnergyProfiler prof;
    prof.start(stream);

    lif_kernel<<<cfg.grid_size, cfg.block_size, 0, stream>>>(
        input.data_ptr<float>(),
        voltage.data_ptr<float>(),
        spikes.data_ptr<float>(),
        v_th, tau_inv, B, N, T
    );

    KernelEnergyResult r = prof.stop(stream);
    EnergyProfiler::print_result(r, "snn_forward");

    return std::make_tuple(
        spikes,
        r.elapsed_ms, r.power_before_mw, r.power_after_mw,
        r.energy_mj,  r.nvml_available
    );
}
