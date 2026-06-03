// SM-saturation warp-oriented LIF kernel: fills ALL SMs via a grid-stride loop
// regardless of B*N size. Hot path is fully async (no CPU-GPU sync per call).
// Device properties and grid size cached after first call — zero query overhead.
// Timing measured lazily every N calls for the energy feedback loop.
#include "lif_warp_oriented.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdio>

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ static void lif_warp_oriented_kernel(
    const float* __restrict__ input,    // [B, N, T]
    float*       __restrict__ voltage,  // [B, N]
    float*       __restrict__ spikes,   // [B, N, T]
    float v_th, float tau_inv,
    int64_t B, int64_t N, int64_t T
) {
    const int64_t total_neurons = B * N;
    const int64_t grid_stride   = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_neurons;
         idx += grid_stride)
    {
        const int64_t b = idx / N;
        const int64_t n = idx % N;

        float v = voltage[idx];  // register — zero global traffic across T

        const float* in_ptr = input  + b * N * T + n * T;
        float*       sp_ptr = spikes + b * N * T + n * T;

        const uint32_t active_mask = __ballot_sync(0xffffffffu, idx < total_neurons);

        for (int64_t t = 0; t < T; ++t) {
            v = v * (1.0f - tau_inv) + in_ptr[t];
            const bool fired = (v >= v_th);
            if (fired) v = 0.0f;
            sp_ptr[t] = fired ? 1.0f : 0.0f;
            __ballot_sync(active_mask, fired);  // sparsity mask, free warp op
        }

        voltage[idx] = v;
    }
}

// ---------------------------------------------------------------------------
// Process-wide cache — filled once, reused every call
// ---------------------------------------------------------------------------
struct DeviceCache {
    int  sm_count    = 0;
    int  cached_grid = 0;
    int  cached_bps  = -1;   // target_blocks_per_sm used when grid was computed
    bool ready       = false;
};
static DeviceCache s_dev;

static int compute_grid(int64_t total_neurons, int target_bps) {
    const int block            = 256;
    const int sm_fill_grid     = s_dev.sm_count * target_bps;
    const int neuron_ceil_grid = static_cast<int>((total_neurons + block - 1) / block);
    return (sm_fill_grid < neuron_ceil_grid) ? sm_fill_grid : neuron_ceil_grid;
}

// ---------------------------------------------------------------------------
// Hot path — fully async, no CPU-GPU sync, no device query after first call
// ---------------------------------------------------------------------------
WarpOrientedResult lif_warp_oriented_cuda(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th,
    float tau_inv,
    int   target_blocks_per_sm
) {
    TORCH_CHECK(input.is_cuda() && voltage.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(input.dim() == 3,   "input must be [B, N, T]");
    TORCH_CHECK(voltage.dim() == 2, "voltage must be [B, N]");

    input   = input.contiguous();
    voltage = voltage.contiguous();

    const int64_t B             = input.size(0);
    const int64_t N             = input.size(1);
    const int64_t T             = input.size(2);
    const int64_t total_neurons = B * N;

    // One-time device property query — cached for all subsequent calls
    if (!s_dev.ready) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, input.device().index());
        s_dev.sm_count = prop.multiProcessorCount;
        s_dev.ready    = true;
        printf("[WarpOriented] Device cached: %d SMs\n", s_dev.sm_count);
    }

    // Recompute grid only when target_blocks_per_sm changes
    if (s_dev.cached_bps != target_blocks_per_sm) {
        s_dev.cached_grid = compute_grid(total_neurons, target_blocks_per_sm);
        s_dev.cached_bps  = target_blocks_per_sm;
    }

    const int block = 256;
    const int grid  = s_dev.cached_grid;

    auto spikes = torch::zeros_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Async kernel launch — no sync, no blocking
    lif_warp_oriented_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        voltage.data_ptr<float>(),
        spikes.data_ptr<float>(),
        v_th, tau_inv, B, N, T
    );

    const float neurons_per_thread =
        static_cast<float>(total_neurons) / static_cast<float>(grid * block);

    // elapsed_ms = 0 signals "async — not measured this call"
    return WarpOrientedResult{spikes, 0.f, grid, neurons_per_thread};
}

// ---------------------------------------------------------------------------
// Timed variant — syncs once for benchmarking / energy feedback sampling.
// Call periodically (not every forward pass) to avoid sync overhead.
// ---------------------------------------------------------------------------
WarpOrientedResult lif_warp_oriented_timed(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th,
    float tau_inv,
    int   target_blocks_per_sm
) {
    TORCH_CHECK(input.is_cuda() && voltage.is_cuda(), "tensors must be on CUDA");
    input   = input.contiguous();
    voltage = voltage.contiguous();

    const int64_t B             = input.size(0);
    const int64_t N             = input.size(1);
    const int64_t T             = input.size(2);
    const int64_t total_neurons = B * N;

    if (!s_dev.ready) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, input.device().index());
        s_dev.sm_count = prop.multiProcessorCount;
        s_dev.ready    = true;
    }
    if (s_dev.cached_bps != target_blocks_per_sm) {
        s_dev.cached_grid = compute_grid(total_neurons, target_blocks_per_sm);
        s_dev.cached_bps  = target_blocks_per_sm;
    }

    const int block = 256;
    const int grid  = s_dev.cached_grid;

    auto spikes = torch::zeros_like(input);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventRecord(ev0, stream);

    lif_warp_oriented_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        voltage.data_ptr<float>(),
        spikes.data_ptr<float>(),
        v_th, tau_inv, B, N, T
    );

    cudaEventRecord(ev1, stream);
    cudaEventSynchronize(ev1);  // single sync — only for timed variant

    float elapsed_ms = 0.f;
    cudaEventElapsedTime(&elapsed_ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    printf("[WarpOriented|timed] grid=%d  block=%d  elapsed=%.4f ms  "
           "neurons/thread=%.2f\n", grid, block, elapsed_ms,
           static_cast<float>(total_neurons) / static_cast<float>(grid * block));

    return WarpOrientedResult{
        spikes, elapsed_ms, grid,
        static_cast<float>(total_neurons) / static_cast<float>(grid * block)
    };
}
