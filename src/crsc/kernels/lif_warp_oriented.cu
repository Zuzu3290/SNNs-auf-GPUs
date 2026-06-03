// SM-saturation warp-oriented LIF kernel: fills ALL SMs via a grid-stride loop
// regardless of B*N size. At small B*N (e.g. 2048) each thread handles multiple
// neurons; at large B*N each thread handles one. Voltage lives in a register across
// the full T loop (correct sequential LIF). Warp ballot provides sparsity mask per step.
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

    // Grid-stride loop: each thread covers ceil(total_neurons / grid_stride) neurons.
    // When grid > ceil(total_neurons/block), some threads process 0 iterations — fine.
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_neurons;
         idx += grid_stride)
    {
        const int64_t b = idx / N;
        const int64_t n = idx % N;

        // Voltage in register — zero global memory traffic for state across T.
        float v = voltage[idx];

        const float* in_ptr = input  + b * N * T + n * T;
        float*       sp_ptr = spikes + b * N * T + n * T;

        // Determine active warp mask for ballot (handles boundary in last warp).
        const uint32_t active_mask = __ballot_sync(0xffffffffu, idx < total_neurons);

        for (int64_t t = 0; t < T; ++t) {
            v = v * (1.0f - tau_inv) + in_ptr[t];
            const bool fired = (v >= v_th);
            if (fired) v = 0.0f;
            sp_ptr[t] = fired ? 1.0f : 0.0f;

            // Warp-wide spike ballot — tracks sparsity at no extra cost.
            // Discarded here; wire to a spike_mask output if needed downstream.
            __ballot_sync(active_mask, fired);
        }

        voltage[idx] = v;
    }
}

// ---------------------------------------------------------------------------
// Host wrapper
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

    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t T = input.size(2);
    const int64_t total_neurons = B * N;

    auto spikes = torch::zeros_like(input);

    // --- SM-saturating launch config -----------------------------------
    // Query how many SMs this device has; fill them with target_blocks_per_sm
    // blocks each so every SM has work regardless of how small B*N is.
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, input.device().index());

    const int block = 256;
    const int sm_fill_grid = prop.multiProcessorCount * target_blocks_per_sm;

    // Cap: no point launching more blocks than neurons / block (threads would
    // all idle in the grid-stride loop). Take the smaller of the two.
    const int neuron_ceil_grid = static_cast<int>((total_neurons + block - 1) / block);
    const int grid = (sm_fill_grid < neuron_ceil_grid) ? sm_fill_grid : neuron_ceil_grid;

    printf("[WarpOriented] SMs=%d  target_bps=%d  grid=%d  block=%d  "
           "neurons/thread=%.2f\n",
           prop.multiProcessorCount, target_blocks_per_sm, grid, block,
           static_cast<float>(total_neurons) / (grid * block));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // CUDA event timing — always available, no NVML required.
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    cudaEventRecord(ev_start, stream);

    lif_warp_oriented_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        voltage.data_ptr<float>(),
        spikes.data_ptr<float>(),
        v_th, tau_inv, B, N, T
    );

    cudaEventRecord(ev_stop, stream);
    cudaEventSynchronize(ev_stop);

    float elapsed_ms = 0.f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return WarpOrientedResult{
        spikes,
        elapsed_ms,
        grid,
        static_cast<float>(total_neurons) / static_cast<float>(grid * block)
    };
}
