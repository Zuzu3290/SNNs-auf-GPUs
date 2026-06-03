// Main CRSC kernel dispatcher: reads KernelConfig and routes to dispatch_basic or dispatch_accelerated.
// dispatch_basic runs lif_basic with a fixed 256-thread block and zero profiling overhead.
// dispatch_accelerated applies all three GPU-attribute modules: memory audit, occupancy-tuned launch, and optional NVML energy profiling.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>

#include "../../acceleration/kernel_config.h"
#include "../../acceleration/GPU_attributes/energy_management.h"
#include "../../acceleration/GPU_attributes/memory_management.h"
#include "../../acceleration/GPU_attributes/throughput_optimization.h"

// ---------------------------------------------------------------------------
// Basic LIF kernel — fixed 256-thread launch, no GPU-attribute overhead.
// One thread per (batch × neuron × timestep) work item.
// ---------------------------------------------------------------------------
__global__ static void lif_basic(
    const float* __restrict__ input,    // [B, N, T]
    float*       __restrict__ voltage,  // [B, N]  — updated in place
    float*       __restrict__ spikes,   // [B, N, T]
    float v_th, float tau_inv,
    int64_t B, int64_t N, int64_t T
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N * T) return;

    const int64_t b   = idx / (N * T);
    const int64_t rem = idx % (N * T);
    const int64_t n   = rem / T;

    float v   = voltage[b * N + n];
    v         = v * (1.0f - tau_inv) + input[idx];

    const float spike = (v >= v_th) ? 1.0f : 0.0f;
    if (spike > 0.0f) v = 0.0f;   // hard reset

    voltage[b * N + n] = v;
    spikes[idx]        = spike;
}

// ---------------------------------------------------------------------------
// Standard path — bare kernel, fixed block=256, no profiling/tuning.
// ---------------------------------------------------------------------------
static torch::Tensor dispatch_basic(
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th, float tau_inv
) {
    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t T = input.size(2);

    auto spikes = torch::zeros_like(input);

    const int64_t total = B * N * T;
    const int block = 256;
    const int grid  = static_cast<int>((total + block - 1) / block);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    lif_basic<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(), voltage.data_ptr<float>(),
        spikes.data_ptr<float>(), v_th, tau_inv, B, N, T
    );
    return spikes;
}

// ---------------------------------------------------------------------------
// Accelerated path — applies the three GPU-attribute modules in order:
//   1. memory      → audit free headroom before allocating output tensor
//   2. throughput  → auto-tune block/grid via occupancy API
//   3. energy      → wrap launch with NVML-backed EnergyProfiler
// ---------------------------------------------------------------------------
static torch::Tensor dispatch_accelerated(
    const KernelConfig& cfg,
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th, float tau_inv
) {
    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t T = input.size(2);
    const int64_t total = B * N * T;

    // --- 1. Memory audit -----------------------------------------------
    if (cfg.optimize_memory) {
        size_t free_bytes = 0, total_bytes = 0;
        snn_query_memory(&free_bytes, &total_bytes);
        const size_t needed = total * sizeof(float);   // output spikes tensor
        if (free_bytes < needed * 2) {
            // 2× headroom: spikes + any internal allocations
            printf("[MemoryMgr] WARNING: %.1f MB free, need ~%.1f MB — "
                   "consider reducing batch size\n",
                   free_bytes  / 1048576.0,
                   needed * 2  / 1048576.0);
        }
    }

    auto spikes = torch::zeros_like(input);

    // --- 2. Throughput: auto-tune block/grid ---------------------------
    LaunchConfig lc;
    if (cfg.optimize_throughput) {
        lc = compute_1d_launch((const void*)lif_basic, (int)total);
        printf("[Throughput] block=%d  grid=%d  occupancy=%.1f%%\n",
               lc.block_size, lc.grid_size,
               lc.theoretical_occupancy * 100.f);
    } else {
        lc.block_size = 256;
        lc.grid_size  = static_cast<int>((total + 255) / 256);
        lc.theoretical_occupancy = 0.f;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --- 3. Energy profiler --------------------------------------------
    EnergyProfiler* prof = nullptr;
    if (cfg.profile_energy) {
        prof = new EnergyProfiler();
        prof->start(stream);
    }

    lif_basic<<<lc.grid_size, lc.block_size, 0, stream>>>(
        input.data_ptr<float>(), voltage.data_ptr<float>(),
        spikes.data_ptr<float>(), v_th, tau_inv, B, N, T
    );

    if (prof) {
        KernelEnergyResult r = prof->stop(stream);
        EnergyProfiler::print_result(r, "lif_basic [accelerated]");
        delete prof;
    }

    return spikes;
}

// ---------------------------------------------------------------------------
// Public dispatcher — routes based on KernelConfig::accelerate flag.
// Call with KERNEL_CONFIG_DEFAULT or KERNEL_CONFIG_ACCELERATED (from
// kernel_config.h), or build a custom KernelConfig from accel_config.yaml.
// ---------------------------------------------------------------------------
torch::Tensor snn_engine_dispatch(
    const KernelConfig& cfg,
    torch::Tensor input,    // [B, N, T] float32 on CUDA
    torch::Tensor voltage,  // [B, N]    float32 on CUDA — mutated in place
    float v_th    = 1.0f,
    float tau_inv = 0.1f
) {
    TORCH_CHECK(input.is_cuda()   && voltage.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(input.dim() == 3,  "input must be [B, N, T]");
    TORCH_CHECK(voltage.dim() == 2, "voltage must be [B, N]");

    input   = input.contiguous();
    voltage = voltage.contiguous();

    if (!cfg.accelerate)
        return dispatch_basic(input, voltage, v_th, tau_inv);

    return dispatch_accelerated(cfg, input, voltage, v_th, tau_inv);
}
