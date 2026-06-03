// Main CRSC kernel dispatcher: reads KernelConfig and routes to dispatch_basic, dispatch_accelerated,
// or dispatch_temporal. dispatch_temporal uses the T-in-register lif_temporal_cuda (correct LIF,
// warp ballot) and wraps it with the full GPU-attribute stack when accelerate:true + temporal:true.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>

#include "../../acceleration/kernel_config.h"
#include "../../acceleration/GPU_attributes/energy_management.h"
#include "../../acceleration/GPU_attributes/memory_management.h"
#include "../../acceleration/GPU_attributes/throughput_optimization.h"
#include "kernels/lif_temporal.h"

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
// Accelerated path — applies GPU-attribute modules then selects kernel:
//   temporal:true  → lif_temporal_cuda  (T-in-register, correct LIF)
//   temporal:false → lif_basic          (B*N*T threads, benchmark reference)
//   1. memory      → audit free VRAM headroom
//   2. throughput  → occupancy-tuned block/grid (B*N work items for temporal)
//   3. energy      → optional NVML energy profiler
// ---------------------------------------------------------------------------
static torch::Tensor dispatch_accelerated(
    const KernelConfig& cfg,
    torch::Tensor input,
    torch::Tensor voltage,
    float v_th, float tau_inv
) {
    const int64_t B       = input.size(0);
    const int64_t N       = input.size(1);
    const int64_t T       = input.size(2);
    const int64_t neurons = B * N;           // work items for temporal kernel
    const int64_t total   = neurons * T;     // work items for basic kernel

    // --- 1. Memory audit -----------------------------------------------
    if (cfg.optimize_memory) {
        size_t free_bytes = 0, total_bytes = 0;
        snn_query_memory(&free_bytes, &total_bytes);
        const size_t needed = (size_t)total * sizeof(float);
        if (free_bytes < needed * 2)
            printf("[MemoryMgr] WARNING: %.1f MB free, need ~%.1f MB\n",
                   free_bytes / 1048576.0, needed * 2 / 1048576.0);
    }

    // --- 2. Throughput: temporal uses B*N work items, basic uses B*N*T ---
    const int64_t work = cfg.temporal ? neurons : total;
    LaunchConfig lc;
    if (cfg.optimize_throughput) {
        lc = compute_1d_launch((const void*)lif_basic, (int)work);
        printf("[Throughput] kernel=%s  block=%d  grid=%d  occupancy=%.1f%%\n",
               cfg.temporal ? "temporal" : "basic",
               lc.block_size, lc.grid_size,
               lc.theoretical_occupancy * 100.f);
    } else {
        lc.block_size = 256;
        lc.grid_size  = static_cast<int>((work + 255) / 256);
        lc.theoretical_occupancy = 0.f;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --- 3. Energy profiler (start) ------------------------------------
    EnergyProfiler* prof = nullptr;
    if (cfg.profile_energy) {
        prof = new EnergyProfiler();
        prof->start(stream);
    }

    // --- Kernel dispatch -----------------------------------------------
    torch::Tensor spikes;
    if (cfg.temporal) {
        // T-in-register: correct sequential LIF, voltage in registers
        spikes = lif_temporal_cuda(input, voltage, v_th, tau_inv);
    } else {
        spikes = torch::zeros_like(input);
        lif_basic<<<lc.grid_size, lc.block_size, 0, stream>>>(
            input.data_ptr<float>(), voltage.data_ptr<float>(),
            spikes.data_ptr<float>(), v_th, tau_inv, B, N, T
        );
    }

    // --- 3. Energy profiler (stop) -------------------------------------
    if (prof) {
        KernelEnergyResult r = prof->stop(stream);
        EnergyProfiler::print_result(r, cfg.temporal ? "lif_temporal" : "lif_basic");
        delete prof;
    }

    return spikes;
}

// ---------------------------------------------------------------------------
// Public dispatcher — three routes based on KernelConfig:
//   accelerate:false              → dispatch_basic  (benchmark reference)
//   accelerate:true, temporal:false → dispatch_accelerated with lif_basic
//   accelerate:true, temporal:true  → dispatch_accelerated with lif_temporal
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
