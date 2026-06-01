//below is an example snippet of a CUDA kernel for a simple leaky integrate-and-fire neuron model.
//simple kernel for one timestep over a batch of neurons:
//This is a toy example; in practice you’d split across timesteps and optimize memory layout 

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "GPU_attributes/memory_management.h"
#include "GPU_attributes/energy_management.h"
#include "GPU_attributes/throughput_optimization.h"

__global__ void snn_forward_kernel(
    const float* input,
    float* v,
    float* spikes,
    float v_th,
    float tau_inv,
    int64_t B,
    int64_t N,
    int64_t T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = B * N * T;
    if (idx >= total) return;

    // map idx to (b, n, t)
    int64_t b = idx / (N * T);
    int64_t rem = idx % (N * T);
    int64_t n = rem / T;
    int64_t t = rem % T;

    int64_t v_idx = b * N + n;
    int64_t inp_idx = b * N * T + n * T + t;

    // leaky integration
    float prev_v = v[v_idx];
    float exc = input[inp_idx];
    float new_v = prev_v * (1.0f - tau_inv) + exc;

    float spike = (new_v >= v_th) ? 1.0f : 0.0f;
    if (spike > 0.0f) new_v = 0.0f;  // reset

    v[v_idx] = new_v;
    spikes[inp_idx] = spike;
}

// ---------------------------------------------------------------------------
// PyTorch-callable wrapper
// ---------------------------------------------------------------------------
#include <ATen/cuda/CUDAContext.h>

torch::Tensor snn_forward_cuda(
    torch::Tensor input,    // [B, N, T]
    torch::Tensor voltage,  // [B, N]  – mutated in place
    float v_th    = 1.0f,
    float tau_inv = 0.1f
) {
    TORCH_CHECK(input.is_cuda() && voltage.is_cuda(), "tensors must be on CUDA");
    input   = input.contiguous();
    voltage = voltage.contiguous();

    const int64_t B = input.size(0);
    const int64_t N = input.size(1);
    const int64_t T = input.size(2);
    auto spikes = torch::zeros_like(input);

    const int64_t total = B * N * T;
    const int block     = 256;
    const int grid      = static_cast<int>((total + block - 1) / block);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    snn_forward_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(), voltage.data_ptr<float>(),
        spikes.data_ptr<float>(), v_th, tau_inv, B, N, T
    );
    return spikes;
}
