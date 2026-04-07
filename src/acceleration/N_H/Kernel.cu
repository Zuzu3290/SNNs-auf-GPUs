//below is an example snippet of a CUDA kernel for a simple leaky integrate-and-fire neuron model.
//simple kernel for one timestep over a batch of neurons:
//This is a toy example; in practice you’d split across timesteps and optimize memory layout 

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include "memory_management.h"
#include "energy_management.h"
#include "throughput_optimization.h"

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