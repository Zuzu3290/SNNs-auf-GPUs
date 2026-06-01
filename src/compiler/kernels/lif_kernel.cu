#include "lif_kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>

static constexpr int BLOCK = 256;

// ── Forward kernel ────────────────────────────────────────────────────────────
//
// One thread per element. Computes:
//   integrated = beta * mem_in + input          (leaky integration)
//   spike      = integrated >= threshold ? 1 : 0
//   mem_out    = integrated - spike * threshold  (soft reset)
//
// mem_integrated is written out separately so the backward pass can reuse it
// without re-running the forward computation.

template <typename scalar_t>
__global__ void lif_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ mem_in,
    scalar_t*       __restrict__ spikes,
    scalar_t*       __restrict__ mem_out,
    scalar_t*       __restrict__ mem_integ,
    float beta,
    float threshold,
    int   total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    float integ = beta * static_cast<float>(mem_in[i])
                      +        static_cast<float>(input[i]);
    float s     = integ >= threshold ? 1.0f : 0.0f;

    mem_integ[i] = static_cast<scalar_t>(integ);
    spikes[i]    = static_cast<scalar_t>(s);
    mem_out[i]   = static_cast<scalar_t>(integ - s * threshold);
}

std::vector<torch::Tensor> lif_fused_forward(
    torch::Tensor input,
    torch::Tensor mem_in,
    float         beta,
    float         threshold
) {
    TORCH_CHECK(input.is_cuda(),        "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    TORCH_CHECK(mem_in.is_contiguous(), "mem_in must be contiguous");
    TORCH_CHECK(input.sizes() == mem_in.sizes(), "input and mem_in must have the same shape");

    auto spikes    = torch::empty_like(input);
    auto mem_out   = torch::empty_like(input);
    auto mem_integ = torch::empty_like(input);

    int total  = static_cast<int>(input.numel());
    int blocks = (total + BLOCK - 1) / BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lif_fused_forward", [&] {
        lif_forward_kernel<scalar_t><<<blocks, BLOCK>>>(
            input.data_ptr<scalar_t>(),
            mem_in.data_ptr<scalar_t>(),
            spikes.data_ptr<scalar_t>(),
            mem_out.data_ptr<scalar_t>(),
            mem_integ.data_ptr<scalar_t>(),
            beta,
            threshold,
            total
        );
    });

    return {spikes, mem_out, mem_integ};
}

// ── Backward kernel ───────────────────────────────────────────────────────────
//
// Surrogate gradient for the spike discontinuity: triangular window
//   surrogate(u) = max(0,  1 - |u - threshold| * slope)
//
// Chain rule (treating spikes as the only path for grad_spikes, and
// mem_out = integrated − spikes*threshold as the path for grad_mem_out):
//   d(loss)/d(integrated) = grad_spikes * surrogate  +  grad_mem_out
//   d(loss)/d(input)      = d(loss)/d(integrated)          (d_integrated/d_input  = 1)
//   d(loss)/d(mem_in)     = d(loss)/d(integrated) * beta   (d_integrated/d_mem_in = beta)

template <typename scalar_t>
__global__ void lif_backward_kernel(
    const scalar_t* __restrict__ grad_spikes,
    const scalar_t* __restrict__ grad_mem_out,
    const scalar_t* __restrict__ mem_integ,
    scalar_t*       __restrict__ grad_input,
    scalar_t*       __restrict__ grad_mem_in,
    float beta,
    float threshold,
    float slope,
    int   total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    float delta     = static_cast<float>(mem_integ[i]) - threshold;
    float surrogate = fmaxf(0.0f, 1.0f - fabsf(delta) * slope);

    float g_integ      = static_cast<float>(grad_spikes[i]) * surrogate
                       + static_cast<float>(grad_mem_out[i]);

    grad_input[i]  = static_cast<scalar_t>(g_integ);
    grad_mem_in[i] = static_cast<scalar_t>(g_integ * beta);
}

std::vector<torch::Tensor> lif_fused_backward(
    torch::Tensor grad_spikes,
    torch::Tensor grad_mem_out,
    torch::Tensor mem_integ,
    float         beta,
    float         threshold,
    float         slope
) {
    TORCH_CHECK(grad_spikes.is_contiguous(),  "grad_spikes must be contiguous");
    TORCH_CHECK(grad_mem_out.is_contiguous(), "grad_mem_out must be contiguous");
    TORCH_CHECK(mem_integ.is_contiguous(),    "mem_integ must be contiguous");

    auto grad_input  = torch::empty_like(grad_spikes);
    auto grad_mem_in = torch::empty_like(grad_spikes);

    int total  = static_cast<int>(grad_spikes.numel());
    int blocks = (total + BLOCK - 1) / BLOCK;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_spikes.scalar_type(), "lif_fused_backward", [&] {
        lif_backward_kernel<scalar_t><<<blocks, BLOCK>>>(
            grad_spikes.data_ptr<scalar_t>(),
            grad_mem_out.data_ptr<scalar_t>(),
            mem_integ.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            grad_mem_in.data_ptr<scalar_t>(),
            beta,
            threshold,
            slope,
            total
        );
    });

    return {grad_input, grad_mem_in};
}
