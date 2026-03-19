from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lif_step_kernel(
    input_ptr,
    mem_ptr,
    out_mem_ptr,
    spike_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mem = tl.load(mem_ptr + offsets, mask=mask, other=0.0)

    new_mem = beta * mem + x
    spikes = new_mem >= threshold
    reset_mem = tl.where(spikes, 0.0, new_mem)

    tl.store(out_mem_ptr + offsets, reset_mem, mask=mask)
    tl.store(spike_ptr + offsets, spikes.to(tl.float32), mask=mask)


def lif_step(
    input_tensor: torch.Tensor,
    mem_tensor: torch.Tensor,
    beta: float = 0.95,
    threshold: float = 1.0,
):
    """
    One GPU LIF-like step using a Triton kernel.

    Parameters
    ----------
    input_tensor:
        Shape [batch, neurons] or any contiguous tensor.
    mem_tensor:
        Same shape as input_tensor.
    beta:
        Leak/decay factor.
    threshold:
        Firing threshold.

    Returns
    -------
    new_mem, spikes
    """
    if not input_tensor.is_cuda or not mem_tensor.is_cuda:
        raise ValueError("lif_step requires CUDA tensors.")
    if input_tensor.shape != mem_tensor.shape:
        raise ValueError("input_tensor and mem_tensor must have the same shape.")
    if input_tensor.dtype != torch.float32 or mem_tensor.dtype != torch.float32:
        raise ValueError("This starter kernel expects float32 tensors.")
    if not input_tensor.is_contiguous() or not mem_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
        mem_tensor = mem_tensor.contiguous()

    n_elements = input_tensor.numel()
    new_mem = torch.empty_like(mem_tensor)
    spikes = torch.empty_like(input_tensor)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _lif_step_kernel[grid](
        input_tensor,
        mem_tensor,
        new_mem,
        spikes,
        n_elements,
        beta,
        threshold,
        BLOCK_SIZE=1024,
    )
    return new_mem, spikes
