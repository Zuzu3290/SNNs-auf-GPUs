"""
Python interface to the fused LIF CUDA kernel.

Loads the kernel on first call to _load_ops() via torch.utils.cpp_extension.load(),
which invokes NVCC. Compilation takes 30-60 s the first time; the binary is cached
in a temp directory so subsequent imports are instant.

Public API
----------
lif_fused_step(input, mem, beta, threshold) -> (spikes, mem_new)
    Differentiable fused LIF step. Backward uses a triangular surrogate gradient
    so BPTT works without any changes to the training loop.

    Falls back to a pure-PyTorch soft-reset LIF when NVCC is unavailable (CPU
    fallback path in runtime.py handles this transparently).
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_KERNEL_DIR = Path(__file__).parent / "kernels"
_ops        = None
_load_failed = False


def _load_ops():
    """Compile and load the CUDA extension on first call. Cached after that."""
    global _ops, _load_failed
    if _ops is not None:
        return _ops
    if _load_failed:
        return None

    try:
        from torch.utils.cpp_extension import load
        logger.info("[CUDA_OPS] Compiling lif_cuda kernel (first run — ~30-60 s)…")
        _ops = load(
            name              = "lif_cuda",
            sources           = [
                str(_KERNEL_DIR / "lif_kernel.cu"),
                str(_KERNEL_DIR / "bindings.cpp"),
            ],
            extra_cuda_cflags = ["-O3", "--use_fast_math"],
            verbose           = False,
        )
        logger.info("[CUDA_OPS] lif_cuda kernel compiled and loaded")
        return _ops
    except Exception as exc:
        _load_failed = True
        logger.warning("[CUDA_OPS] Kernel compilation failed (%s) — falling back to PyTorch LIF", exc)
        return None


class _LIFFused(torch.autograd.Function):
    """
    Autograd wrapper around the fused CUDA LIF kernel.

    Forward  : membrane integration + threshold + spike + soft-reset in one kernel.
    Backward : triangular surrogate gradient (slope=10) through the spike discontinuity.
    """

    @staticmethod
    def forward(ctx, input, mem_in, beta, threshold):
        ops                          = _load_ops()
        spikes, mem_out, mem_integ   = ops.lif_forward(input, mem_in, float(beta), float(threshold))
        ctx.save_for_backward(mem_integ)
        ctx.beta      = float(beta)
        ctx.threshold = float(threshold)
        ctx.slope     = 10.0
        return spikes, mem_out

    @staticmethod
    def backward(ctx, grad_spikes, grad_mem_out):
        mem_integ, = ctx.saved_tensors
        ops        = _load_ops()
        grad_input, grad_mem_in = ops.lif_backward(
            grad_spikes.contiguous(),
            grad_mem_out.contiguous(),
            mem_integ,
            ctx.beta,
            ctx.threshold,
            ctx.slope,
        )
        return grad_input, grad_mem_in, None, None


def lif_fused_step(
    input:     torch.Tensor,
    mem:       torch.Tensor,
    beta:      float,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused LIF step dispatched to the CUDA kernel.
    Returns (spikes [B, N], mem_new [B, N]).
    """
    return _LIFFused.apply(input.contiguous(), mem.contiguous(), beta, threshold)
