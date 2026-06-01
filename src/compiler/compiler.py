"""
Compiler entry point.

compile_model(model, cfg) applies torch.compile() when cfg.TORCH_COMPILE is True.
TorchInductor fuses ops and generates optimised Triton/CUDA kernels automatically.
When disabled the original model is returned unchanged.

The custom LIF CUDA kernel (kernels/lif_kernel.cu) is retained for reference and
future deployment work on dedicated hardware.
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from skeleton import Settings

logger = logging.getLogger(__name__)


def compile_model(model: nn.Module, cfg: Settings) -> nn.Module:
    if not cfg.TORCH_COMPILE:
        return model

    logger.info("[COMPILER] Applying torch.compile() — backend=inductor")
    compiled = torch.compile(model)
    logger.info("[COMPILER] torch.compile() ready")
    return compiled
