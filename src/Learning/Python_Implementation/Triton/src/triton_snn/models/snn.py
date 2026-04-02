from __future__ import annotations

import torch
import torch.nn as nn

from triton_snn.kernels.lif import lif_step
from triton_snn.config import SNNConfig


class SurrogateSpike(torch.autograd.Function):
    """
    Straight-through style surrogate gradient.
    Forward: hard threshold.
    Backward: smooth derivative around threshold.
    """

    @staticmethod
    def forward(ctx, mem_minus_threshold: torch.Tensor):
        ctx.save_for_backward(mem_minus_threshold)
        return (mem_minus_threshold > 0).to(mem_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = 5.0
        grad = grad_output / (1.0 + alpha * x.abs()).pow(2)
        return grad


class TritonLIFLayer(nn.Module):
    """
    PyTorch layer shell that uses a Triton kernel for the membrane update.
    """

    def __init__(self, beta: float = 0.95, threshold: float = 1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, current: torch.Tensor, mem: torch.Tensor):
        new_mem, hard_spikes = lif_step(
            current,
            mem,
            beta=self.beta,
            threshold=self.threshold,
        )

        surrogate_spikes = SurrogateSpike.apply(new_mem - self.threshold)
        return new_mem, surrogate_spikes + (hard_spikes - surrogate_spikes).detach()


class TritonSNNClassifier(nn.Module):
    def __init__(self, cfg: SNNConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.spike1 = TritonLIFLayer(beta=cfg.beta, threshold=cfg.threshold)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        device = x.device

        mem1 = torch.zeros(batch, self.cfg.hidden_dim, device=device, dtype=torch.float32)
        spike_sum = torch.zeros(batch, self.cfg.output_dim, device=device, dtype=torch.float32)

        for _ in range(self.cfg.time_steps):
            cur1 = self.fc1(x)
            mem1, spk1 = self.spike1(cur1.float(), mem1)
            logits_t = self.fc2(spk1)
            spike_sum = spike_sum + logits_t

        return spike_sum / self.cfg.time_steps
