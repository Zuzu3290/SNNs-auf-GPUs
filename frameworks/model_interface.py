from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class ModelInterface(ABC):
    """
    Contract every model must satisfy to work with SNNTrainer, SNNTester,
    and AdversarialEvaluator. PyTorch only — every model here is an
    nn.Module trained via standard PyTorch autograd (loss.backward() +
    optimizer.step()). No non-PyTorch backend (JAX, TensorFlow) is
    supported or accommodated; that flexibility was speculative and never
    used, so it's been removed rather than kept as unused surface area.
    """

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input and output are always PyTorch tensors."""

    @abstractmethod
    def backward_pass(self, loss: torch.Tensor, scaler=None, do_step: bool = True) -> None:
        """
        Compute gradients and update weights via standard PyTorch autograd.

        scaler  — torch.amp.GradScaler for AMP; pass None to skip scaling
        do_step — set False to accumulate gradients without stepping
                  (used for gradient accumulation over N batches)

        scaler.scale(loss).backward(); if do_step: scaler.step(optimizer)
        """

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear gradients: optimizer.zero_grad()."""

    @abstractmethod
    def train_mode(self) -> None:
        """Set model to training mode."""

    @abstractmethod
    def eval_mode(self) -> None:
        """Set model to evaluation mode."""

    @abstractmethod
    def get_lr(self) -> float:
        """Return current learning rate for logging."""

    @abstractmethod
    def get_state(self) -> dict:
        """Return serialisable state for checkpointing via torch.save()."""

    def tensor_format(self) -> str:
        """
        Tensor layout this model's forward() expects from the DataLoader.

        "TB" — [T, B, C, H, W]  time-first  (default — SNNTorch, Norse, SpikingJelly)
        "BT" — [B, T, C, H, W]  batch-first (Sinabs)

        The trainer transposes automatically before calling forward().
        Only override this if your framework needs batch-first input.
        """
        return "TB"

    def reset_state(self) -> None:
        """Reset hidden neuron state between sequences. Override if needed."""

    def credit_assignment(self) -> str:
        """Credit-assignment (backpropagation) algorithm this model trains
        with — required metadata alongside every timing/memory result per
        SNN_GPU_Evaluation_Metrics.md §2.3b, since it's the biggest
        determinant of what those numbers mean and isn't derivable from a
        latency number alone. Default "BPTT+SG" (BPTT + surrogate gradient)
        covers every framework currently wired in (Norse/SNNTorch/
        SpikingJelly/Sinabs all train this way); override only if a future
        framework uses a different algorithm (e-prop, FPTT, SLTT, ...)."""
        return "BPTT+SG"

    def synops_layer_map(self) -> dict:
        """Maps an ActivityMonitor-hooked spiking-layer name to the dense
        module immediately downstream of it (e.g. {'lif1': self.conv2}) —
        the module whose MACs get gated by that layer's firing rate for the
        SynOps energy estimate (SNN_GPU_Evaluation_Metrics.md §2.4/§4.4):
        SynOps_layer = firing_rate_layer * dense_MACs(downstream_module) * T.

        Empty dict (the default) means "unsupported for this framework" —
        e.g. Sinabs, whose LIF layers run once per forward() over the whole
        (B,T,...) tensor rather than once per timestep, the same reason its
        ActivityMonitor has no hooks (see snn_sinabs.py)."""
        return {}

    def get_trainer(self, train_loader):
        from learning.training import SNNTrainer
        return SNNTrainer(self, train_loader, self.cfg, self.device)

    def get_inference(self, test_loader):
        from learning.inference import SNNTester
        return SNNTester(self, test_loader, self.cfg, self.device)
