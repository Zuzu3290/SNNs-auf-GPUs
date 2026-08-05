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

    def get_trainer(self, train_loader):
        from learning.training import SNNTrainer
        return SNNTrainer(self, train_loader, self.cfg, self.device)

    def get_inference(self, test_loader):
        from learning.inference import SNNTester
        return SNNTester(self, test_loader, self.cfg, self.device)

    def get_adversarial_evaluator(self, test_loader):
        from learning.adversarial_robustness import AdversarialEvaluator
        return AdversarialEvaluator(self, test_loader, self.cfg, self.device)
