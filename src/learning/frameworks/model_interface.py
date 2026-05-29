from __future__ import annotations
from abc import ABC, abstractmethod
import torch


class ModelInterface(ABC):
    """
    Contract every model must satisfy to work with SNNTrainer, SNNTester,
    and AdversarialEvaluator.

    Boundary rule: forward() always receives and returns a PyTorch tensor.
    What runs inside — PyTorch, JAX, TensorFlow, or custom — is irrelevant.

    Backward pass:
      PyTorch models  — backward_pass() calls loss.backward() + optimizer.step()
      JAX / TF models — backward_pass() is a no-op; gradients are computed
                        inside forward() via jax.value_and_grad / GradientTape

    Adversarial robustness:
      JAX / TF models return False from is_differentiable() so the evaluator
      skips attack generation and runs clean evaluation only.
    """

    @abstractmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input and output are always PyTorch tensors."""

    @abstractmethod
    def backward_pass(self, loss: torch.Tensor, scaler=None, do_step: bool = True) -> None:
        """
        Compute gradients and update weights.

        scaler  — torch.amp.GradScaler for AMP; pass None to skip scaling
        do_step — set False to accumulate gradients without stepping
                  (used for gradient accumulation over N batches)

        PyTorch: scaler.scale(loss).backward(); if do_step: scaler.step(opt)
        JAX/TF:  pass — weights already updated inside forward()
        """

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear gradients. PyTorch: optimizer.zero_grad(). JAX/TF: pass."""

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

    def reset_state(self) -> None:
        """Reset hidden neuron state between sequences. Override if needed."""

    def is_differentiable(self) -> bool:
        """
        True if PyTorch autograd can trace back through this model to the input.
        JAX and TF backends return False — adversarial eval skips attacks.
        """
        return True

    def get_trainer(self, train_loader):
        from learning.training import SNNTrainer
        return SNNTrainer(self, train_loader, self.cfg, self.device)

    def get_inference(self, test_loader):
        from learning.inference import SNNTester
        return SNNTester(self, test_loader, self.cfg, self.device)

    def get_adversarial_evaluator(self, test_loader):
        from learning.adversarial_robustness import AdversarialEvaluator
        return AdversarialEvaluator(self, test_loader, self.cfg, self.device)
