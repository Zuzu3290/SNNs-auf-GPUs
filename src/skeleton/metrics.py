"""Distributed metric tracking and aggregation.

Provides a lightweight running-average tracker and helpers to
``all_reduce`` scalars across the process group.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.distributed as dist


def all_reduce_scalar(
    value: float,
    world_size: int,
    device: torch.device,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> float:
    """Reduce a Python float across all ranks and return the mean.

    Parameters
    ----------
    value : float
        Local scalar to reduce.
    world_size : int
        Total number of processes.
    device : torch.device
        Device on which to create the tensor for communication.
    op : dist.ReduceOp
        Reduction operation (default: SUM, then divide by world_size).
    """
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    return (tensor / world_size).item()


def compute_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute top-1 accuracy (%).

    Parameters
    ----------
    output : Tensor
        Raw logits of shape ``(B, C)``.
    target : Tensor
        Ground-truth labels of shape ``(B,)``.
    """
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = preds.eq(target).sum().item()
        return correct / target.size(0) * 100.0


class MetricTracker:
    """Accumulates running sums and counts for arbitrary named metrics.

    Usage::

        tracker = MetricTracker()
        tracker.update("loss", 0.45, n=32)
        tracker.update("acc", 91.2, n=32)
        print(tracker.averages())  # {"loss": ..., "acc": ...}
    """

    def __init__(self) -> None:
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._sums.clear()
        self._counts.clear()

    def update(self, name: str, value: float, n: int = 1) -> None:
        """Add *value* (weighted by *n* samples) to metric *name*."""
        self._sums[name] = self._sums.get(name, 0.0) + value * n
        self._counts[name] = self._counts.get(name, 0) + n

    def average(self, name: str) -> float:
        """Return the running average for *name*."""
        total = self._counts.get(name, 0)
        if total == 0:
            return 0.0
        return self._sums[name] / total

    def averages(self) -> Dict[str, float]:
        """Return running averages for every tracked metric."""
        return {k: self.average(k) for k in self._sums}