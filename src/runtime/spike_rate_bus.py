# EWMA spike rate singleton — receives ballot / dense push from the kernel layer,
# serves a smoothed rate to every subsystem (activity regulariser, energy feedback,
# cache coordinator, diagnostic logger). Eliminates duplicate .item() syncs.
from __future__ import annotations

import threading
import torch


class SpikeRateBus:
    """
    System-wide spike rate observable.

    Any component calls push_dense() or push_count() exactly once per batch.
    Any consumer reads .rate to get the EWMA-smoothed fraction (0–1).

    EWMA formula:  rate(t) = α × raw(t) + (1−α) × rate(t−1),  α = 0.1
    """

    singleton: "SpikeRateBus | None" = None
    class_lock = threading.Lock()

    def __init__(self, alpha: float = 0.1) -> None:
        self.ALPHA = alpha
        self.ema   = 0.0   # EWMA-smoothed spike rate
        self.raw   = 0.0   # last unsmoothed sample
        self.lock  = threading.Lock()

    @classmethod
    def get(cls, alpha: float = 0.1) -> "SpikeRateBus":
        """Return the singleton. First call sets alpha — subsequent calls ignore it."""
        with cls.class_lock:
            if cls.singleton is None:
                cls.singleton = cls(alpha)
            return cls.singleton

    def push_dense(self, spikes: torch.Tensor) -> None:
        """Push a spike tensor — single .sum().item() regardless of shape."""
        total = spikes.numel()
        if total == 0:
            return
        raw = float(spikes.detach().float().sum().item()) / float(total)
        self.update(raw)

    def push_count(self, fired: int, total: int) -> None:
        """Push raw integer counts — zero CUDA sync (caller already has the int)."""
        if total <= 0:
            return
        self.update(fired / total)

    def update(self, raw: float) -> None:
        with self.lock:
            self.raw = raw
            self.ema = self.ALPHA * raw + (1.0 - self.ALPHA) * self.ema

    def reset(self) -> None:
        """Reset EWMA — call at the start of each epoch to clear stale state."""
        with self.lock:
            self.ema = 0.0
            self.raw = 0.0

    @property
    def rate(self) -> float:
        """Current EWMA smoothed spike rate (0–1)."""
        return self.ema

    @property
    def raw_rate(self) -> float:
        """Most recent unsmoothed rate — useful for per-batch diagnostics."""
        return self.raw
