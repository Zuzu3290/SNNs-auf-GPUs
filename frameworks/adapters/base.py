"""The contract every framework's LIF layer must satisfy.

Ported from the SNNs_2 comparison pipeline. This is the whole adapter layer: the
shared network, the training loop and the metrics talk only to this interface and
never learn which framework they got.

Only the NEURON differs between snnTorch, SpikingJelly, Norse and Sinabs. Conv2d,
MaxPool2d and Linear are plain torch.nn in all four cases. So instead of four network
classes that can drift apart, there is ONE network class (frameworks/spiking_net.py)
and four implementations of the single layer below.

State handling is the interesting part, and it differs per framework:

    snnTorch      returns (spikes, state) and takes the previous state back
    Norse         same, with its own state namedtuple
    SpikingJelly  keeps state on the module, returns spikes alone
    Sinabs        keeps state in a buffer, and expects a WHOLE SEQUENCE

Hiding all four behind `forward(x) -> spikes` for one timestep is what makes the
network framework-agnostic. It also makes the classic silent SNN bug -- forgetting to
reset state between batches -- structurally impossible, because the network resets
itself at the start of every forward pass rather than relying on a call site.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseLIF(nn.Module):
    """One layer of leaky integrate-and-fire neurons, for ONE timestep."""

    def __init__(self) -> None:
        super().__init__()
        # Spike counting for the spike-rate metric.
        #
        # Off by default and deliberately so: summing spikes costs an extra GPU kernel
        # per layer per timestep, and this project measures wall-clock time. Enable it
        # only for dedicated measurement passes.
        self.count_spikes: bool = False
        self.spike_total: torch.Tensor | float = 0.0  # stays on-device, never .item()ed here
        self.spike_slots: int = 0                     # neurons x timesteps x batch seen
        # Shape of one sample's output, e.g. (12, 30, 30). Learned from the first counted
        # step, so a results file can report neurons per layer without the network
        # having to describe itself.
        self.spike_shape: tuple[int, ...] | None = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Charge, fire, reset for one timestep. Returns spikes shaped like x."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all neuron state. Called at the start of every batch."""

    def has_state(self) -> bool:
        """Is any neuron state currently held?

        Not part of the computational contract -- this exists so a reset can be
        VERIFIED rather than assumed. Each framework stores state somewhere different,
        so each adapter answers for itself.
        """
        return False

    def membrane(self) -> torch.Tensor | None:
        """Current membrane potential, or None if no state is held yet.

        Exists so equivalence_check.py can compare membrane trajectories WITHOUT
        knowing which framework it is looking at -- each framework parks the membrane
        somewhere different (an attribute we hold, a module attribute, a state
        namedtuple, a registered buffer). Post-reset value, i.e. after any spike has
        already knocked it down.
        """
        return None

    def describe(self) -> dict[str, Any]:
        """Framework-specific settings actually in force, for the run record."""
        return {}

    # ---- spike statistics -------------------------------------------------------
    def _record(self, spikes: torch.Tensor) -> None:
        """Accumulate spike statistics without stalling the GPU.

        `.sum()` stays a device tensor -- calling `.item()` here would force a
        host/device sync on every layer of every timestep and wreck the timing
        measurements. It is read back only when someone asks for the rate.
        """
        if not self.count_spikes:
            return
        if self.spike_shape is None:
            self.spike_shape = tuple(spikes.shape[1:])  # drop the batch dimension
        self.spike_total = self.spike_total + spikes.detach().sum()
        self.spike_slots += spikes.numel()

    def reset_spike_stats(self) -> None:
        self.spike_total = 0.0
        self.spike_slots = 0
        self.spike_shape = None

    def neurons(self) -> int:
        """Neurons in this layer, per sample. 0 until something has been counted."""
        if self.spike_shape is None:
            return 0
        count = 1
        for dimension in self.spike_shape:
            count *= dimension
        return count

    def spike_rate(self) -> float:
        """Mean spikes per neuron per timestep, as a FRACTION (x100 for a percentage).

        Reads the accumulated device tensor exactly once. Note this is only a true
        fraction while the spike function is binary -- sinabs' MultiSpike can exceed
        1.0, which is one of the reasons the neuron spec pins it to SingleSpike.
        """
        if self.spike_slots == 0:
            return 0.0
        total = self.spike_total
        if isinstance(total, torch.Tensor):
            total = total.item()
        return float(total) / self.spike_slots
