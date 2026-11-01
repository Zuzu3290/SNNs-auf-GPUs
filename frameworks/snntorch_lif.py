"""snnTorch LIF behind BaseNeuron; every value comes from neuron.snntorch in network_architecture.yaml."""
from __future__ import annotations
import snntorch as snn
import torch
from snntorch import surrogate
from frameworks.snn_model import BaseNeuron

SURROGATES = {"atan": surrogate.atan, "fast_sigmoid": surrogate.fast_sigmoid}


class Neuron(BaseNeuron):
    """snnTorch hands state back to the caller; it is held here because init_hidden=True resets every snnTorch model in the process."""

    def __init__(self, cfg) -> None:
        super().__init__()
        n = cfg.NEURON["snntorch"]
        self.neuron = snn.Leaky(
            beta=n["beta"],
            threshold=n["threshold"],
            spike_grad=SURROGATES[n["surrogate"]["type"]](n["surrogate"]["alpha"]),
            reset_mechanism=n["reset_mechanism"],  # snnTorch defaults to soft reset; the other three hard-reset
            reset_delay=n["reset_delay"],          # snnTorch defaults to resetting one step late; the others reset at once
            init_hidden=False,                     # state is held here, in self.mem
        )
        self.mem = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        spikes, self.mem = self.neuron(x, self.mem)
        self.record(spikes)
        return spikes

    def reset(self) -> None:
        self.mem = None
