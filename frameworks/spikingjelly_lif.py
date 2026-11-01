"""SpikingJelly LIF behind BaseNeuron; every value comes from neuron.spikingjelly in network_architecture.yaml."""
from __future__ import annotations
import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate
from frameworks.snn_model import BaseNeuron

SURROGATES = {"atan": surrogate.ATan, "sigmoid": surrogate.Sigmoid}


class Neuron(BaseNeuron):
    """SpikingJelly keeps state on the module and returns spikes alone."""

    def __init__(self, cfg) -> None:
        super().__init__()
        n = cfg.NEURON["spikingjelly"]
        if n["backend"] == "cupy":
            if n["step_mode"] != "m":
                # SpikingJelly would silently fall back to the torch path.
                raise ValueError("neuron.spikingjelly.backend = 'cupy' requires step_mode = 'm'.")
            if not hasattr(np, "int"):
                np.int = int  # spikingjelly 0.0.0.0.14's cupy backend reads np.int, which NumPy 2 removed
        self.neuron = neuron.LIFNode(
            tau=n["tau"],
            decay_input=n["decay_input"],    # SpikingJelly defaults to True, which scales the input gain down to 1/tau
            v_threshold=n["v_threshold"],
            v_reset=n["v_reset"],
            surrogate_function=SURROGATES[n["surrogate"]["type"]](alpha=n["surrogate"]["alpha"]),
            detach_reset=n["detach_reset"],  # False lets gradient flow through the reset, as in snnTorch
            step_mode=n["step_mode"],
            backend=n["backend"],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes = self.neuron(x)
        self.record(spikes)
        return spikes

    def reset(self) -> None:
        self.neuron.reset()  # this layer only, unlike functional.reset_net()

    def consumes_sequence(self) -> bool:
        """step_mode 'm' takes the whole [T, batch, ...] stack in one call, which lets the cupy backend fuse the time loop."""
        return self.neuron.step_mode == "m"
