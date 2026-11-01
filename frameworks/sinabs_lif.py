"""Sinabs LIF behind BaseNeuron.

Three sinabs defaults differ from the other frameworks and are set explicitly from the neuron spec:
norm_input=True scales the input gain down to (1 - decay), spike_fn=MultiSpike lets a neuron fire several
spikes per step, and reset_fn=MembraneSubtract is a soft reset.
"""
from __future__ import annotations
import sinabs.layers as sl
import torch
from sinabs.activation import (
    Gaussian, Heaviside, MembraneReset, MembraneSubtract, MultiGaussian,
    MultiSpike, PeriodicExponential, SingleExponential, SingleSpike,
)
from frameworks.snn_model import BaseNeuron

SPIKE_FNS = {"single": SingleSpike, "multi": MultiSpike}


class Neuron(BaseNeuron):
    def __init__(self, cfg) -> None:
        super().__init__()
        n = cfg.NEURON["sinabs"]

        # The surrogate gradient named by surrogate.type, with that type's own parameter names.
        s = n["surrogate"]
        if s["type"] == "single_exponential":
            surrogate = SingleExponential(grad_width=s["grad_width"], grad_scale=s["grad_scale"])
        elif s["type"] == "periodic_exponential":
            surrogate = PeriodicExponential(grad_width=s["grad_width"], grad_scale=s["grad_scale"])
        elif s["type"] == "gaussian":
            surrogate = Gaussian(mu=s["mu"], sigma=s["sigma"], grad_scale=s["grad_scale"])
        elif s["type"] == "multi_gaussian":
            surrogate = MultiGaussian(mu=s["mu"], sigma=s["sigma"], h=s["h"], s=s["s"], grad_scale=s["grad_scale"])
        else:
            surrogate = Heaviside(window=s["window"])

        self.neuron = sl.LIF(
            tau_mem=n["tau_mem"],    # in timesteps; sinabs has no dt
            tau_syn=n["tau_syn"],    # null = membrane only
            spike_threshold=torch.as_tensor(n["spike_threshold"]),
            spike_fn=SPIKE_FNS[n["spike_fn"]],
            reset_fn=MembraneReset(reset_value=n["v_reset"]) if n["reset_mechanism"] == "zero" else MembraneSubtract(),
            min_v_mem=n["min_v_mem"],
            train_alphas=n["train_alphas"],
            norm_input=n["norm_input"],
            surrogate_grad_fn=surrogate,
        )
        # sinabs always registers tau_mem as a trainable parameter; the other three cannot learn theirs, so it is frozen.
        for parameter in self.neuron.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sinabs reads input as (batch, time, ...); without the unsqueeze the channel axis would be taken as time.
        spikes = self.neuron(x.unsqueeze(1)).squeeze(1)
        self.record(spikes)
        return spikes

    def reset(self) -> None:
        """Restore the state buffers to zero size, so the next batch re-infers its shape (reset_states() keeps the old batch shape)."""
        for name, buffer in list(self.neuron.named_buffers()):
            self.neuron.register_buffer(name, torch.zeros((0), device=buffer.device))
