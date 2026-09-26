"""The non-spiking control: the SNN's architecture with ReLU where the LIF was.

This is the denominator for every SNN claim in the experiment. It reads the SAME event
frames from the SAME cache through the SAME pipeline, is built from the SAME
`convolution.blocks` list, trains with the SAME optimiser and loss, and is run over the
SAME T timesteps. One difference: its units pass a continuous value to the next layer
instead of accumulating a membrane potential and emitting a spike.

That single difference is what the experiment measures, and it has two consequences
worth stating because they ARE the result, not artefacts:

  * No state across time. A LIF carries its membrane from one timestep to the next, so
    the SNN can integrate evidence over a sequence. ReLU cannot -- each timestep is
    independent and only the averaged output ties them together. On a task whose label
    is a motion this should cost the control model real accuracy; on near-static data it
    should cost it nothing. That gradient across the four datasets is the finding.

  * Dense activations. Every unit contributes every timestep, so activation density runs
    far higher than a spiking layer's, and the operation count is multiply-accumulates
    rather than accumulates. Both are measured, not assumed.

This is NOT a frame-camera CNN. There is no ordinary video anywhere in this project. It
is a conventional network reading event frames -- the standard event-vision baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from frameworks.model_interface import ModelInterface
from frameworks.relu_activation import relu_factory
from frameworks.spiking_net import build_network
from learning.utilities import ActivityMonitor, build_loss, build_optimizer
from skeleton.snn_config import Settings

# What cfg.FRAMEWORK is set to for a control run, so every results row, plot legend and
# run_id says plainly which model produced it.
FRAMEWORK_NAME = "cnn"


class CNNModel(ModelInterface, nn.Module):
    """The non-spiking control model. Same network, ReLU units."""

    def __init__(self, cfg: Settings, spiking_readout: bool = True):
        super().__init__()
        cfg.FRAMEWORK = FRAMEWORK_NAME
        self.cfg = cfg
        self.framework = FRAMEWORK_NAME
        self.device = torch.device(cfg.DEVICE)

        # The same builder the four SNN backends use, so the layer list cannot drift.
        self.net = build_network(relu_factory(cfg), cfg, spiking_readout=spiking_readout)
        self.to(self.device)

        self.optimizer = build_optimizer(self.parameters(), cfg)
        self.loss_fn = build_loss(cfg)

        named = self.net.named_lif_layers()
        self.activity = ActivityMonitor(
            {name: layer for name, layer in named.items() if name != "lif_out"}
        )

    # ---- ModelInterface ---------------------------------------------------------
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: [T, B, C, H, W]  ->  [T, B, num_classes], same shape contract as the SNN."""
        self.activity.clear()
        return self.net(data)

    def backward_pass(self, loss: torch.Tensor, scaler=None, do_step: bool = True) -> None:
        if scaler is not None:
            scaler.scale(loss).backward()
            if do_step:
                scaler.step(self.optimizer)
                scaler.update()
        else:
            loss.backward()
            if do_step:
                self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def train_mode(self) -> None:
        self.train()

    def eval_mode(self) -> None:
        self.eval()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def get_state(self) -> dict:
        return {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "framework": self.framework,
            "neuron": self.describe_neuron(),
        }

    def reset_state(self) -> None:
        self.net.reset()

    def credit_assignment(self) -> str:
        """Plain backpropagation through time -- no surrogate gradient, because there is no non-differentiable spike to get past."""
        return "BPTT"

    def synops_layer_map(self) -> dict:
        """Same derivation as the SNN's, so the operation count is gathered identically. The per-operation energy constant differs (MAC, not AC) and is applied downstream."""
        mapping = {}
        for name in self.net.named_lif_layers():
            if name == "lif_out":
                continue
            downstream = self.net.dense_after(name)
            if downstream is not None:
                mapping[name] = downstream
        return mapping

    # ---- extras -----------------------------------------------------------------
    def describe_neuron(self) -> dict:
        units = self.net.lif_layers()
        return units[0].describe() if units else {}

    def set_spike_counting(self, enabled: bool) -> None:
        self.net.set_spike_counting(enabled)

    def spike_rates(self) -> dict:
        """Activation density per layer -- the fraction of units that emitted anything. See frameworks/relu_activation.py for why this is the fair counterpart to a spike rate."""
        return self.net.spike_rates()
