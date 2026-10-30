"""The one model for all four frameworks: a shared network in which only the LIF neuron changes, chosen by cfg.FRAMEWORK."""
from __future__ import annotations

import torch
import torch.nn as nn
from frameworks.adapters import adapter_class
from frameworks.spiking_net import build_network
from learning.utilities import ActivityMonitor, build_loss, build_optimizer
from skeleton.config import Settings


class SNNModel(nn.Module):
    """The comparison model: shared network, framework-specific neuron, one optimizer and one loss."""

    def __init__(self, cfg: Settings, framework=None):
        """`framework` overrides cfg.FRAMEWORK and is written back, so the display and results report the right one."""
        super().__init__()
        if framework is not None:
            cfg.FRAMEWORK = framework
        self.cfg = cfg
        self.framework = cfg.FRAMEWORK
        self.device = torch.device(cfg.DEVICE)

        lif_class = adapter_class(self.framework)
        self.net = build_network(lambda: lif_class(cfg), cfg)
        self.to(self.device)

        self.optimizer = build_optimizer(self.parameters(), cfg)
        self.loss_fn = build_loss(cfg)

        # Every hidden LIF is hooked for SynOps; the readout's spikes are counted by the tester instead.
        named = self.net.named_lif_layers()
        self.activity = ActivityMonitor({name: layer for name, layer in named.items() if name != "lif_out"})

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data [T, B, C, H, W] -> output spikes [T, B, num_classes]."""
        self.activity.clear()
        return self.net(data)

    def backward_pass(self, loss: torch.Tensor, scaler=None, do_step: bool = True) -> None:
        """Backward, then an optimizer step unless gradients are still being accumulated."""
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
        """Checkpoint payload; neuron state is reset first because sinabs keeps it in batch-shaped buffers."""
        self.net.reset()
        return {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "framework": self.framework,
            "neuron": self.describe_neuron(),
        }

    def load_state(self, state: dict) -> None:
        """Restore model and optimizer from a get_state() payload."""
        self.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

    def synops_layer_map(self) -> dict:
        """Each hidden LIF slot -> the dense layer its spikes feed, for the SynOps estimate."""
        mapping = {}
        for name in self.net.named_lif_layers():
            if name == "lif_out":
                continue
            downstream = self.net.dense_after(name)
            if downstream is not None:
                mapping[name] = downstream
        return mapping

    def describe_neuron(self) -> dict:
        """The neuron parameters actually built, read off the live layer for the run record."""
        lifs = self.net.lif_layers()
        return lifs[0].describe() if lifs else {}

    def set_spike_counting(self, enabled: bool) -> None:
        self.net.set_spike_counting(enabled)

    def spike_rates(self) -> dict:
        return self.net.spike_rates()
