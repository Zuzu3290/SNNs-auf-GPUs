"""ONE model class for all four frameworks.

Replaces SNN_TORCH / SNN_NORSE / SNN_SJ / SNN_SINABS, which were four hand-written
networks that happened to agree. Here the network is shared (frameworks/spiking_net.py)
and only the neuron is injected, chosen from cfg.FRAMEWORK.

Everything the old four classes duplicated -- backward_pass, zero_grad, train_mode,
eval_mode, get_lr, get_state -- was already identical in all four, so it lives here once.

Three consequences worth knowing, all improvements that fall out of sharing the network
rather than being separately implemented:

  * tensor_format() is "TB" for EVERY framework now. Sinabs used to need "BT" because
    its LIF layers consume a whole (B,T,...) sequence; the adapter feeds them one
    timestep at a time instead, so the special case is gone and the trainer no longer
    transposes for one framework only.

  * forward() returns [T, B, C] for every framework. SpikingJelly used to pre-sum over
    T and return [B, C], which forced a different loss per framework and made the
    spike-rate metric read roughly T times high for it. One shape means one loss path.

  * ActivityMonitor hooks now work for sinabs. They assume the hooked layer is called
    once per TIMESTEP, which was true for three frameworks and false for sinabs -- so
    sinabs silently got no SynOps, no CV-ISI and, with activity regularisation on, no
    penalty at all while the other three were penalised. Its LIF is now called once per
    timestep like the rest, so it is hooked like the rest.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from frameworks.adapters import lif_factory
from frameworks.model_interface import ModelInterface
from frameworks.spiking_net import build_network
from learning.utilities import ActivityMonitor, build_loss, build_optimizer
from skeleton.snn_config import Settings


class SNNModel(ModelInterface, nn.Module):
    """The comparison model. Which framework it uses comes from cfg.FRAMEWORK."""

    def __init__(self, cfg: Settings, framework: str | None = None):
        """`framework` overrides cfg.FRAMEWORK, and is also written BACK onto cfg.

        The write-back matters: cfg.display(), cfg.active_fw_cfg and the results row all
        read cfg.FRAMEWORK, so leaving it stale would report the wrong framework for the
        run. run_one() in the benchmark already sets it the same way before construction.
        """
        super().__init__()
        if framework is not None:
            cfg.FRAMEWORK = framework
        self.cfg = cfg
        self.framework = cfg.FRAMEWORK
        self.device = torch.device(cfg.DEVICE)

        self.net = build_network(lif_factory(self.framework, cfg), cfg)
        self.to(self.device)

        # ONE optimizer and ONE loss, read straight from Settings -- not from a
        # per-framework block. forward() returns [T, B, C] for every framework, so
        # there is a single correct loss path and no framework argument to pass.
        self.optimizer = build_optimizer(self.parameters(), cfg)
        self.loss_fn = build_loss(cfg)

        named = self.net.named_lif_layers()
        self.activity = ActivityMonitor(named)

    # ---- ModelInterface ---------------------------------------------------------
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """data: [T, B, C, H, W]  ->  [T, B, num_classes]"""
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
        """Checkpoint payload.

        The network is reset FIRST, so transient neuron state never reaches the
        checkpoint. This is not tidiness -- sinabs registers `v_mem` (and `i_syn`) as
        BUFFERS, so a charged membrane lands in state_dict() with whatever shape the
        last batch had. Loading that into a freshly built model, whose buffers are
        still zero-size, fails outright with a shape mismatch, which would only be
        discovered when someone tried to resume a sinabs run.

        Resetting here is safe and loses nothing: SpikingNet.forward() resets every
        neuron at the start of every pass anyway, so the saved state was never going
        to be read back.
        """
        self.net.reset()
        return {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "framework": self.framework,
            "neuron": self.describe_neuron(),
        }

    def reset_state(self) -> None:
        self.net.reset()

    def synops_layer_map(self) -> dict:
        """Derived from the layer list rather than hand-written per framework, so it
        cannot fall out of sync with the architecture. Every named LIF layer is
        considered; a layer with no downstream dense module (lif_out, since it is the
        last layer) is correctly absent -- dense_after() returning None is what excludes
        it, not the iteration skipping it."""
        mapping = {}
        for name in self.net.named_lif_layers():
            downstream = self.net.dense_after(name)
            if downstream is not None:
                mapping[name] = downstream
        return mapping

    # ---- extras -----------------------------------------------------------------
    def describe_neuron(self) -> dict:
        """What the neuron actually is, for the run record -- read off the live layer
        rather than re-read from the config, so it reports what was built."""
        lifs = self.net.lif_layers()
        return lifs[0].describe() if lifs else {}

    def set_spike_counting(self, enabled: bool) -> None:
        self.net.set_spike_counting(enabled)

    def spike_rates(self) -> dict:
        return self.net.spike_rates()

    def neuron_counts(self) -> dict:
        """Slot name -> neurons per sample, measured off the live network.

        See SpikingNet.neuron_counts() for why this is measured rather than derived.
        The size axis of the scalability study is the SUM of these.

        Recording is paused around the probe and its prior state restored, so the
        shape probe cannot leave a spurious timestep in the ActivityMonitor's buffers
        for whatever reads them next.
        """
        was_paused = self.activity.paused
        self.activity.pause()
        try:
            return self.net.neuron_counts(
                self.cfg.IN_CHANNELS, self.cfg.SENSOR_H, self.cfg.SENSOR_W
            )
        finally:
            self.activity.paused = was_paused
            self.activity.clear()
