"""The spiking convolutional network. ONE class, shared by all four frameworks.

Ported from the SNNs_2 comparison pipeline, and the whole point of it: only `make_lif`
changes between frameworks. Every Conv2d, MaxPool2d and Linear here is plain torch.nn
and is therefore literally the same code in all four runs, which makes "identical
architecture, only the framework changed" true BY CONSTRUCTION rather than by careful
copying of four separate definitions that can drift apart.

Architecture: 12C5 - MP2 - 32C5 - MP2 - FC, from the snnTorch/Tonic N-MNIST tutorial --
a cited design rather than one invented here.

Channel counts, kernel sizes, pool size and sensor shape all come from
network_architecture.yaml via Settings, so this pipeline's existing per-dataset shaping
(apply_dataset_shape) keeps working unchanged. Nothing about the architecture is
hardcoded here.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from frameworks.adapters.base import BaseLIF

# The layer slot names the rest of this pipeline already uses: ActivityMonitor's keys,
# synops_layer_map's keys, and the per-layer neuron picker in network_architecture.yaml.
LIF_SLOTS = ("lif1", "lif2", "lif_out")


class FlattenSizeMismatch(Exception):
    """cfg.FC_IN and the measured flatten size disagree."""


def build_layers(make_lif: Callable[[str], BaseLIF], cfg) -> list[nn.Module]:
    """Every layer, in order, one line each.

    Shapes on N-MNIST (34x34, 2 channels), per timestep:

        input                    2 x 34 x 34
        Conv2d(2->12, k5)       12 x 30 x 30     34 - 5 + 1 = 30
        LIF   (lif1)            12 x 30 x 30
        MaxPool2d(2)            12 x 15 x 15     30 / 2 = 15
        Conv2d(12->32, k5)      32 x 11 x 11     15 - 5 + 1 = 11
        LIF   (lif2)            32 x 11 x 11
        MaxPool2d(2)            32 x  5 x  5     11 / 2 = 5, rounded down
        Flatten                      800         32 * 5 * 5
        Linear(800 -> classes)
        LIF   (lif_out)

    Pooling comes AFTER the LIF, so it pools on spikes rather than on membrane voltages.

    `make_lif` takes the slot name so the per-layer neuron picker in
    network_architecture.yaml is consulted once per slot -- see frameworks/adapters.
    """
    features: list[nn.Module] = [
        nn.Conv2d(cfg.IN_CHANNELS, cfg.CONV1_OUT, kernel_size=cfg.CONV1_KERNEL),
        make_lif("lif1"),
        nn.MaxPool2d(cfg.POOL_KERNEL),
        nn.Conv2d(cfg.CONV1_OUT, cfg.CONV2_OUT, kernel_size=cfg.CONV2_KERNEL),
        make_lif("lif2"),
        nn.MaxPool2d(cfg.POOL_KERNEL),
        nn.Flatten(),
    ]
    flat = measure_flat_features(features, cfg)
    check_flat_features(flat, cfg)
    return features + [nn.Linear(flat, cfg.NUM_CLASSES), make_lif("lif_out")]


def measure_flat_features(layers: list[nn.Module], cfg) -> int:
    """Push one empty frame through the real layers and see how wide the output is.

    A batch of 1 zeros is enough -- only shapes matter, and LIF layers preserve shape
    whatever the values. State is reset afterwards so the probe leaves nothing behind.

    This is the AUTHORITY for the Linear's input width. Settings.compute_fc_in()
    computes the same number by formula, ahead of the model, so cfg.display() can print
    it; check_flat_features() below asserts the two agree.
    """
    probe = torch.zeros(1, cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W)
    with torch.no_grad():
        for layer in layers:
            probe = layer(probe)
    for layer in layers:
        if isinstance(layer, BaseLIF):
            layer.reset()
    return probe.shape[1]


def check_flat_features(measured: int, cfg) -> None:
    """Assert the measured flatten size matches Settings.compute_fc_in()'s arithmetic.

    Two sources of truth for one number is normally a smell. Here it is deliberate, and
    it earns its keep by being checked: the formula runs before any layer object exists
    (so cfg.display() can report FC_IN up front), the probe runs on the layers actually
    built, and a disagreement means the formula no longer models the architecture.

    Why this matters even though the formula is correct for every dataset in the
    registry: it goes SILENTLY wrong rather than failing. At a 10x10 sensor the second
    stage computes (3 - 5 + 1) // 2, and Python floor-divides -1 // 2 to -1, so
    32 * -1 * -1 = 32 -- a positive, plausible number for an architecture that cannot
    be built. The probe raises at the conv that does not fit. Measured agreement at
    34x34 (800), 128x128 (26,912) and 180x240 (76,608); divergence at 10x10 and 8x8.
    """
    expected = getattr(cfg, "FC_IN", None)
    if expected is None or int(expected) == int(measured):
        return
    raise FlattenSizeMismatch(
        f"flatten size disagreement: Settings.compute_fc_in() says {expected}, the "
        f"dummy forward pass measured {measured}.\n"
        f"  sensor {cfg.SENSOR_H}x{cfg.SENSOR_W}, in_channels {cfg.IN_CHANNELS}, "
        f"conv1 {cfg.CONV1_OUT}@k{cfg.CONV1_KERNEL}, conv2 {cfg.CONV2_OUT}@"
        f"k{cfg.CONV2_KERNEL}, pool {cfg.POOL_KERNEL}\n"
        f"  The measured value is the real one. compute_fc_in() assumes exactly two "
        f"conv+pool stages with stride 1 and no padding -- if the layer list in "
        f"build_layers() has changed, that formula needs to change with it."
    )


class SpikingNet(nn.Module):
    """Runs the layer stack over T timesteps.

    Input  [T, batch, channels, height, width]
    Output [T, batch, num_classes] -- the full spike stack, NOT summed.

    Returning the stack rather than the sum is what lets every framework share one loss
    path and one spike-rate metric: previously SpikingJelly's forward pre-summed over T
    and returned [B, C] while the others returned [T, B, C], so the loss had to differ
    per framework and the spike-rate figure read roughly T times high for SpikingJelly.
    """

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def lif_layers(self) -> list[BaseLIF]:
        return [layer for layer in self.layers if isinstance(layer, BaseLIF)]

    def named_lif_layers(self) -> dict[str, BaseLIF]:
        """Slot name -> layer, using the names the rest of this pipeline expects
        (ActivityMonitor keys, synops_layer_map): lif1, lif2, lif_out."""
        lifs = self.lif_layers()
        return {
            LIF_SLOTS[i] if i < len(LIF_SLOTS) else f"lif{i + 1}": layer
            for i, layer in enumerate(lifs)
        }

    def dense_after(self, lif_name: str) -> nn.Module | None:
        """The first dense (Conv2d/Linear) module DOWNSTREAM of a given LIF slot.

        Used for the SynOps estimate, whose per-layer term is
        firing_rate(layer) * dense_MACs(next dense module) * T. Derived from the layer
        list rather than hand-written per framework, so it cannot fall out of sync with
        the architecture.
        """
        target = self.named_lif_layers().get(lif_name)
        if target is None:
            return None
        seen = False
        for layer in self.layers:
            if layer is target:
                seen = True
                continue
            if seen and isinstance(layer, (nn.Conv2d, nn.Linear)):
                return layer
        return None

    def reset(self) -> None:
        """Clear every neuron's state, whatever framework it came from."""
        for layer in self.lif_layers():
            layer.reset()

    def set_spike_counting(self, enabled: bool) -> None:
        for layer in self.lif_layers():
            layer.count_spikes = enabled
            if enabled:
                layer.reset_spike_stats()

    def spike_rates(self) -> dict[str, float]:
        """Mean spikes per neuron per timestep, per layer. A true fraction, directly
        comparable across frameworks."""
        return {
            name: layer.spike_rate()
            for name, layer in self.named_lif_layers().items()
        }

    def neuron_counts(self, in_channels: int, sensor_h: int, sensor_w: int) -> dict[str, int]:
        """Neurons per spiking layer, per sample -- MEASURED, not derived.

        Slot name -> neuron count, e.g. {"lif1": 10800, "lif2": 3872, "lif_out": 10}.

        Measured off a real forward pass for the same reason measure_flat_features() is:
        the layer list is the AUTHORITY, and a formula that models the architecture can
        silently stop matching it. This project has already been bitten by exactly that
        -- see check_flat_features() for the case where compute_fc_in() returns a
        plausible positive number for an architecture that cannot be built.

        Counts every BaseLIF slot present, so an FC hidden layer added later is included
        automatically without this function needing to know it exists.

        WHY THIS EXISTS AT ALL: neuron count is the size axis of the scalability study.
        Neurons scale roughly LINEARLY with conv width while parameters scale roughly
        QUADRATICALLY, so `trainable_params` alone cannot place a run on a size ladder --
        the two counts disagree about how much bigger a network got.

        BaseLIF.neurons() cannot serve here: it reads spike_shape, which is only
        populated while count_spikes is on, and spike counting is off in the normal run
        path (it costs an extra GPU kernel per layer per timestep, and this project
        measures wall-clock time).

        Cost is one no-grad forward at T=1, batch 1. LIF layers preserve shape, so a
        single timestep is enough to learn every layer's per-sample shape. Neuron state
        is reset afterwards, so the probe leaves nothing behind.
        """
        named = self.named_lif_layers()
        shapes: dict[str, tuple[int, ...]] = {}
        handles = []

        def make_hook(name: str):
            def hook(module, inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                shapes.setdefault(name, tuple(tensor.shape[1:]))  # drop the batch dim
            return hook

        for name, layer in named.items():
            handles.append(layer.register_forward_hook(make_hook(name)))

        device = next(self.parameters()).device
        probe = torch.zeros(1, 1, in_channels, sensor_h, sensor_w, device=device)
        try:
            with torch.no_grad():
                self.forward(probe)
        finally:
            for handle in handles:
                handle.remove()
            self.reset()

        counts: dict[str, int] = {}
        for name in named:
            shape = shapes.get(name)
            if shape is None:
                continue
            count = 1
            for dimension in shape:
                count *= int(dimension)
            counts[name] = count
        return counts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"expected [T, batch, C, H, W], got shape {tuple(x.shape)}"
            )

        # Reset HERE, not in the training loop. Stateful neurons leaking across samples
        # is the classic silent bug in SNN code; doing it here means it cannot be
        # forgotten at a call site.
        self.reset()

        out_steps = []
        for step in range(x.shape[0]):
            out = x[step]
            for layer in self.layers:
                out = layer(out)
            out_steps.append(out)
        return torch.stack(out_steps)


def build_network(make_lif: Callable[[str], BaseLIF], cfg) -> SpikingNet:
    """Build the network.

    Seeding is the CALLER's job (skeleton.seeding.seed_model_init immediately before
    this), so weight init depends only on the seed and every framework starts from
    byte-identical conv/linear weights.
    """
    return SpikingNet(build_layers(make_lif, cfg))
