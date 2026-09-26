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
def lif_slot_names(cfg, spiking_readout: bool = True) -> list[str]:
    """Every LIF slot this config's architecture asks for, in build order: one lifN per conv block, then lif_fc when a hidden FC is configured, then lif_out unless the readout is linear."""
    names = [f"lif{index}" for index in range(1, len(cfg.CONV_BLOCKS) + 1)]
    names += [f"lif_fc{index}" for index in range(1, len(cfg.FC_HIDDEN) + 1)]
    return names + ["lif_out"] if spiking_readout else names


class FlattenSizeMismatch(Exception):
    """cfg.FC_IN and the measured flatten size disagree."""


def build_layers(make_lif: Callable[[str], BaseLIF], cfg,
                  spiking_readout: bool = True) -> list[nn.Module]:
    """Every layer, in order: one Conv -> LIF -> MaxPool block per cfg.CONV_BLOCKS entry,
    then Flatten, then an optional hidden Linear -> LIF, then the Linear -> LIF output.

    Shapes on N-MNIST (34x34, 2 channels, the two-block default), per timestep:

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

    `convolution.batch_norm` inserts a BatchNorm2d between each conv and its LIF. It is
    what keeps a DEEP stack alive: every LIF layer attenuates, so without normalisation
    the firing rate falls roughly fourfold per layer and a six-layer network emits no
    output spikes at all at initialisation (MEASURED on N-Caltech101: 4.16% -> 1.08% ->
    0.40% -> 0.10% -> 0.00% -> 0.00%; with normalisation 5.51% -> 10.21% -> 14.61% ->
    16.81% -> 6.94% -> 0.63%). BatchNorm2d is plain torch.nn, so it is literally the
    same module in all four frameworks and leaves the neuron -- the thing being compared
    -- untouched.

    `spiking_readout=False` drops the final LIF, leaving the output Linear's continuous
    value as the per-timestep prediction. That is what a REGRESSION head needs: a pose
    or a flow vector is a real number, and a spike count cannot represent one. The conv
    stack above it is unchanged, so a regression run and a classification run share the
    same feature extractor and the same measurements.
    """
    features: list[nn.Module] = []
    in_channels = cfg.IN_CHANNELS
    for index, block in enumerate(cfg.CONV_BLOCKS, start=1):
        features.append(nn.Conv2d(in_channels, block["out"], kernel_size=block["kernel"],
                                   padding=cfg.CONV_PADDING))
        if cfg.BATCH_NORM:
            features.append(nn.BatchNorm2d(block["out"]))
        features.append(name_slot(make_lif(f"lif{index}"), f"lif{index}"))
        features.append(nn.MaxPool2d(cfg.POOL_KERNEL))
        in_channels = block["out"]
    features.append(nn.Flatten())

    flat = measure_flat_features(features, cfg)
    check_flat_features(flat, cfg)

    head: list[nn.Module] = []
    for index, width in enumerate(cfg.FC_HIDDEN, start=1):
        head += [nn.Linear(flat, width), name_slot(make_lif(f"lif_fc{index}"), f"lif_fc{index}")]
        flat = width
    head.append(nn.Linear(flat, cfg.NUM_CLASSES))
    if spiking_readout:
        head.append(name_slot(make_lif("lif_out"), "lif_out"))
    return features + head


def name_slot(layer: BaseLIF, name: str) -> BaseLIF:
    """Tag a LIF with the slot it fills, so named_lif_layers() reports real names at any depth."""
    layer.slot_name = name
    return layer


def measure_flat_features(layers: list[nn.Module], cfg) -> int:
    """Push one empty frame through the real layers and see how wide the output is.

    A batch of 1 zeros is enough -- only shapes matter, and LIF layers preserve shape
    whatever the values. State is reset afterwards so the probe leaves nothing behind.

    This is the AUTHORITY for the Linear's input width. Settings.compute_fc_in()
    computes the same number by formula, ahead of the model, so cfg.display() can print
    it; check_flat_features() below asserts the two agree.
    """
    sequence = any(layer.consumes_sequence() for layer in layers
                   if isinstance(layer, BaseLIF))
    with torch.no_grad():
        if sequence:
            # The neurons are SKIPPED here, unlike the per-timestep path below, and the
            # reason is not cosmetic: this probe runs before the model is moved to the
            # GPU, and SpikingJelly's cupy kernel asserts that every tensor it is handed
            # is already on a CUDA device. Driving it with a CPU probe fails model
            # construction outright. Skipping is sound because a LIF returns spikes
            # shaped exactly like its input -- the line this docstring already relies on
            # -- so the measured width is unchanged either way.
            probe = torch.zeros(1, 1, cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W)
            for layer in layers:
                if isinstance(layer, BaseLIF):
                    continue
                folded = layer(probe.flatten(0, 1))
                probe = folded.reshape(1, 1, *folded.shape[1:])
        else:
            probe = torch.zeros(1, cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W)
            for layer in layers:
                probe = layer(probe)
    for layer in layers:
        if isinstance(layer, BaseLIF):
            layer.reset()
            layer.reset_spike_stats()  # the probe must not be counted as real activity
    return probe.shape[-1]


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
        f"blocks {cfg.CONV_BLOCKS}, pool {cfg.POOL_KERNEL}, padding {cfg.CONV_PADDING}\n"
        f"  The measured value is the real one. compute_fc_in() walks the same block "
        f"list with stride 1 -- if the layer list in build_layers() has changed, that "
        f"formula needs to change with it."
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
        (ActivityMonitor keys, synops_layer_map): lif1..lifN, lif_fc, lif_out."""
        return {
            getattr(layer, "slot_name", f"lif{index + 1}"): layer
            for index, layer in enumerate(self.lif_layers())
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

    def consumes_sequence(self) -> bool:
        """Whether the neurons in this network take the whole stack at once. Asked of the layers rather than configured separately, so the two can never disagree."""
        return any(layer.consumes_sequence() for layer in self.lif_layers())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"expected [T, batch, C, H, W], got shape {tuple(x.shape)}"
            )

        # Reset HERE, not in the training loop. Stateful neurons leaking across samples
        # is the classic silent bug in SNN code; doing it here means it cannot be
        # forgotten at a call site.
        self.reset()

        if self.consumes_sequence():
            return run_sequence(self.layers, x)

        out_steps = []
        for step in range(x.shape[0]):
            out = x[step]
            for layer in self.layers:
                out = layer(out)
            out_steps.append(out)
        return torch.stack(out_steps)


def run_sequence(layers, x: torch.Tensor) -> torch.Tensor:
    """The same layer stack, driven once with the whole [T, batch, ...] sequence.

    Only the neurons are sequence-aware. Conv2d, BatchNorm2d, MaxPool2d, Flatten and
    Linear are plain torch.nn and know nothing about time, so T is folded into the batch
    dimension for them and unfolded afterwards -- which is what SpikingJelly's own
    SeqToANNContainer does, written out here so the shared network keeps working for
    every framework instead of importing one framework's wrapper into the common path.

    Identical to the per-timestep loop for pointwise layers, but NOT for BatchNorm2d:
    folded, it pools statistics over T and batch together rather than per timestep.
    """
    timesteps, batch = x.shape[0], x.shape[1]
    out = x
    for layer in layers:
        if isinstance(layer, BaseLIF):
            out = layer(out)
        else:
            folded = layer(out.flatten(0, 1))
            out = folded.reshape(timesteps, batch, *folded.shape[1:])
    return out


def build_network(make_lif: Callable[[str], BaseLIF], cfg,
                   spiking_readout: bool = True) -> SpikingNet:
    """Build the network.

    Seeding is the CALLER's job (skeleton.seeding.seed_model_init immediately before
    this), so weight init depends only on the seed and every framework starts from
    byte-identical conv/linear weights.
    """
    return SpikingNet(build_layers(make_lif, cfg, spiking_readout=spiking_readout))
