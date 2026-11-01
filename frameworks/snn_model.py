"""The spiking network shared by all four frameworks, and BaseNeuron, the one layer that changes between them.

Only the neuron differs between snnTorch, SpikingJelly, Norse and Sinabs; Conv2d, BatchNorm2d, MaxPool2d and
Linear are plain torch.nn. Each framework's file (snntorch_lif.py, norse_lif.py, ...) defines a class Neuron
that implements BaseNeuron; main.py imports the chosen one and SNNModel builds the network around it: 12C5-MP2-32C5-MP2-FC
(snnTorch/Tonic N-MNIST tutorial) by default, with blocks, kernels, pooling and hidden layers from
network_architecture.yaml. The optimizer and the loss live in the training loop, as in plain PyTorch.
"""
from __future__ import annotations
from abc import abstractmethod
import torch
import torch.nn as nn
from learning.utilities import SpikeRecorder
from skeleton.config import FW_TO_CFG_KEY


class BaseNeuron(nn.Module):
    """One layer of spiking neurons, for one timestep: input current in, spikes of the same shape out, state held inside."""

    def __init__(self) -> None:
        super().__init__()

        self.count_spikes: bool = False
        self.spike_total: torch.Tensor | float = 0.0  
        self.spike_slots: int = 0                     
        self.spike_shape = None                       

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Charge, fire, reset for one timestep. Returns spikes shaped like x."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all neuron state. Called at the start of every batch."""

    def consumes_sequence(self) -> bool:
        """True when the layer takes the whole [T, batch, ...] stack in one call (SpikingJelly step_mode 'm')."""
        return False

    def record(self, spikes: torch.Tensor) -> None:
        """Accumulate spike counts on the device, without a host sync."""
        if not self.count_spikes:
            return
        if self.spike_shape is None:
            leading = 2 if self.consumes_sequence() else 1
            self.spike_shape = tuple(spikes.shape[leading:])
        self.spike_total = self.spike_total + spikes.detach().sum()
        self.spike_slots += spikes.numel()

    def reset_spike_stats(self) -> None:
        self.spike_total = 0.0
        self.spike_slots = 0
        self.spike_shape = None

    def neurons(self) -> int:
        """Neurons in this layer, per sample; 0 until something has been counted."""
        if self.spike_shape is None:
            return 0
        count = 1
        for dimension in self.spike_shape:
            count *= dimension
        return count

    def spike_rate(self) -> float:
        """Mean spikes per neuron per timestep, as a fraction."""
        if self.spike_slots == 0:
            return 0.0
        total = self.spike_total
        if isinstance(total, torch.Tensor):
            total = total.item()
        return float(total) / self.spike_slots


class SNNModel(nn.Module):
    """[T, batch, C, H, W] in, [T, batch, num_classes] output spikes out; seeded first, so all four frameworks start identical."""

    def __init__(self, cfg, neuron_class: type[BaseNeuron]) -> None:
        super().__init__()
        self.cfg = cfg
        self.neuron_layers = {}  
        layers = []

        # Conv -> [BatchNorm] -> neuron -> MaxPool
        in_channels = cfg.IN_CHANNELS
        for index, block in enumerate(cfg.CONV_BLOCKS, start=1):
            layers.append(nn.Conv2d(in_channels, block["out"], kernel_size=block["kernel"], padding=cfg.CONV_PADDING))
            if cfg.BATCH_NORM:
                layers.append(nn.BatchNorm2d(block["out"]))  
            self.neuron_layers[f"lif{index}"] = neuron_class(cfg)
            layers.append(self.neuron_layers[f"lif{index}"])
            layers.append(nn.MaxPool2d(cfg.POOL_KERNEL))     
            in_channels = block["out"]
        layers.append(nn.Flatten())

        
        probe = torch.zeros(1, cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W)
        with torch.no_grad():
            for layer in layers:
                if not (isinstance(layer, BaseNeuron) and layer.consumes_sequence()):
                    probe = layer(probe)
        for neuron in self.neuron_layers.values():
            neuron.reset()
            neuron.reset_spike_stats()
        flat = probe.shape[1]

        for index, width in enumerate(cfg.FC_HIDDEN, start=1):
            layers.append(nn.Linear(flat, width))
            self.neuron_layers[f"lif_fc{index}"] = neuron_class(cfg)
            layers.append(self.neuron_layers[f"lif_fc{index}"])
            flat = width
        layers.append(nn.Linear(flat, cfg.NUM_CLASSES))
        self.neuron_layers["lif_out"] = neuron_class(cfg)
        layers.append(self.neuron_layers["lif_out"])

        self.layers = nn.Sequential(*layers)
        self.to(torch.device(cfg.DEVICE))

        hidden = {name: neuron for name, neuron in self.neuron_layers.items() if name != "lif_out"}
        self.spike_recorder = SpikeRecorder(hidden) if cfg.ACTIVITY_REG_ENABLED else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spike_recorder is not None:
            self.spike_recorder.clear()
        self.reset()

        if any(neuron.consumes_sequence() for neuron in self.neuron_layers.values()):
            timesteps, batch = x.shape[0], x.shape[1]
            out = x
            for layer in self.layers:
                if isinstance(layer, BaseNeuron):
                    out = layer(out)
                else:
                    folded = layer(out.flatten(0, 1))
                    out = folded.reshape(timesteps, batch, *folded.shape[1:])
            return out

        return torch.stack([self.layers(x[step]) for step in range(x.shape[0])])

    def reset(self) -> None:
        """Clear every neuron's state, whatever framework it came from."""
        for neuron in self.neuron_layers.values():
            neuron.reset()

    def set_spike_counting(self, enabled: bool) -> None:
        for neuron in self.neuron_layers.values():
            neuron.count_spikes = enabled
            if enabled:
                neuron.reset_spike_stats()

    def spike_rates(self) -> dict[str, float]:
        """Mean spikes per neuron per timestep, per layer."""
        return {name: neuron.spike_rate() for name, neuron in self.neuron_layers.items()}

    def synops_layer_map(self) -> dict:
        """Each hidden neuron layer -> the first Conv2d/Linear after it, whose MACs its spikes drive in the SynOps estimate."""
        layers = list(self.layers)
        mapping = {}
        for name, neuron in self.neuron_layers.items():
            if name == "lif_out":
                continue
            after = layers[layers.index(neuron) + 1:]
            dense = next((layer for layer in after if isinstance(layer, (nn.Conv2d, nn.Linear))), None)
            if dense is not None:
                mapping[name] = dense
        return mapping

    def describe_neuron(self) -> dict:
        """The neuron parameters this run was configured with, for the run record."""
        key = FW_TO_CFG_KEY[self.cfg.FRAMEWORK]
        return {"framework": key, **self.cfg.NEURON[key]}
