"""
Activity Regularization — framework-agnostic spike monitor and penalty.

Attach to any SNN framework with one call:

    from learning.frameworks.activity_reg import register_activity_hooks

    # In __init__:
    register_activity_hooks(self, {'lif1': hidden_layer_1, 'lif2': hidden_layer_2})

    # In forward():
    clear_hidden_spikes(self)
    ...forward pass...

    # In training loop (after loss):
    hidden = get_hidden_spike_recordings(model)
    penalty = activity_regularization(hidden, cfg)
    loss = loss + penalty

Works with SNNTorch (Alpha/Leaky), Norse (LIFCell), SpikingJelly (LIFNode/IzhikevichNode).
Norse neurons return (spk, state) tuples — handled automatically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def register_activity_hooks(model: nn.Module, layer_map: Dict[str, nn.Module]) -> None:
    """
    Attach forward hooks to the specified hidden LIF layers.
    Spike tensors from each timestep are appended to model.hidden_spk_buf[name].

    Args:
        model:     SNN model instance (SNN_TORCH, SNN_NORSE, SNN_SJ, or any nn.Module)
        layer_map: {name: module} for each hidden layer to monitor.
                   e.g. {'lif1': self.net[1], 'lif2': self.net[4]}
    """
    model.hidden_spk_buf = {name: [] for name in layer_map}

    def make_hook(name):
        def hook(module, inp, output):
            # Norse LIFCell returns (spk, state) — extract spike tensor only
            spk = output[0] if isinstance(output, tuple) else output
            model.hidden_spk_buf[name].append(spk)
        return hook

    for name, layer in layer_map.items():
        layer.register_forward_hook(make_hook(name))


def clear_hidden_spikes(model: nn.Module) -> None:
    """
    Clear the capture buffers. Call at the start of each forward pass.
    Safe no-op if register_activity_hooks was never called.
    """
    if hasattr(model, 'hidden_spk_buf'):
        for buf in model.hidden_spk_buf.values():
            buf.clear()


def get_hidden_spike_recordings(model: nn.Module) -> Dict[str, Optional[torch.Tensor]]:
    """
    Stack per-timestep spike lists into (T, B, ...) tensors.
    Returns empty dict if hooks were not registered.
    """
    if not hasattr(model, 'hidden_spk_buf'):
        return {}
    return {
        name: torch.stack(buf) if buf else None
        for name, buf in model.hidden_spk_buf.items()
    }


def activity_regularization(
    hidden_spikes: Dict[str, Optional[torch.Tensor]],
    min_rate: float = 0.01,
    max_rate: float = 0.50,
    lambda_low: float = 0.1,
    lambda_high: float = 0.1,
) -> torch.Tensor:
    """
    Two-sided per-neuron activity regularization for hidden LIF layers.

    Penalizes:
      Dead neurons      — mean firing rate < min_rate  (encourages spiking)
      Saturated neurons — mean firing rate > max_rate  (discourages always-on firing)

    Per-neuron rates are computed independently so a few overactive neurons
    cannot mask a majority of silent ones in the global mean.

    Args:
        hidden_spikes: output of get_hidden_spike_recordings()
        min_rate:      minimum acceptable spike probability per timestep (default 1%)
        max_rate:      maximum acceptable spike probability per timestep (default 50%)
        lambda_low:    penalty weight for dead neurons
        lambda_high:   penalty weight for saturated neurons

    Returns:
        Scalar penalty tensor with gradient graph intact.
    """
    device = None
    total = None
    n_layers = 0

    for spk in hidden_spikes.values():
        if spk is None:
            continue

        if device is None:
            device = spk.device
            total = torch.zeros(1, device=device)

        # spk: (T, B, *neuron_dims)
        # Average over T (dim 0) and B (dim 1) → per-neuron firing rate (*neuron_dims,)
        rate = spk.float().mean(dim=0).mean(dim=0)

        dead_penalty      = torch.mean(F.relu(min_rate - rate) ** 2)
        saturated_penalty = torch.mean(F.relu(rate - max_rate) ** 2)

        total = total + lambda_low * dead_penalty + lambda_high * saturated_penalty
        n_layers += 1

    if total is None or n_layers == 0:
        return torch.tensor(0.0)

    return total / n_layers
