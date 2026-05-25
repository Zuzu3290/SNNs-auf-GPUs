"""
Activity Regularization and STDP local regularizer — framework-agnostic.

-- Activity Regularization --
Two-sided per-neuron penalty that discourages dead neurons (rate < min_rate)
and saturated neurons (rate > max_rate) in hidden LIF layers.

-- STDP Regularizer --
Spike-Timing Dependent Plasticity implemented as a loss term alongside BPTT.
STDP rule: pre fires BEFORE post → LTP (strengthen); post fires BEFORE pre → LTD (weaken).
Here it is a soft constraint: adds a causal correlation loss to the task loss,
nudging the network toward temporally structured spike patterns without replacing
gradient-based learning.

Attach to any SNN framework with one call:

    from learning.frameworks.activity_reg import register_activity_hooks

    # In __init__:
    register_activity_hooks(self, {'lif1': hidden_layer_1, 'lif2': hidden_layer_2})

    # In forward():
    clear_hidden_spikes(self)
    ...forward pass...

    # In training loop:
    hidden  = get_hidden_spike_recordings(model)
    penalty = activity_regularization(hidden, ...)
    stdp    = stdp_regularization(hidden, output_spikes, ...)
    loss    = task_loss + penalty + stdp

Works with SNNTorch (Alpha/Leaky), Norse (LIFCell), SpikingJelly (LIFNode/IzhikevichNode).
Norse neurons return (spk, state) tuples — handled automatically.
SpikingJelly returns pre-summed [B, C] output — STDP skips the output pair automatically.
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


def stdp_regularization(
    hidden_spikes: Dict[str, Optional[torch.Tensor]],
    output_spikes: Optional[torch.Tensor],
    tau: float    = 20.0,
    A_plus: float  = 0.01,
    A_minus: float = 0.01,
) -> torch.Tensor:
    """
    STDP-inspired causal correlation loss over consecutive layer pairs.

    For each (pre, post) pair in network depth order:
        trace_pre[t]  = decay * trace_pre[t-1]  + mean(pre_spikes[t])
        trace_post[t] = decay * trace_post[t-1] + mean(post_spikes[t])

        LTP += trace_pre[t]  * mean(post_spikes[t])   pre recently active → post fires
        LTD += mean(pre_spikes[t]) * trace_post[t]    post recently active → pre fires

        L_STDP = -A_plus * LTP + A_minus * LTD

    Minimising L_STDP encourages causal (pre → post) spike ordering.
    Gradients flow through all terms back to model weights via BPTT.

    Args:
        hidden_spikes: from get_hidden_spike_recordings() — {name: (T, B, ...)}
        output_spikes: (T, B, C) from model.forward(). Pass None or a [B,C] tensor
                       (SpikingJelly pre-sums) — the output pair is skipped automatically.
        tau:           Trace decay time constant in timesteps.
        A_plus:        LTP strength — scale up to reinforce causal correlations more.
        A_minus:       LTD strength — scale up to penalise anticausal correlations more.

    Returns:
        Scalar STDP loss tensor with gradient graph intact.
    """
    layers = [spk for spk in hidden_spikes.values() if spk is not None]

    # Only append output if it has a time dimension (T, B, C) — SpikingJelly pre-sums to (B, C)
    if output_spikes is not None and output_spikes.dim() == 3:
        layers.append(output_spikes.float())

    if len(layers) < 2:
        device = layers[0].device if layers else torch.device('cpu')
        return torch.zeros(1, device=device)

    decay = torch.tensor(-1.0 / tau).exp()
    total = torch.zeros(1, device=layers[0].device)
    n_pairs = 0

    for i in range(len(layers) - 1):
        pre_spk  = layers[i].float()
        post_spk = layers[i + 1].float()
        T = pre_spk.size(0)

        # Collapse batch and spatial dims to a scalar per timestep
        pre_t  = pre_spk.mean(dim=list(range(1, pre_spk.dim())))   # (T,)
        post_t = post_spk.mean(dim=list(range(1, post_spk.dim()))) # (T,)

        trace_pre  = torch.zeros(1, device=pre_spk.device)
        trace_post = torch.zeros(1, device=pre_spk.device)
        ltp = torch.zeros(1, device=pre_spk.device)
        ltd = torch.zeros(1, device=pre_spk.device)

        for t in range(T):
            trace_pre  = decay * trace_pre  + pre_t[t]
            trace_post = decay * trace_post + post_t[t]

            ltp = ltp + trace_pre  * post_t[t]  # pre was recently active, post fires now
            ltd = ltd + pre_t[t]   * trace_post  # post was recently active, pre fires now

        total = total + (-A_plus * ltp + A_minus * ltd) / T
        n_pairs += 1

    return total / n_pairs
