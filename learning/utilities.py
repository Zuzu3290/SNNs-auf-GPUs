"""
Shared helpers and framework-agnostic training utilities used by all SNN
framework modules.

Import pattern in each framework file:
    from learning.utilities import build_optimizer, build_loss, ActivityMonitor
"""
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


def build_optimizer(params, fw_cfg: dict) -> torch.optim.Optimizer:
    """
    Factory that reads optimizer name + lr + wd from fw_cfg.

    fw_cfg must contain: optimizer, learning_rate, weight_decay.
    Supported names: adam (default), adamw, sgd.
    """
    lr  = fw_cfg["learning_rate"]
    wd  = fw_cfg["weight_decay"]
    opt = fw_cfg.get("optimizer", "adam").lower()

    if opt == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)


def build_loss(fw_cfg: dict, framework: str = "norse"):
    """
    Factory that reads loss_fn name from fw_cfg and returns a callable.

    Supported loss names:
      cross_entropy   — standard classification loss.
          Norse/SNNTorch: spk_rec is [T, B, C]; sums over T before loss.
          SpikingJelly:   forward already sums T, returns [B, C]; uses nn.CrossEntropyLoss.
      mse_count       — SNNTorch mse_count_loss (requires snntorch installed).
      mse_regression  — flat-vector regression (unused today — the datasets that needed
          it, MVSEC/TUM-VIE, were removed; kept in case a future pose-like target returns).
          Norse/SNNTorch/Sinabs: readout is [T, B, output_dim]; averages over T before loss.
          SpikingJelly: forward already averages T, returns [B, output_dim]; uses nn.MSELoss.
      flow_masked_mse — DSEC dense optical-flow regression (personal/snn_*_regression.py).
          readout: [T, B, 2, H, W] (or [B, 2, H, W] if the framework already reduces T,
          e.g. SpikingJelly) predicting (flow_x, flow_y). targets: [B, H, W, 3] — DSEC's
          own layout, channels (flow_x, flow_y, valid_mask); the third channel is ground
          truth's validity flag, not something the model predicts. Only ~19% of pixels
          are valid in a typical DSEC frame (LiDAR-derived, sparse by nature) — loss is
          masked to those pixels only, not averaged over the whole frame.

    Args:
        fw_cfg    : dict from cfg.FRAMEWORK_CFG[<framework>] merged with lr/wd
        framework : "norse" | "torch" | "spikingjelly" | "sinabs"
    """
    loss_name = fw_cfg.get("loss_fn", "cross_entropy")

    if loss_name == "cross_entropy":
        if framework == "spikingjelly":
            return nn.CrossEntropyLoss()
        return lambda spk_rec, targets: F.cross_entropy(spk_rec.float().sum(0), targets)

    if loss_name == "mse_count":
        from snntorch import functional as SF
        return SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    if loss_name == "mse_regression":
        if framework == "spikingjelly":
            return nn.MSELoss()
        return lambda readout, targets: F.mse_loss(readout.float().mean(0), targets.float())

    if loss_name == "flow_masked_mse":
        return flow_masked_mse

    raise NotImplementedError(
        f"loss_fn='{loss_name}' not supported for framework='{framework}'. "
        "Supported: cross_entropy, mse_count, mse_regression, flow_masked_mse."
    )


def flow_masked_mse(readout: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """readout: [T, B, 2, H, W] or [B, 2, H, W] (predicted flow_x, flow_y).
    targets: [B, H, W, 3] (DSEC layout: flow_x, flow_y, valid_mask).
    MSE over valid pixels only — see build_loss's flow_masked_mse docstring for why."""
    pred = readout.float().mean(0) if readout.dim() == 5 else readout.float()  # -> [B, 2, H, W]
    target_flow = targets[..., :2].float().permute(0, 3, 1, 2)                 # [B, H, W, 2] -> [B, 2, H, W]
    valid = targets[..., 2].float().unsqueeze(1)                               # [B, 1, H, W]

    sq_err = (pred - target_flow) ** 2 * valid
    n_valid_elements = (valid.sum() * pred.shape[1]).clamp_min(1.0)
    return sq_err.sum() / n_valid_elements


class DenseTimestepBuffer:
    """Per-timestep spike buffer for SNN forward passes: push() once per
    timestep, stack() to reconstruct [T, B, ...] for loss/metrics."""

    def __init__(self) -> None:
        self.events: List[torch.Tensor] = []
        self.step_shape: Optional[tuple] = None
        self.lock = threading.Lock()

    def push(self, spk: torch.Tensor) -> None:
        tensor = spk.detach()
        with self.lock:
            if self.step_shape is None:
                self.step_shape = tuple(spk.shape)
            self.events.append(tensor)

    def stack(self) -> Optional[torch.Tensor]:
        with self.lock:
            if not self.events:
                return None
            return torch.stack(self.events)

    def clear(self) -> None:
        with self.lock:
            self.events.clear()
            self.step_shape = None

    @property
    def num_spikes(self) -> int:
        with self.lock:
            return int(sum(e.sum().item() for e in self.events))

    @property
    def num_timesteps(self) -> int:
        with self.lock:
            return len(self.events)

    @property
    def memory_bytes(self) -> int:
        with self.lock:
            return sum(e.element_size() * e.numel() for e in self.events)

    @property
    def firing_rate(self) -> float:
        with self.lock:
            if not self.events or self.step_shape is None:
                return 0.0
            total_per_step = 1
            for d in self.step_shape:
                total_per_step *= d
            total = total_per_step * len(self.events)
            fired = int(sum(e.sum().item() for e in self.events))
            return fired / total if total > 0 else 0.0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()


class ActivityMonitor:
    """
    Per-layer spike recording and activity regularization for hidden LIF
    layers, attached via forward hooks. Owns its buffer/pause state itself
    rather than bolting attributes onto the model instance.

    Attach in __init__ (omit layer_map for a model that can't support
    per-timestep hooks — e.g. Sinabs calls each LIF layer once per forward
    with the whole (B,T,...) tensor rather than once per timestep; see its
    own __init__ for the full reasoning):

        self.activity = ActivityMonitor({'lif1': self.lif1, 'lif2': self.lif2})

    Clear at the start of each forward pass:

        self.activity.clear()

    Pull the regularization penalty in the training loop:

        penalty = model.activity.regularization_loss(min_rate=..., max_rate=..., ...)
    """

    def __init__(self, layer_map: Optional[Dict[str, nn.Module]] = None):
        layer_map = layer_map or {}
        self.buffers: Dict[str, DenseTimestepBuffer] = {name: DenseTimestepBuffer() for name in layer_map}
        self.paused = False
        for name, layer in layer_map.items():
            layer.register_forward_hook(self._make_hook(name))

    def _make_hook(self, name: str):
        def hook(module, inp, output):
            if self.paused:
                return
            spk = output[0] if isinstance(output, tuple) else output
            self.buffers[name].push(spk)
        return hook

    def clear(self) -> None:
        for buf in self.buffers.values():
            buf.clear()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def recordings(self) -> Dict[str, Optional[torch.Tensor]]:
        return {name: buf.stack() for name, buf in self.buffers.items()}

    def regularization_loss(
        self,
        min_rate: float = 0.01,
        max_rate: float = 0.50,
        lambda_low: float = 0.1,
        lambda_high: float = 0.1,
    ) -> torch.Tensor:
        """
        Two-sided per-neuron activity regularization for hidden LIF layers.

        Penalizes dead neurons (rate < min_rate) and saturated neurons
        (rate > max_rate) independently per neuron, so a few overactive
        neurons can't mask a majority of silent ones in a global mean.
        """
        hidden_spikes = self.recordings()
        device = None
        total = None
        n_layers = 0

        for spk in hidden_spikes.values():
            if spk is None:
                continue
            if device is None:
                device = spk.device
                total = torch.zeros(1, device=device)

            rate = spk.float().mean(dim=0).mean(dim=0)
            dead_penalty      = torch.mean(F.relu(min_rate - rate) ** 2)
            saturated_penalty = torch.mean(F.relu(rate - max_rate) ** 2)
            total = total + lambda_low * dead_penalty + lambda_high * saturated_penalty
            n_layers += 1

        if total is None or n_layers == 0:
            return torch.tensor(0.0)
        return total / n_layers

