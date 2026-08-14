"""
Shared helpers and framework-agnostic training utilities used by all SNN
framework modules.

Import pattern in each framework file:
    from learning.utilities import build_optimizer, build_loss, ActivityMonitor
"""
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from typing import Dict, List, Optional
import pynvml


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

    raise NotImplementedError(
        f"loss_fn='{loss_name}' not supported for framework='{framework}'. "
        "Supported: cross_entropy, mse_count."
    )


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

    def firing_rate_tensor(self) -> Optional[torch.Tensor]:
        """GPU-resident equivalent of `firing_rate` — same fraction-of-
        neuron-timesteps-active ratio, but returned as a 0-dim tensor on the
        buffer's own device instead of a Python float. No `.item()`/`.cpu()`
        call happens here, so this is safe to call every batch without
        forcing a CUDA sync; the caller decides when (if ever) to read it
        back to host memory."""
        with self.lock:
            if not self.events:
                return None
            return torch.stack(self.events).float().mean()

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
            layer.register_forward_hook(self.make_hook(name))

    def make_hook(self, name: str):
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


def cv_isi_single_neuron(spike_times: np.ndarray) -> Optional[float]:
    """CV_ISI for one neuron's spike-time index array — SNN_GPU_Evaluation_Metrics.md
    §4.6 reference implementation. None (not 0.0) when fewer than 2 spikes
    exist, since an ISI is undefined for a single spike — this is filtered
    out by the caller rather than silently averaged in as a 0."""
    if len(spike_times) < 2:
        return None
    isi = np.diff(spike_times)
    mean = isi.mean()
    return float(isi.std() / mean) if mean > 0 else None


def compute_cv_isi(activity_snapshot: Dict[str, Optional[torch.Tensor]], sample_idx: int = 0) -> Dict[str, float]:
    """Per-layer + network-wide Coefficient of Variation of Inter-Spike-Interval
    (SNN_GPU_Evaluation_Metrics.md §2.2/§4.6): a per-neuron firing-regularity
    diagnostic, independent of firing rate. Low CV_ISI = clock-like, regular
    firing; near/above 1 = bursty/irregular.

    activity_snapshot: output of ActivityMonitor.recordings(), i.e.
    {layer_name: [T, B, ...] spike tensor}. Only sample `sample_idx` of the
    batch is used (same "sample 0" convention as SNNTrainer.plot_raster) —
    this is a diagnostic snapshot, not a training-time metric, so a single
    representative sample per layer is the intentional scope, not a
    shortcut. Does the CPU/numpy conversion internally; call this only from
    an already-deferred (end-of-epoch/end-of-run) reporting pass, never from
    inside the hot batch loop.

    Returns {layer_name: mean_cv_isi} plus a "network_wide" key averaging
    across all layers that had at least one multi-spike neuron. Layers with
    no qualifying neuron (all silent or all firing exactly once) are
    omitted rather than reported as a misleading 0.0.
    """
    per_layer: Dict[str, float] = {}
    for name, spk in activity_snapshot.items():
        if spk is None:
            continue
        # [T, B, ...] -> sample -> [T, N]
        sample = spk[:, sample_idx] if spk.dim() > 1 else spk
        flat = sample.detach().cpu().numpy().reshape(sample.shape[0], -1)
        cvs = [
            cv_isi_single_neuron(np.nonzero(flat[:, n])[0])
            for n in range(flat.shape[1])
        ]
        cvs = [c for c in cvs if c is not None]
        if cvs:
            per_layer[name] = float(np.mean(cvs))

    if per_layer:
        per_layer["network_wide"] = float(np.mean(list(per_layer.values())))
    return per_layer


def measure_dense_macs(model, sample_batch: torch.Tensor) -> Dict[str, float]:
    """The "measure FLOPs first" step SynOps energy is built on
    (SNN_GPU_Evaluation_Metrics.md §2.4/§4.4): dense (non-sparsity-adjusted)
    multiply-accumulate count for the module immediately downstream of each
    spiking layer named in `model.synops_layer_map()`.

    Runs ONE real forward pass with temporary hooks capturing the exact
    input tensor each downstream module receives, then re-invokes each
    captured (module, input) pair once inside torch.utils.flop_counter.
    FlopCounterMode (ships with torch >=2.1 — no fvcore/ptflops dependency
    needed) to get that single-timestep invocation's dense FLOPs, halved to
    MACs. This is a one-time, static measurement — call it once per model
    (e.g. at the top of SNNTrainer.train()/SNNTester.run()), never per-batch.

    Returns {} if synops_layer_map() is empty (framework doesn't expose
    per-timestep hooks, e.g. Sinabs — matches ActivityMonitor's existing
    no-op convention for that case).
    """
    layer_map = model.synops_layer_map()
    if not layer_map:
        return {}

    captured: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(module, inp, output):
            if name not in captured:
                captured[name] = inp[0].detach().clone()
        return hook

    for name, module in layer_map.items():
        handles.append(module.register_forward_hook(make_hook(name)))

    # eval_mode() here is just to make the probe pass deterministic (no
    # dropout/BN-update side effects); it's not restored afterward since the
    # caller (SNNTrainer.train() / SNNTester.run()) always sets its own
    # correct mode immediately after this one-time measurement anyway.
    model.eval_mode()
    try:
        with torch.no_grad():
            model(sample_batch)
    finally:
        for h in handles:
            h.remove()

    dense_macs: Dict[str, float] = {}
    for name, module in layer_map.items():
        captured_input = captured.get(name)
        if captured_input is None:
            continue
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            module(captured_input)
        dense_macs[name] = fc.get_total_flops() / 2.0

    return dense_macs


def read_gpu_runtime_diagnostics(pipeline_monitor) -> Dict[str, object]:
    """Point-in-time GPU runtime diagnostics beyond what PipelineMonitor
    already tracks (SNN_GPU_Evaluation_Metrics.md new "Runtime GPU diagnostics"
    section): max memory *reserved* by PyTorch's caching allocator (distinct
    from max allocated — the allocator's high-water mark, including memory
    held but not currently in use), whether CUDNN autotune is active, and —
    when NVML is available — GPU temperature and SM/memory clock speed.
    Single-GPU only (device 0) — see event_data_workflow/README.md's
    "Known Limitation" note.

    These are NVML/driver queries, not CUDA-stream operations, so unlike
    `.item()`/`.cpu()` they do NOT force a wait on kernel completion — safe
    to call once per epoch/test-run without reintroducing the sync stalls
    the deferred-logging design is eliminating elsewhere.
    """
    diag: Dict[str, object] = {
        "cudnn_benchmark_enabled": torch.backends.cudnn.benchmark,
        "max_memory_reserved_gb": (
            torch.cuda.max_memory_reserved(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        ),
    }

    nvml_handle = getattr(pipeline_monitor, "nvml_handle", None)
    if nvml_handle is not None and pynvml is not None:
        try:
            diag["gpu_temp_c"]   = pynvml.nvmlDeviceGetTemperature(nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
            diag["sm_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_SM)
            diag["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(nvml_handle, pynvml.NVML_CLOCK_MEM)
        except Exception:
            pass  # driver/permission hiccup — diagnostics are best-effort, never worth failing a run over

    return diag

