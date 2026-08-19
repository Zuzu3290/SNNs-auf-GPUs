"""
Shared helpers and framework-agnostic training utilities used by all SNN
framework modules.

Import pattern in each framework file:
    from learning.utilities import build_optimizer, build_loss, ActivityMonitor
"""
import gc
import logging
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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


def sum_over_time_cross_entropy(spk_rec: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy over spike counts summed across the time dimension
    (dim 0): spk_rec is [T, B, C], as returned by Norse/SNNTorch/SNN_LIF."""
    return F.cross_entropy(spk_rec.float().sum(0), targets)


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
        return sum_over_time_cross_entropy

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
        self.lock = threading.Lock()

    def push(self, spk: torch.Tensor) -> None:
        tensor = spk.detach()
        with self.lock:
            self.events.append(tensor)

    def stack(self) -> Optional[torch.Tensor]:
        with self.lock:
            if not self.events:
                return None
            return torch.stack(self.events)

    def clear(self) -> None:
        with self.lock:
            self.events.clear()

    def firing_rate_tensor(self) -> Optional[torch.Tensor]:
        """GPU-resident fraction-of-neuron-timesteps-active ratio, returned
        as a 0-dim tensor on the buffer's own device instead of a Python
        float. No `.item()`/`.cpu()` call happens here, so this is safe to
        call every batch without forcing a CUDA sync; the caller decides
        when (if ever) to read it back to host memory."""
        with self.lock:
            if not self.events:
                return None
            return torch.stack(self.events).float().mean()


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


def safe_empty_cache() -> None:
    """gc.collect() then torch.cuda.empty_cache(), with the cache-clear
    itself guarded against raising."""
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, torch.AcceleratorError):
        pass


def measure_batch_vram(model_cls, cfg, device, batch_size: int) -> Optional[float]:
    """One real forward+backward pass at batch_size, on synthetic data at
    the exact target shape, with AMP autocast+GradScaler (cfg.USE_AMP) and
    gradient accumulation (cfg.GRAD_ACCUM_STEPS micro-batches, peak read
    after the last one) -- mirrors SNNTrainer's own use_amp/grad_accum_steps
    setup. Returns peak VRAM in GB, or None on OOM.

    Catches both torch.cuda.OutOfMemoryError and torch.AcceleratorError --
    sibling exception classes on this PyTorch build, neither a subclass of
    the other, so a real OOM can surface as either one.
    """
    safe_empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model = model_cls(cfg)
        use_amp = getattr(cfg, "USE_AMP", True) and device.type == "cuda"
        grad_accum_steps = max(1, getattr(cfg, "GRAD_ACCUM_STEPS", 1))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        for _ in range(grad_accum_steps):
            data = torch.rand(cfg.TIMESTEPS, batch_size, cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                spk_rec = model(data)
                loss = spk_rec.float().sum()
            scaler.scale(loss).backward()
            del data, spk_rec, loss
        del scaler
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        del model
        safe_empty_cache()
        return peak_gb
    except (torch.cuda.OutOfMemoryError, torch.AcceleratorError):
        safe_empty_cache()
        return None


# The dataset's batch must never be the constraint on how large the model
# itself can grow -- so calibration targets a bounded VRAM fraction, not
# the largest batch size that avoids OOM. One policy, covers both training
# and inference (inference's actual footprint is smaller, so a size that
# fits training fits inference too).
BATCH_VRAM_BAND_MIN = 0.30
BATCH_VRAM_BAND_OPTIMAL = 0.34
BATCH_VRAM_BAND_MAX = 0.35


def calibrate_batch_size(model_cls, cfg, device, data_vram_fraction: float = BATCH_VRAM_BAND_OPTIMAL,
                          max_batch_size: int = 256, min_batch_size: int = 1,
                          baseline_sensor_px: int = 34 * 34, baseline_batch_size: int = 128) -> int:
    """Picks a batch size that fits within data_vram_fraction of total VRAM
    (default: the 30-35% policy band above), instead of the largest one
    that avoids OOM.

    Starting guess: baseline_batch_size scaled by sensor pixel-count ratio
    against baseline_sensor_px. One probe at the guess, then one probe at a
    linearly-scaled target (memory ~ batch size for fixed architecture/T) --
    at most 2 real forward/backward passes total. Falls back to halving on
    OOM, or to the already-confirmed guess if the scaled probe fails.
    """
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    vram_budget_gb = total_vram_gb * data_vram_fraction

    baseline_px = cfg.SENSOR_H * cfg.SENSOR_W
    guess = max(min_batch_size, int(baseline_batch_size * (baseline_sensor_px / baseline_px)))
    guess = min(guess, max_batch_size)

    peak_gb = measure_batch_vram(model_cls, cfg, device, guess)
    while peak_gb is None and guess > min_batch_size:
        guess //= 2
        peak_gb = measure_batch_vram(model_cls, cfg, device, guess)
    if peak_gb is None:
        _log_stable_batch_size(cfg, min_batch_size, 0.0, total_vram_gb)
        return min_batch_size

    scaled = max(min_batch_size, min(max_batch_size, int(guess * (vram_budget_gb / peak_gb))))
    if scaled == guess:
        _log_stable_batch_size(cfg, guess, peak_gb, total_vram_gb)
        return guess  # already at budget, no second probe needed

    scaled_peak_gb = measure_batch_vram(model_cls, cfg, device, scaled)
    if scaled_peak_gb is not None:
        _log_stable_batch_size(cfg, scaled, scaled_peak_gb, total_vram_gb)
        return scaled
    _log_stable_batch_size(cfg, guess, peak_gb, total_vram_gb)
    return guess  # scaled estimate didn't hold up -- fall back to the already-confirmed value


def _log_stable_batch_size(cfg, batch_size: int, peak_gb: float, total_vram_gb: float) -> None:
    """One-line notification: the stable batch size found for this
    dataset, and whether it landed inside the policy band. Uses print(),
    not logger.info() -- logger.info has no attached handler anywhere in
    this project (configure_logging() in skeleton/snn_logging.py is never
    called), so it would otherwise be silently dropped.
    """
    dataset = getattr(cfg, "DATASET_NAME", None) or "unknown dataset"
    pct = 100 * peak_gb / total_vram_gb if total_vram_gb else 0.0
    in_band = BATCH_VRAM_BAND_MIN * 100 <= pct <= BATCH_VRAM_BAND_MAX * 100
    band_note = "within policy band" if in_band else "OUTSIDE policy band"
    print(f"[CALIBRATE] {dataset}: stable batch_size={batch_size} at {pct:.1f}% VRAM ({band_note})", flush=True)


def select_inference_mode() -> bool:
    """Ask whether inference should show a live visualization alongside the usual
    statistical output, or statistics only. No stdin attached (Colab, CI, batch)
    defaults to statistics-only rather than crashing on EOFError.
    Returns True if visualization was requested."""
    print("\n[MAIN] Inference output:")
    print("  1) Statistics only")
    print("  2) Statistics + live visualization (opens a window showing input frames vs. predictions)")
    try:
        choice = input("Enter number [1]: ").strip() or "1"
    except EOFError:
        choice = "1"
    if choice not in ("1", "2"):
        print(f"[MAIN] Invalid selection '{choice}' — defaulting to statistics only")
        choice = "1"
    return choice == "2"
