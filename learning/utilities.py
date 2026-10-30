"""
Shared helpers and framework-agnostic training utilities used by all SNN
framework modules.

Import pattern in each framework file:
    from learning.utilities import build_optimizer, build_loss, ActivityMonitor
"""
import gc
import logging
import threading
import time
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from typing import Dict, List

logger = logging.getLogger(__name__)


OPTIMIZERS = ("nadam", "adam", "adamw", "sgd")


def build_optimizer(params, cfg) -> torch.optim.Optimizer:
    """The ONE optimizer, shared by every framework.

    Reads training.optimizer.{type,lr,weight_decay,momentum} from the merged config.
    Experiment 2 shares one optimizer across frameworks so the neuron is the only
    variable; experiment 3 overrides it per framework, because an author publishes a
    learning rate along with a neuron and the two are not separable in practice.

    An unrecognised name RAISES. This previously fell through to Adam for anything it
    did not recognise, so a typo produced a real run with the wrong optimizer and no
    warning anywhere in the log.
    """
    name = str(getattr(cfg, "OPTIMIZER", "nadam")).lower()
    lr = cfg.LEARNING_RATE
    wd = cfg.WEIGHT_DECAY

    if name not in OPTIMIZERS:
        raise ValueError(
            f"training.optimizer.type = {name!r} is not supported. "
            f"Supported: {sorted(OPTIMIZERS)}."
        )
    if name == "nadam":
        return torch.optim.NAdam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        # Momentum was hardcoded at 0.9 here; it is a config value now.
        return torch.optim.SGD(
            params, lr=lr, momentum=getattr(cfg, "SGD_MOMENTUM", 0.0), weight_decay=wd
        )
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)


def collect_single_samples(loader, device, count: int) -> list:
    """`count` individual samples, on-device, each shaped [T, 1, C, H, W].

    Pre-loaded so the latency measurement below times the NETWORK, not the data
    pipeline. The host-to-device copy is excluded for the same reason, and it is
    identical for all four frameworks anyway.
    """
    samples: list = []
    for frames, _ in loader:
        frames = frames.to(device)
        for index in range(frames.shape[1]):          # dim 1 = batch
            samples.append(frames[:, index: index + 1].contiguous())
            if len(samples) >= count:
                return samples
    return samples


def measure_latency(model, samples: list, device: torch.device, warmup: int = 5) -> dict:
    """Single-stream latency: batch size 1, synchronised around EACH sample.

    This is the MLPerf Single-Stream convention and the question a deployment actually
    asks -- "one event arrives, how long until the answer is ready?"

    It is NOT the same measurement as dividing a batch's wall-clock by the batch size.
    That figure is throughput under batching: it benefits from parallelism a single
    arriving event cannot use, so it is systematically optimistic, and percentiles built
    from it describe batch-to-batch variation rather than sample-to-sample -- every
    sample in a batch is assigned the same divided value. Both are recorded, under
    names that say which is which (see SNN_GPU_Evaluation_Metrics.md 2.3).

    Per-sample synchronisation IS the point here, unlike in bulk timing: latency is
    defined as when the output is genuinely ready, not when the work was queued.
    """
    import statistics

    if not samples:
        raise ValueError("measure_latency needs at least one sample")

    model.eval_mode()
    def forward(sample):
        return model(sample)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):          # kernels compiled before the clock starts
            forward(samples[0])
        sync()

        durations_ms: list[float] = []
        for sample in samples:
            sync()
            start = time.perf_counter()
            forward(sample)
            sync()
            durations_ms.append(1000.0 * (time.perf_counter() - start))

    ordered = sorted(durations_ms)
    p90_index = min(len(ordered) - 1, int(round(0.90 * (len(ordered) - 1))))
    return {
        "latency_ms": statistics.median(durations_ms),      # headline
        "latency_p90_ms": ordered[p90_index],               # MLPerf convention
        "latency_min_ms": ordered[0],
        "latency_max_ms": ordered[-1],
        "latency_samples": len(durations_ms),
    }


def warm_up(model, sample_batch: torch.Tensor, iterations: int) -> dict:
    """Run untimed forward+backward passes so the timed epochs measure steady state.

    The first CUDA kernel launch pays compilation, cuDNN algorithm selection and
    allocator pool growth. Left inside the timed region that one-off cost is charged to
    training -- and in a multi-framework run only to whichever framework happens to go
    FIRST. Measured on this pipeline before this existed: the first framework reported
    174 ms forward latency against 41-80 ms for the other three, which was cache and
    kernel warm-up, not the framework.

    Backward as well as forward, because backward kernels need compiling too.

    NO optimizer.step() is called and gradients are cleared afterwards, so the weights
    are untouched and the timed epochs begin exactly where they would have. The caller
    is expected to verify that with a weight fingerprint -- `weights_unchanged` in the
    returned dict is that check, done here so every call site gets it.

    The same batch is reused rather than consuming the epoch's data.
    """
    from skeleton.seeding import shared_weight_fingerprint

    report = {"iterations": 0, "weights_unchanged": True, "fingerprint": None}
    if iterations <= 0:
        return report

    before = shared_weight_fingerprint(model)
    model.train_mode()
    # Recordings from warm-up must not reach the metrics. forward() clears at entry, so
    # the real first batch overwrites them anyway -- pausing makes that independent of
    # call order rather than a coincidence.
    model.activity.pause()
    for _ in range(iterations):
        # A plain sum stands in for the loss: warm-up only needs the same kernels compiled and allocations made.
        model(sample_batch).float().sum().backward()
    model.activity.resume()
    model.activity.clear()
    model.zero_grad()

    after = shared_weight_fingerprint(model)
    report.update(iterations=iterations, weights_unchanged=(before == after), fingerprint=after)
    return report


def sum_over_time_cross_entropy(spk_rec: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy over spike counts summed across the time dimension
    (dim 0): spk_rec is [T, B, C], as returned by Norse/SNNTorch/SNN_LIF."""
    return F.cross_entropy(spk_rec.float().sum(0), targets)


LOSSES = ("cross_entropy", "mse_count")


def build_loss(cfg):
    """The ONE loss, shared by every framework.

    There used to be three paths here, branching on which framework was running:
    SpikingJelly's forward pre-summed over T and returned [B, C] while the others
    returned [T, B, C], so SpikingJelly needed nn.CrossEntropyLoss and the rest needed
    sum-over-T. The shared network returns [T, B, C] for all four, so one path covers
    everything and the `framework` argument is gone.

      cross_entropy — torch's own loss on spike counts summed over T. Belongs to none
                      of the four libraries, which is what makes it neutral.
      mse_count     — snnTorch's functional.mse_count_loss. Available, but it belongs
                      to ONE of the four frameworks: do not use it for a
                      cross-framework comparison run.
    """
    loss_name = str(getattr(cfg, "LOSS_FN", "cross_entropy"))

    if loss_name == "cross_entropy":
        return sum_over_time_cross_entropy

    if loss_name == "mse_count":
        from snntorch import functional as SF
        logger.warning(
            "[LOSS] training.loss = 'mse_count' is snnTorch's own loss function. It is "
            "not neutral across frameworks -- do not use it for a comparison run."
        )
        return SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    raise ValueError(
        f"training.loss = {loss_name!r} is not supported. Supported: {sorted(LOSSES)}."
    )


class DenseTimestepBuffer:
    """Per-timestep spike buffer for SNN forward passes: push() once per
    timestep, stack() to reconstruct [T, B, ...] for loss/metrics."""

    def __init__(self, sequence: bool = False) -> None:
        self.events: List[torch.Tensor] = []
        self.lock = threading.Lock()
        # A layer driven in multi-step mode is called ONCE with the whole [T, batch, ...]
        # stack instead of T times with [batch, ...]. Stacking those pushes would give
        # [1, T, batch, ...] -- a rank every metric below reads wrongly, and wrongly
        # without failing. Concatenating along time is the correct join for that shape,
        # and it stays correct if a pass ever pushes more than once.
        self.sequence = sequence

    def push(self, spk: torch.Tensor, keep_graph: bool = False) -> None:
        tensor = spk if keep_graph else spk.detach()  # the activity penalty needs the graph; every metric reads a detached copy
        with self.lock:
            self.events.append(tensor)

    def join(self):
        if not self.events:
            return None
        return torch.cat(self.events, dim=0) if self.sequence else torch.stack(self.events)

    def stack(self):
        with self.lock:
            return self.join()

    def clear(self) -> None:
        with self.lock:
            self.events.clear()

    def firing_rate_tensor(self):
        """GPU-resident fraction-of-neuron-timesteps-active ratio, returned
        as a 0-dim tensor on the buffer's own device instead of a Python
        float. No `.item()`/`.cpu()` call happens here, so this is safe to
        call every batch without forcing a CUDA sync; the caller decides
        when (if ever) to read it back to host memory."""
        with self.lock:
            joined = self.join()
            return None if joined is None else joined.float().mean()


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

    def __init__(self, layer_map=None):
        layer_map = layer_map or {}
        # The buffer is told the layer's calling convention rather than guessing from
        # the tensor it receives: a [T, batch, C, H, W] push and a [batch, C, H, W] push
        # from a 5-D-input layer are the same rank, so shape alone cannot distinguish
        # them. The layer already declares it -- see BaseLIF.consumes_sequence.
        self.buffers: Dict[str, DenseTimestepBuffer] = {
            name: DenseTimestepBuffer(sequence=layer.consumes_sequence())
            for name, layer in layer_map.items()
        }
        self.paused = False
        self.keep_graph = False  # True only while activity regularization trains through the recordings
        for name, layer in layer_map.items():
            layer.register_forward_hook(self.make_hook(name))

    def make_hook(self, name: str):
        def hook(module, inp, output):
            if self.paused:
                return
            spk = output[0] if isinstance(output, tuple) else output
            self.buffers[name].push(spk, keep_graph=self.keep_graph)
        return hook

    def clear(self) -> None:
        for buf in self.buffers.values():
            buf.clear()

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def recordings(self):
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
    with torch.no_grad():
        model(sample_batch)
    for h in handles:
        h.remove()

    # PER TIMESTEP, always -- the caller multiplies by T (inference.py: rate * macs * T),
    # so a figure that already covered T would be counted twice.
    #
    # Whether it does depends on how the network drives its neurons, which is invisible
    # here: a per-timestep network hands each Conv2d [batch, C, H, W], but a multi-step
    # one folds time into the batch and hands it [T*batch, C, H, W] (see
    # spiking_net.run_sequence). The captured tensor looks perfectly ordinary in both
    # cases -- only its leading dimension differs -- so the fold is detected by comparing
    # it against the batch dimension of the real input rather than assumed either way.
    #
    # MEASURED: without this, SpikingJelly's multi-step + cupy arm recorded a SynOps
    # energy of 1.91e9 pJ/sample against the single-step control's 1.21e8 -- a ratio of
    # 15.8 on a T=16 run. The two are verified to emit bit-identical spikes, so the
    # entire difference was this double count.
    batch = sample_batch.shape[1] if sample_batch.dim() == 5 else sample_batch.shape[0]

    dense_macs: Dict[str, float] = {}
    for name, module in layer_map.items():
        captured_input = captured.get(name)
        if captured_input is None:
            continue
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            module(captured_input)
        fold = max(1, round(captured_input.shape[0] / batch)) if batch else 1
        dense_macs[name] = fc.get_total_flops() / 2.0 / fold

    return dense_macs


def safe_empty_cache() -> None:
    """gc.collect() then torch.cuda.empty_cache()."""
    gc.collect()
    torch.cuda.empty_cache()


def measure_batch_vram(model_cls, cfg, device, batch_size: int, timesteps: int):
    """One real forward+backward pass at batch_size, on synthetic data at
    the exact target shape, with AMP autocast+GradScaler (cfg.USE_AMP) and
    gradient accumulation (cfg.GRAD_ACCUM_STEPS micro-batches, peak read
    after the last one) -- mirrors SNNTrainer's own use_amp/grad_accum_steps
    setup. Returns peak VRAM in GB, or None on OOM.

    timesteps must be the real per-sample frame count (cfg.N_TIME_BINS,
    from data_workflow.yaml) -- the actual BPTT unroll length, not a separately
    configured constant.

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
            data = torch.rand(timesteps, batch_size, cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W, device=device)
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



def calibrate_batch_size(model_cls, cfg, device, timesteps: int, *, data_vram_fraction: float,
                          band_min: float,
                          max_batch_size: int = 256, min_batch_size: int = 1,
                          baseline_sensor_px: int = 34 * 34, baseline_batch_size: int = 128) -> int:

    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    vram_budget_gb = total_vram_gb * data_vram_fraction

    baseline_px = cfg.SENSOR_H * cfg.SENSOR_W
    guess = max(min_batch_size, int(baseline_batch_size * (baseline_sensor_px / baseline_px)))
    guess = min(guess, max_batch_size)

    peak_gb = measure_batch_vram(model_cls, cfg, device, guess, timesteps)
    while peak_gb is None and guess > min_batch_size:
        guess //= 2
        peak_gb = measure_batch_vram(model_cls, cfg, device, guess, timesteps)
    if peak_gb is None:
        raise RuntimeError(
            f"[CALIBRATE_BATCH_SIZE] No batch size fits -- even batch_size={min_batch_size} "
            f"OOMs at {cfg.SENSOR_H}x{cfg.SENSOR_W} resolution, T={timesteps}, on this "
            f"{total_vram_gb:.1f} GB GPU. Reduce sensor resolution, timesteps, or model "
            "size, or run on a GPU with more VRAM."
        )

    scaled = max(min_batch_size, min(max_batch_size, int(guess * (vram_budget_gb / peak_gb))))
    if scaled == guess:
        _log_stable_batch_size(cfg, guess, peak_gb, total_vram_gb, band_min, data_vram_fraction)
        return guess  # already at budget, no second probe needed

    scaled_peak_gb = measure_batch_vram(model_cls, cfg, device, scaled, timesteps)
    if scaled_peak_gb is not None:
        _log_stable_batch_size(cfg, scaled, scaled_peak_gb, total_vram_gb, band_min, data_vram_fraction)
        return scaled
    _log_stable_batch_size(cfg, guess, peak_gb, total_vram_gb, band_min, data_vram_fraction)
    return guess  # scaled estimate didn't hold up -- fall back to the already-confirmed value


def _log_stable_batch_size(cfg, batch_size: int, peak_gb: float, total_vram_gb: float,
                            band_min: float, band_max: float) -> None:
    """One-line notification: the stable batch size found for this
    dataset, and whether it landed inside the policy band. Uses print(),
    not logger.info() -- logger.info has no attached handler anywhere in
    this project (configure_logging() in skeleton/snn_logging.py is never
    called), so it would otherwise be silently dropped.

    band_max is the caller's data_vram_fraction, not a separately configured
    value: the search never deliberately lands above its own target, so the
    band's upper edge and the target are the same number by construction.
    """
    dataset = getattr(cfg, "DATASET_NAME", None) or "unknown dataset"
    pct = 100 * peak_gb / total_vram_gb if total_vram_gb else 0.0
    in_band = band_min * 100 <= pct <= band_max * 100
    band_note = "within policy band" if in_band else "OUTSIDE policy band"
    print(f"[CALIBRATE] {dataset}: stable batch_size={batch_size} at {pct:.1f}% VRAM ({band_note})", flush=True)


def measure_activation_density(model, loader, device, max_batches: int = 8) -> dict:
    """Per-layer spike rate on the test set, in its own untimed pass; also fills the layer counters layers.csv reads."""
    net = getattr(model, "net", None) or model
    if not hasattr(net, "set_spike_counting"):
        return {}

    was_training = getattr(model, "training", False)
    model.eval_mode()
    net.set_spike_counting(True)
    with torch.no_grad():
        for index, (data, _) in enumerate(loader):
            if index >= max_batches:
                break
            model(data)
    per_layer = net.spike_rates()
    net.set_spike_counting(False)
    if was_training:
        model.train_mode()

    hidden = {name: rate for name, rate in per_layer.items() if name != "lif_out"}
    # Hidden layers only: the readout's rate is set by the class count, not the representation.
    return {
        "per_layer": per_layer,
        "hidden_mean": (sum(hidden.values()) / len(hidden)) if hidden else None,
        "layers_measured": len(per_layer),
    }
