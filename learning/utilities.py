"""Shared helpers for training and testing: loss, timing, SynOps, latency, warm-up, batch-size calibration."""
import gc
import logging
import statistics
import time
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from typing import Dict

logger = logging.getLogger(__name__)

# A synaptic operation is one addition, 0.9 pJ at 45 nm (Horowitz, ISSCC 2014)
SYNOPS_ENERGY_PJ = 0.9


@contextmanager
def timed(device: torch.device, event_list: list):
    """Time one operation without stalling the GPU: a CUDA event pair on cuda, a perf_counter pair on cpu, read later by elapsed_ms()."""
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        event_list.append((start, end))
    else:
        t0 = time.perf_counter()
        yield
        event_list.append((t0, time.perf_counter()))


def elapsed_ms(pair) -> float:
    """Milliseconds for one timed() pair; for CUDA events the stream must be synchronised first."""
    a, b = pair
    if isinstance(a, torch.cuda.Event):
        return a.elapsed_time(b)
    return (b - a) * 1000.0


def spike_count_loss(spk_rec: torch.Tensor, targets: torch.Tensor, loss: str) -> torch.Tensor:
    """The loss on the output spikes [T, B, C]: cross_entropy on spike counts summed over time, or snnTorch's mse_count."""
    if loss == "cross_entropy":
        return F.cross_entropy(spk_rec.float().sum(0), targets)
    if loss == "mse_count":
        from snntorch import functional as SF
        return SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)(spk_rec, targets)
    raise ValueError(f"training.loss = {loss!r} is not supported; use cross_entropy or mse_count.")


def synops_estimate(model, dense_macs_per_layer: dict, timesteps: int, device: torch.device) -> torch.Tensor:
    """SynOps of the last counted pass: each hidden layer's firing rate (its own spike counter) x the dense MACs it drives x T; stays on the device."""
    layers = model.neuron_layers
    synops = torch.zeros((), device=device)
    for name, macs in dense_macs_per_layer.items():
        layer = layers[name]
        if layer.spike_slots:
            synops = synops + layer.spike_total / layer.spike_slots * macs * timesteps
    return synops


class SpikeRecorder:
    """Keeps every hidden layer's spikes from the last forward pass, graph included, for the activity penalty."""

    def __init__(self, layers: dict):
        self.spikes = {name: [] for name in layers}
        self.sequence = {name: layer.consumes_sequence() for name, layer in layers.items()}  # one [T, B, ...] push instead of T pushes
        for name, layer in layers.items():
            layer.register_forward_hook(self.make_hook(name))

    def make_hook(self, name: str):
        def hook(module, inp, output):
            self.spikes[name].append(output)
        return hook

    def clear(self) -> None:
        for recorded in self.spikes.values():
            recorded.clear()

    def penalty(self, min_rate: float, max_rate: float, lambda_low: float, lambda_high: float) -> torch.Tensor:
        """Per-neuron penalty on dead (rate < min_rate) and saturated (rate > max_rate) hidden neurons, averaged over layers."""
        total, n_layers = None, 0
        for name, recorded in self.spikes.items():
            if not recorded:
                continue
            spk = torch.cat(recorded) if self.sequence[name] else torch.stack(recorded)
            rate = spk.float().mean(dim=0).mean(dim=0)
            term = (lambda_low * torch.mean(F.relu(min_rate - rate) ** 2)
                    + lambda_high * torch.mean(F.relu(rate - max_rate) ** 2))
            total = term if total is None else total + term
            n_layers += 1
        return torch.tensor(0.0) if total is None else total / n_layers


def measure_latency(model, loader, device: torch.device, count: int, warmup: int = 5) -> dict:
    """Batch-size-1 latency over `count` test samples, each timed alone and synchronised: how long until one answer is ready (MLPerf Single-Stream)."""
    samples = []
    for frames, _ in loader:
        samples += [frames[:, i:i + 1].to(device).contiguous() for i in range(frames.shape[1])]
        if len(samples) >= count:
            break
    samples = samples[:count]

    model.eval()
    durations_ms = []
    with torch.no_grad():
        for step, sample in enumerate([samples[0]] * warmup + samples):  # the first passes compile kernels and are not counted
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step >= warmup:
                durations_ms.append(1000.0 * (time.perf_counter() - start))

    ordered = sorted(durations_ms)
    return {
        "latency_ms": statistics.median(durations_ms),
        "latency_p90_ms": ordered[min(len(ordered) - 1, round(0.90 * (len(ordered) - 1)))],
        "latency_min_ms": ordered[0],
        "latency_max_ms": ordered[-1],
        "latency_samples": len(durations_ms),
    }


def warm_up(model, sample_batch: torch.Tensor, iterations: int) -> dict:
    """Untimed forward+backward passes on one batch, so kernel compilation is not charged to training; no optimizer step, checked by fingerprint."""
    from skeleton.seeding import shared_weight_fingerprint

    report = {"iterations": 0, "weights_unchanged": True, "fingerprint": None}
    if iterations <= 0:
        return report

    before = shared_weight_fingerprint(model)
    model.train()
    for _ in range(iterations):
        model(sample_batch).float().sum().backward()
    model.zero_grad()

    after = shared_weight_fingerprint(model)
    report.update(iterations=iterations, weights_unchanged=(before == after), fingerprint=after)
    return report


def measure_dense_macs(model, sample_batch: torch.Tensor) -> Dict[str, float]:
    """Dense MACs per timestep of the module downstream of each hidden LIF, measured once with FlopCounterMode; the base of the SynOps estimate."""
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

    model.eval()  # no BatchNorm updates from the probe; the caller sets its own mode afterwards
    with torch.no_grad():
        model(sample_batch)
    for h in handles:
        h.remove()

    # Per timestep: a multi-step network folds T into the batch ([T*B, ...]), so that fold is divided out here.
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


def measure_batch_vram(model_cls, neuron_class, cfg, device, batch_size: int, timesteps: int):
    """Peak VRAM (GB) of one forward+backward at batch_size on synthetic data, with the trainer's AMP and accumulation settings; None on OOM."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model = model_cls(cfg, neuron_class)
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
        gc.collect()
        torch.cuda.empty_cache()
        return peak_gb
    except (torch.cuda.OutOfMemoryError, torch.AcceleratorError):
        gc.collect()
        torch.cuda.empty_cache()
        return None


def calibrate_batch_size(model_cls, neuron_class, cfg, device, timesteps: int, *, data_vram_fraction: float,
                          band_min: float,
                          max_batch_size: int = 256, min_batch_size: int = 1,
                          baseline_sensor_px: int = 34 * 34, baseline_batch_size: int = 128) -> int:
    """Largest batch size whose measured peak VRAM fits data_vram_fraction of the GPU: probe a guess, scale it, confirm."""
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    vram_budget_gb = total_vram_gb * data_vram_fraction

    baseline_px = cfg.SENSOR_H * cfg.SENSOR_W
    guess = max(min_batch_size, int(baseline_batch_size * (baseline_sensor_px / baseline_px)))
    guess = min(guess, max_batch_size)

    peak_gb = measure_batch_vram(model_cls, neuron_class, cfg, device, guess, timesteps)
    while peak_gb is None and guess > min_batch_size:
        guess //= 2
        peak_gb = measure_batch_vram(model_cls, neuron_class, cfg, device, guess, timesteps)
    if peak_gb is None:
        raise RuntimeError(
            f"[CALIBRATE_BATCH_SIZE] No batch size fits -- even batch_size={min_batch_size} "
            f"OOMs at {cfg.SENSOR_H}x{cfg.SENSOR_W} resolution, T={timesteps}, on this "
            f"{total_vram_gb:.1f} GB GPU. Reduce sensor resolution, timesteps, or model "
            "size, or run on a GPU with more VRAM."
        )

    batch_size, batch_peak_gb = guess, peak_gb
    scaled = max(min_batch_size, min(max_batch_size, int(guess * (vram_budget_gb / peak_gb))))
    if scaled != guess:
        scaled_peak_gb = measure_batch_vram(model_cls, neuron_class, cfg, device, scaled, timesteps)
        if scaled_peak_gb is not None:  # otherwise keep the already-confirmed guess
            batch_size, batch_peak_gb = scaled, scaled_peak_gb

    pct = 100 * batch_peak_gb / total_vram_gb
    band = "within" if band_min * 100 <= pct <= data_vram_fraction * 100 else "OUTSIDE"
    logger.info(f"[CALIBRATE] {cfg.DATASET_NAME}: batch_size={batch_size} at {pct:.1f}% VRAM ({band} policy band)")
    return batch_size


def measure_activation_density(model, loader, device, max_batches: int = 8) -> dict:
    """Share of hidden neurons that fire per timestep on the test set, per layer and averaged; its own untimed pass, which also fills layers.csv."""
    was_training = model.training
    model.eval()
    model.set_spike_counting(True)
    with torch.no_grad():
        for index, (data, _) in enumerate(loader):
            if index >= max_batches:
                break
            model(data)
    per_layer = model.spike_rates()
    model.set_spike_counting(False)
    if was_training:
        model.train()

    hidden = {name: rate for name, rate in per_layer.items() if name != "lif_out"}  # the readout's rate follows the class count
    return {
        "per_layer": per_layer,
        "hidden_mean": (sum(hidden.values()) / len(hidden)) if hidden else None,
        "layers_measured": len(per_layer),
    }
