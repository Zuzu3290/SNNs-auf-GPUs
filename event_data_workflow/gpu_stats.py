"""
GPU utilization and memory tracker for SNN training.

Samples compute utilization in a background thread so the training loop
is never blocked. Memory peak is read from PyTorch's built-in counters.
NVML Python bindings for real power readings.

Degrades gracefully on a CPU-only run (or a machine without NVML): all
methods become no-ops returning empty/None rather than raising, since GPU
stats fundamentally don't apply without a GPU.
"""
from __future__ import annotations
import threading
import torch
import pynvml

class GPUStats:
    """
    Per-epoch and overall GPU utilization/memory statistics for training.

    Compute utilization is sampled every `sample_interval` seconds from a
    daemon thread using torch.cuda.utilization(). Peak VRAM is read from
    PyTorch's memory counters (reset at the start of each epoch).

    NVML is initialised lazily here (not at import time) so importing this
    module never crashes on a machine without an NVIDIA driver; `enabled`
    is False whenever CUDA itself isn't available, at which point every
    method is a no-op.
    """

    def __init__(self, device_idx: int = 0, sample_interval: float = 0.5):
        self.device_idx      = device_idx
        self.sample_interval = sample_interval
        self.enabled         = torch.cuda.is_available()

        self.total_memory_gb = 0.0
        self.nvml_handle     = None
        if self.enabled:
            self.total_memory_gb = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
            if pynvml is not None:
                try:
                    pynvml.nvmlInit()
                    self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                except Exception:
                    self.nvml_handle = None  # NVML present but unusable (e.g. driver mismatch)

        self.epoch_samples:   list[float] = []
        self.all_samples:     list[float] = []
        self.peak_mem_each:   list[float] = []
        self.power_samples_mw: list[float] = []
        self.stop_event                   = threading.Event()
        self.thread: threading.Thread | None = None

        self.idle_power_w: float | None = None

    def measure_idle_baseline(self, duration_s: float = 1.0) -> float | None:
        """Sample GPU power for a short window with no work scheduled, before a run
        starts. Lets gpu_energy_j()/dynamic_power_w() report *dynamic* power/energy
        (above idle draw) instead of the raw total, which otherwise overstates what
        a workload actually costs — the GPU pulls non-zero power just sitting idle.
        No-op (returns None) without CUDA/NVML. Safe to call more than once; last
        call wins."""
        if not self.enabled or self.nvml_handle is None:
            return None
        import time as _time
        samples = []
        deadline = _time.perf_counter() + duration_s
        while _time.perf_counter() < deadline:
            try:
                samples.append(float(pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)))
            except Exception:
                pass
            _time.sleep(self.sample_interval / 2)
        if samples:
            self.idle_power_w = (sum(samples) / len(samples)) * 1e-3
        return self.idle_power_w

    def dynamic_power_w(self, avg_power_w: float) -> float:
        """avg_power_w with the idle baseline subtracted (clamped at 0). Falls back
        to avg_power_w unchanged if measure_idle_baseline() was never called."""
        if self.idle_power_w is None:
            return avg_power_w
        return max(0.0, avg_power_w - self.idle_power_w)

    def start_epoch(self):
        """Call at the start of each epoch before the batch loop. No-op without CUDA."""
        if not self.enabled:
            return
        torch.cuda.reset_peak_memory_stats(self.device_idx)
        self.epoch_samples    = []
        self.power_samples_mw = []
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.sample_loop, daemon=True)
        self.thread.start()

    def sample_loop(self):
        """Background thread: records compute utilization % and GPU power at fixed intervals."""
        while not self.stop_event.wait(self.sample_interval):
            try:
                self.epoch_samples.append(float(torch.cuda.utilization(self.device_idx)))
                if self.nvml_handle is not None:
                    self.power_samples_mw.append(float(pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)))
            except Exception:
                pass

    def gpu_energy_j(self, elapsed_s: float) -> float | None:
        """Return estimated GPU energy in joules for a timed region, or None if no power data (no CUDA / no NVML)."""
        if not self.power_samples_mw:
            return None
        avg_w = (sum(self.power_samples_mw) / len(self.power_samples_mw)) * 1e-3
        return avg_w * elapsed_s

    def end_epoch(self) -> dict:
        """Call after the last batch of an epoch. Stops the sampler thread and returns a stats dict. {} without CUDA."""
        if not self.enabled:
            return {}
        self.stop_event.set()
        assert self.thread is not None
        self.thread.join(timeout=2.0)

        peak_gb  = torch.cuda.max_memory_allocated(self.device_idx) / (1024 ** 3)
        curr_gb  = torch.cuda.memory_allocated(self.device_idx)     / (1024 ** 3)
        peak_pct = peak_gb / self.total_memory_gb * 100 if self.total_memory_gb > 0 else 0.0
        avg_util  = sum(self.epoch_samples) / len(self.epoch_samples) if self.epoch_samples else 0.0
        peak_util = max(self.epoch_samples) if self.epoch_samples else 0.0

        self.all_samples.extend(self.epoch_samples)
        self.peak_mem_each.append(peak_gb)

        return {
            "gpu_util_avg_pct":  round(avg_util,  1),
            "gpu_util_peak_pct": round(peak_util, 1),
            "gpu_mem_peak_gb":   round(peak_gb,   2),
            "gpu_mem_curr_gb":   round(curr_gb,   2),
            "gpu_mem_peak_pct":  round(peak_pct,  1),
        }

    def summary(self) -> dict:
        """Overall utilization and memory stats across all completed epochs. {} without CUDA or data."""
        if not self.enabled or not self.all_samples:
            return {}
        overall_avg  = sum(self.all_samples) / len(self.all_samples)
        overall_peak = max(self.all_samples)
        peak_mem     = max(self.peak_mem_each) if self.peak_mem_each else 0.0
        peak_mem_pct = peak_mem / self.total_memory_gb * 100 if self.total_memory_gb > 0 else 0.0
        return {
            "overall_avg_util_pct":  round(overall_avg,  1),
            "overall_peak_util_pct": round(overall_peak, 1),
            "overall_peak_mem_gb":   round(peak_mem,     2),
            "overall_peak_mem_pct":  round(peak_mem_pct, 1),
            "total_vram_gb":         round(self.total_memory_gb, 2),
        }
