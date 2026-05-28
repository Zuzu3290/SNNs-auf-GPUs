"""
GPU utilization and memory tracker for SNN training.

Samples compute utilization in a background thread so the training loop
is never blocked. Memory peak is read from PyTorch's built-in counters.

Usage:
    stats = GPUStats(device_idx=0)

    stats.start_epoch()
    # ... training batches ...
    epoch_report = stats.end_epoch()   # dict of avg/peak util + peak VRAM

    overall = stats.summary()          # aggregated across all epochs
"""
from __future__ import annotations

import threading
import torch

# Optional NVML Python bindings for real power readings.
# Install with: pip install nvidia-ml-py
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    NVML_OK = True
except Exception:
    NVML_OK = False
    NVML_HANDLE = None


class GPUStats:
    """
    Per-epoch and overall GPU utilization/memory statistics for training.

    Compute utilization is sampled every `sample_interval` seconds from a
    daemon thread using torch.cuda.utilization(). Peak VRAM is read from
    PyTorch's memory counters (reset at the start of each epoch).
    """

    def __init__(self, device_idx: int = 0, sample_interval: float = 0.5):
        self.device_idx      = device_idx
        self.sample_interval = sample_interval
        self.available       = torch.cuda.is_available()

        self.total_memory_gb = (
            torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)
            if self.available else 0.0
        )

        self.epoch_samples:   list[float] = []
        self.all_samples:     list[float] = []
        self.peak_mem_each:   list[float] = []
        self.power_samples_mw: list[float] = []
        self.stop_event                   = threading.Event()
        self.thread: threading.Thread | None = None

    def start_epoch(self):
        """Call at the start of each epoch before the batch loop."""
        if not self.available:
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
            except Exception:
                pass
            if NVML_OK and NVML_HANDLE is not None:
                try:
                    self.power_samples_mw.append(float(pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE)))
                except Exception:
                    pass

    def gpu_energy_j(self, elapsed_s: float) -> float | None:
        """Return estimated GPU energy in joules for a timed region.

        Uses average of power samples collected during that region multiplied by
        elapsed wall-clock seconds. Returns None if NVML power data is unavailable.
        """
        if not self.power_samples_mw:
            return None
        avg_w = (sum(self.power_samples_mw) / len(self.power_samples_mw)) * 1e-3
        return avg_w * elapsed_s

    def end_epoch(self) -> dict:
        """
        Call after the last batch of an epoch.
        Stops the sampler thread and returns a stats dict.
        """
        if not self.available:
            return {}

        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

        peak_gb  = torch.cuda.max_memory_allocated(self.device_idx) / (1024 ** 3)
        curr_gb  = torch.cuda.memory_allocated(self.device_idx)     / (1024 ** 3)
        peak_pct = (peak_gb / self.total_memory_gb * 100) if self.total_memory_gb else 0.0

        avg_util  = sum(self.epoch_samples) / len(self.epoch_samples) if self.epoch_samples else 0.0
        peak_util = max(self.epoch_samples)                            if self.epoch_samples else 0.0

        # Detect NVML reporting failure. Two failure modes:
        #   1. torch.cuda.utilization() throws every call → epoch_samples is empty
        #   2. It returns 0 for all calls despite GPU memory being in active use
        # Both are physically impossible if the GPU is genuinely idle (memory would
        # also be ~0). If memory is in use but utilization shows nothing, NVML is broken.
        no_util_signal = (
            len(self.epoch_samples) == 0
            or max(self.epoch_samples) == 0.0
        )
        nvml_ok = not (no_util_signal and peak_gb > 0.1)

        self.all_samples.extend(self.epoch_samples)
        self.peak_mem_each.append(peak_gb)

        return {
            "gpu_util_avg_pct":   round(avg_util,  1),
            "gpu_util_peak_pct":  round(peak_util, 1),
            "gpu_util_available": nvml_ok,
            "gpu_mem_peak_gb":    round(peak_gb,   2),
            "gpu_mem_curr_gb":    round(curr_gb,   2),
            "gpu_mem_peak_pct":   round(peak_pct,  1),
        }

    def summary(self) -> dict:
        """Overall utilization and memory stats across all completed epochs."""
        if not self.available or not self.all_samples:
            return {}
        overall_avg  = sum(self.all_samples) / len(self.all_samples)
        overall_peak = max(self.all_samples)
        peak_mem     = max(self.peak_mem_each) if self.peak_mem_each else 0.0
        peak_mem_pct = (peak_mem / self.total_memory_gb * 100) if self.total_memory_gb else 0.0
        return {
            "overall_avg_util_pct":  round(overall_avg,  1),
            "overall_peak_util_pct": round(overall_peak, 1),
            "overall_peak_mem_gb":   round(peak_mem,     2),
            "overall_peak_mem_pct":  round(peak_mem_pct, 1),
            "total_vram_gb":         round(self.total_memory_gb, 2),
        }
