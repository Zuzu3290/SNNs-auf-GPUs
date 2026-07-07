"""
CPU utilization, thread, runtime, and energy tracker for SNN training.

Fully vendor-agnostic — works on AMD, Intel, ARM, any OS.
No vendor SDKs required.

Energy is read from the Linux RAPL interface (/sys/class/powercap/) when
running on Linux or WSL. RAPL is exposed by the kernel for both Intel and
AMD CPUs (AMD support added in kernel 5.11). If RAPL is unavailable (bare
Windows), energy is reported as None — all other metrics still work.

Mirrors GPUStats so CPU-only and GPU runs produce equivalent diagnostic
output for side-by-side benchmark comparison.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional
import psutil

# RAPL energy file — works on both Intel and AMD under Linux/WSL
_RAPL_PATH = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")


def _read_rapl_uj() -> Optional[int]:
    """Read current RAPL energy counter in microjoules. Returns None if unavailable."""
    try:
        return int(_RAPL_PATH.read_text().strip())
    except Exception:
        return None


class CPUStats:
    """
    Per-epoch and overall CPU diagnostics for training runs.

    Usage mirrors GPUStats:
        stats = CPUStats()
        stats.start_epoch()
        # ... training loop ...
        epoch_result = stats.end_epoch()
        overall      = stats.summary()
    """

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval  = sample_interval
        self.total_ram_gb     = psutil.virtual_memory().total / (1024 ** 3)
        self._process         = psutil.Process()
        self.rapl_available   = _read_rapl_uj() is not None

        self.epoch_cpu_samples:    list[float] = []
        self.epoch_ram_samples:    list[float] = []
        self.epoch_thread_samples: list[int]   = []
        self.all_cpu_samples:      list[float] = []
        self.peak_ram_each:        list[float] = []
        self.epoch_durations_s:    list[float] = []
        self.epoch_energy_j:       list[Optional[float]] = []

        self._stop_event   = threading.Event()
        self._thread: threading.Thread | None = None
        self._epoch_start: float    = 0.0
        self._rapl_start:  Optional[int] = None

    def start_epoch(self) -> None:
        """Call at the start of each epoch before the batch loop."""
        self.epoch_cpu_samples    = []
        self.epoch_ram_samples    = []
        self.epoch_thread_samples = []
        self._stop_event.clear()
        self._epoch_start = time.perf_counter()
        self._rapl_start  = _read_rapl_uj()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        """Background thread: samples CPU utilization, RAM, and thread count."""
        while not self._stop_event.wait(self.sample_interval):
            try:
                self.epoch_cpu_samples.append(psutil.cpu_percent(interval=None))
                self.epoch_ram_samples.append(
                    psutil.virtual_memory().used / (1024 ** 3)
                )
                self.epoch_thread_samples.append(self._process.num_threads())
            except Exception:
                pass

    def end_epoch(self) -> dict:
        """Call after the last batch of an epoch. Stops sampler and returns stats."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        elapsed_s = time.perf_counter() - self._epoch_start
        self.epoch_durations_s.append(elapsed_s)

        # --- energy via RAPL ---
        energy_j: Optional[float] = None
        rapl_end = _read_rapl_uj()
        if self._rapl_start is not None and rapl_end is not None:
            delta_uj = rapl_end - self._rapl_start
            # RAPL counter wraps at max_energy_range_uj — handle rollover
            if delta_uj < 0:
                try:
                    max_uj = int(
                        Path("/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj")
                        .read_text().strip()
                    )
                    delta_uj += max_uj
                except Exception:
                    delta_uj = 0
            energy_j = delta_uj / 1_000_000.0
        self.epoch_energy_j.append(energy_j)

        # --- utilization ---
        avg_cpu  = (
            sum(self.epoch_cpu_samples) / len(self.epoch_cpu_samples)
            if self.epoch_cpu_samples else 0.0
        )
        peak_cpu = max(self.epoch_cpu_samples) if self.epoch_cpu_samples else 0.0

        # --- memory ---
        peak_ram_gb  = max(self.epoch_ram_samples) if self.epoch_ram_samples else 0.0
        curr_ram_gb  = psutil.virtual_memory().used / (1024 ** 3)
        peak_ram_pct = peak_ram_gb / self.total_ram_gb * 100

        # --- threads ---
        avg_threads  = (
            sum(self.epoch_thread_samples) / len(self.epoch_thread_samples)
            if self.epoch_thread_samples else 0.0
        )
        peak_threads = (
            max(self.epoch_thread_samples) if self.epoch_thread_samples else 0
        )

        self.all_cpu_samples.extend(self.epoch_cpu_samples)
        self.peak_ram_each.append(peak_ram_gb)

        return {
            "cpu_util_avg_pct":  round(avg_cpu,      1),
            "cpu_util_peak_pct": round(peak_cpu,     1),
            "ram_peak_gb":       round(peak_ram_gb,  2),
            "ram_curr_gb":       round(curr_ram_gb,  2),
            "ram_peak_pct":      round(peak_ram_pct, 1),
            "threads_avg":       round(avg_threads,  1),
            "threads_peak":      int(peak_threads),
            "epoch_duration_s":  round(elapsed_s,    2),
            "cpu_energy_j":      round(energy_j, 4) if energy_j is not None else None,
        }

    def summary(self) -> dict:
        """Overall stats across all completed epochs."""
        if not self.all_cpu_samples:
            return {}

        overall_avg_cpu  = sum(self.all_cpu_samples) / len(self.all_cpu_samples)
        overall_peak_cpu = max(self.all_cpu_samples)
        peak_ram         = max(self.peak_ram_each) if self.peak_ram_each else 0.0
        peak_ram_pct     = peak_ram / self.total_ram_gb * 100
        total_time_s     = sum(self.epoch_durations_s)

        valid_energy     = [e for e in self.epoch_energy_j if e is not None]
        total_energy_j   = sum(valid_energy) if valid_energy else None

        return {
            "overall_avg_cpu_pct":  round(overall_avg_cpu,  1),
            "overall_peak_cpu_pct": round(overall_peak_cpu, 1),
            "overall_peak_ram_gb":  round(peak_ram,         2),
            "overall_peak_ram_pct": round(peak_ram_pct,     1),
            "total_ram_gb":         round(self.total_ram_gb, 2),
            "total_training_s":     round(total_time_s,     2),
            "total_cpu_energy_j":   round(total_energy_j, 4) if total_energy_j is not None else None,
            "rapl_available":       self.rapl_available,
        }
