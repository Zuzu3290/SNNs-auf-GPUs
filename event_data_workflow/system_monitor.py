"""Resource probes for cache and worker decisions, and a background GPU sampler for utilisation, power and energy."""
from __future__ import annotations
import os
import shutil
import logging
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import psutil
import torch
import pynvml

logger = logging.getLogger(__name__)

# Spawn-started workers (always on Windows) re-import torch and CUDA; measured ~1.6 GB of committed memory each.
SPAWN_WORKER_RELOAD_OVERHEAD_GB = 1.6


def worker_start_method() -> str:
    return mp.get_context().get_start_method()


def apply_worker_cap(num_workers: int, max_workers: int, reason: str) -> int:
    """Clamp num_workers to max_workers, logging the reason only when the cap binds."""
    capped = max(0, min(num_workers, max_workers))
    if capped < num_workers:
        logger.info(f"[PIPELINE] Worker count capped by {reason} -> using {capped} workers")
    return capped


def shared_memory_budget_bytes(available_ram_gb: float, worker_ram_fraction: float) -> float:
    """RAM budget for in-flight worker batches, capped by free /dev/shm where it exists (Linux, Docker)."""
    ram_budget = available_ram_gb * worker_ram_fraction * (1024 ** 3)
    if not Path("/dev/shm").exists():
        return ram_budget
    return min(ram_budget, shutil.disk_usage("/dev/shm").free)


@dataclass
class CacheMetrics:
    """Snapshot of system resource availability."""
    total_ram_gb: float
    available_ram_gb: float
    ram_usage_percent: float
    disk_available_gb: float
    disk_exists: bool
    gpu_memory_gb: float
    gpu_available_gb: float


def gpu_pressure(metrics: CacheMetrics) -> float:
    """VRAM utilisation as 0-1; 0 when no GPU is present."""
    if metrics.gpu_memory_gb == 0:
        return 0.0
    return 1.0 - (metrics.gpu_available_gb / metrics.gpu_memory_gb)


def is_gpu_under_pressure(metrics: CacheMetrics, threshold: float) -> bool:
    return gpu_pressure(metrics) > threshold


# Free-VRAM margin per phase, used only for the headroom log line.
GPU_PHASE_MARGINS: dict[str, float] = {
    "training": 0.20,
    "testing":  0.10,
}


def gpu_total_memory_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


class SystemResourceMonitor:
    """Live RAM, disk and VRAM probe, shared as the module-level `monitor` instance."""

    def __init__(self, cache_path: str = "./cache", cuda_enabled: bool = True):
        self.cache_path = Path(cache_path)
        self.cuda_enabled = cuda_enabled and torch.cuda.is_available()
        self.phase = "training"

    def configure(self, cache_path: str, cuda_enabled: bool) -> None:
        """Set the run's cache path and device once they are known."""
        self.cache_path = Path(cache_path)
        self.cuda_enabled = cuda_enabled and torch.cuda.is_available()

    def snapshot(self) -> CacheMetrics:
        vm = psutil.virtual_memory()

        Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        disk_available = shutil.disk_usage(self.cache_path).free / (1024 ** 3)

        if self.cuda_enabled:
            gpu_total = gpu_total_memory_gb()
            # Conservative: the smaller of driver-reported free VRAM and total minus PyTorch's reserved pool.
            free_driver   = torch.cuda.mem_get_info(0)[0]
            reserved      = torch.cuda.memory_reserved(0)
            total_bytes   = torch.cuda.get_device_properties(0).total_memory
            gpu_available = min(free_driver, total_bytes - reserved) / (1024 ** 3)
        else:
            gpu_total     = 0.0
            gpu_available = 0.0

        return CacheMetrics(
            total_ram_gb=vm.total / (1024 ** 3),
            available_ram_gb=vm.available / (1024 ** 3),
            ram_usage_percent=vm.percent,
            disk_available_gb=disk_available,
            disk_exists=True,
            gpu_memory_gb=gpu_total,
            gpu_available_gb=gpu_available,
        )

    def worker_count(self, device: torch.device, safety_margin_gb: float = 2.0,
                     worker_count_override=None, metrics=None) -> tuple[int, CacheMetrics, int]:
        """DataLoader worker count from physical cores and spawn-reload RAM cost, or the fixed override."""
        metrics = metrics or self.snapshot()
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        num_workers = physical_cores if metrics.available_ram_gb >= safety_margin_gb else 0

        if worker_count_override is not None:
            num_workers = max(0, min(physical_cores, worker_count_override))
            logger.warning(
                f"[PIPELINE] worker_count_override={worker_count_override} active -> using {num_workers} "
                f"workers, bypassing the RAM/spawn-reload safety caps (available RAM: {metrics.available_ram_gb:.2f}GB)"
            )
        elif num_workers > 0 and worker_start_method() != "fork":
            headroom_gb = max(0.0, metrics.available_ram_gb - safety_margin_gb)
            max_workers_by_reload_overhead = int(headroom_gb / SPAWN_WORKER_RELOAD_OVERHEAD_GB)
            num_workers = apply_worker_cap(num_workers, max_workers_by_reload_overhead,
                f"spawn-reload overhead: {num_workers} workers would each re-import torch+CUDA "
                f"(~{SPAWN_WORKER_RELOAD_OVERHEAD_GB:.1f}GB commit) on spawn, over the {headroom_gb:.2f}GB "
                f"headroom above the {safety_margin_gb:.2f}GB safety margin")

        return num_workers, metrics, physical_cores

    def dataloader_config(self, device: torch.device, safety_margin_gb: float = 2.0,
                          bytes_per_batch: float = 0.0, worker_ram_fraction: float = 0.25, worker_timeout_s: float = 0.0,
                          worker_count_override=None, metrics=None) -> dict:
        """DataLoader kwargs: worker count capped by cores, spawn-reload cost and the RAM budget per batch."""
        cuda_enabled = device is not None and getattr(device, "type", "") == "cuda"
        num_workers, metrics, physical_cores = self.worker_count(device, safety_margin_gb, worker_count_override, metrics)
        prefetch_factor = 2

        if worker_count_override is None and num_workers > 0 and bytes_per_batch > 0:
            budget_bytes = shared_memory_budget_bytes(metrics.available_ram_gb, worker_ram_fraction)
            max_workers_by_ram = int(budget_bytes / (prefetch_factor * bytes_per_batch))
            num_workers = apply_worker_cap(num_workers, max_workers_by_ram,
                f"RAM budget: {physical_cores} physical cores would need ~{physical_cores * prefetch_factor * bytes_per_batch / (1024**3):.2f}GB "
                f"of in-flight worker shared memory ({bytes_per_batch / (1024**2):.1f}MB/batch x prefetch_factor={prefetch_factor}), "
                f"over the {worker_ram_fraction:.0%} of {metrics.available_ram_gb:.2f}GB available RAM this policy allows")

        logger.info(f"[PIPELINE] Available workers: {physical_cores} physical cores -> using {num_workers}")

        if cuda_enabled and num_workers == 0:
            cfg = {"num_workers": 0, "prefetch_factor": None, "pin_memory": False, "persistent_workers": False, "timeout": 0}
            logger.info(
                f"[PIPELINE] GPU-only mode — num_workers=0, pin_memory=False "
                f"(available RAM {metrics.available_ram_gb:.2f}GB below the "
                f"{safety_margin_gb:.2f}GB safety margin, or the RAM budget for "
                f"this dataset's batch size capped it to zero)"
            )
        else:
            cfg = {
                "num_workers":        num_workers,
                "prefetch_factor":    prefetch_factor if num_workers > 0 else None,
                "pin_memory":         True,
                "persistent_workers": num_workers > 0,
                "timeout":            worker_timeout_s if num_workers > 0 else 0,  # a stuck worker raises instead of hanging
            }

        logger.info(f"[PIPELINE] DataLoader config: {cfg}")
        return cfg

    def enter_phase(self, phase: str) -> None:
        """Mark the training/testing boundary and log free VRAM against that phase's margin."""
        if phase not in GPU_PHASE_MARGINS:
            raise ValueError(f"Unknown phase '{phase}'. Valid: {list(GPU_PHASE_MARGINS)}")
        self.phase = phase
        self.log_headroom()

    def log_headroom(self) -> None:
        if not self.cuda_enabled:
            return
        metrics = self.snapshot()
        if metrics.gpu_memory_gb <= 0:
            return
        margin_gb = metrics.gpu_memory_gb * GPU_PHASE_MARGINS[self.phase]
        if metrics.gpu_available_gb < margin_gb:
            logger.warning(
                f"[GPU MEMORY] {self.phase} phase: {metrics.gpu_available_gb:.2f}GB free VRAM, "
                f"below the {margin_gb:.2f}GB safety margin for this phase — model weights, "
                f"activations, gradients, or optimizer state may be running close to the edge."
            )
        else:
            logger.info(
                f"[GPU MEMORY] {self.phase} phase: {metrics.gpu_available_gb:.2f}GB free VRAM "
                f"(safety margin {margin_gb:.2f}GB)"
            )


monitor = SystemResourceMonitor()


@dataclass
class PipelineSample:
    """One timestamped reading from PipelineMonitor's background thread."""
    t_s: float
    phase: str
    cpu_percent: float
    ram_available_gb: float
    gpu_util_pct: float
    gpu_power_w: float
    gpu_sm_clock_mhz: float


@dataclass
class BoundThresholds:
    """Levels and durations after which an idle GPU, saturated CPU or low RAM counts as a sustained episode."""
    gpu_idle_pct_below:        float = 5.0
    gpu_idle_sustained_s:      float = 5.0
    cpu_saturated_pct_above:   float = 95.0
    cpu_saturated_sustained_s: float = 10.0
    ram_available_gb_below:    float = 1.0
    ram_low_sustained_s:       float = 5.0


@dataclass
class BoundViolation:
    """One sustained out-of-bounds episode."""
    kind: str
    started_t_s: float
    phase: str
    ended_t_s: float = None


class PipelineMonitor:
    """Background sampler of GPU utilisation, power and clock, CPU and RAM, tagged by phase (one phase per epoch)."""

    def __init__(self, interval_s: float = 0.2):
        self.cuda_enabled = torch.cuda.is_available()
        self.interval_s   = interval_s
        self.thresholds   = BoundThresholds()

        self.total_memory_gb = gpu_total_memory_gb() if self.cuda_enabled else 0.0
        self.peak_mem_each: list[float] = []
        self.idle_power_w = None

        self.nvml_handle = None
        if self.cuda_enabled:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        self.samples: list[PipelineSample] = []
        self.violations: list[BoundViolation] = []
        self.phase = "unset"
        self.t_origin = None
        self.stop_event = threading.Event()
        self.thread = None
        self.open_violation = {"gpu_idle": None, "cpu_saturated": None, "ram_low": None}

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def reset_epoch_memory(self) -> None:
        """Reset PyTorch's peak-VRAM counter at the start of a phase."""
        if self.cuda_enabled:
            torch.cuda.reset_peak_memory_stats(0)

    def epoch_memory_gb(self) -> float:
        """Peak VRAM in GB since the last reset_epoch_memory()."""
        if not self.cuda_enabled:
            return 0.0
        return torch.cuda.max_memory_allocated(0) / (1024 ** 3)

    def measure_idle_baseline(self, duration_s: float = 1.0):
        """Average GPU power over a short idle window, the baseline for dynamic energy; None without NVML."""
        if self.nvml_handle is None:
            return None
        deadline = time.perf_counter() + duration_s
        samples = []
        while time.perf_counter() < deadline:
            samples.append(float(pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)))
            time.sleep(self.interval_s / 2)
        if samples:
            self.idle_power_w = (sum(samples) / len(samples)) * 1e-3
        return self.idle_power_w

    def phase_summary(self, phase: str) -> dict:
        """GPU utilisation, peak VRAM and idle episodes for one phase."""
        samples  = [s for s in self.samples if s.phase == phase]
        gpu_vals = [s.gpu_util_pct for s in samples if s.gpu_util_pct is not None]
        peak_gb  = self.epoch_memory_gb()
        peak_pct = peak_gb / self.total_memory_gb * 100 if self.total_memory_gb > 0 else 0.0
        self.peak_mem_each.append(peak_gb)

        now = time.perf_counter() - (self.t_origin or 0.0)
        idle_episodes = [v for v in self.violations if v.kind == "gpu_idle" and v.phase == phase]
        open_idle = self.open_violation.get("gpu_idle")
        if open_idle is not None and open_idle.phase == phase and open_idle not in self.violations:
            idle_episodes = idle_episodes + [open_idle]
        idle_total_s = sum((v.ended_t_s or now) - v.started_t_s for v in idle_episodes)

        return {
            "gpu_util_avg_pct":  round(sum(gpu_vals) / len(gpu_vals), 1) if gpu_vals else 0.0,
            "gpu_util_peak_pct": round(max(gpu_vals), 1) if gpu_vals else 0.0,
            "gpu_mem_peak_gb":   round(peak_gb, 2),
            "gpu_mem_peak_pct":  round(peak_pct, 1),
            "gpu_idle_episodes": len(idle_episodes),
            "gpu_idle_total_s":  round(idle_total_s, 1),
        }

    def phase_energy_report(self, phase: str, elapsed_s: float) -> dict:
        """GPU energy for one phase: total (idle draw included) and dynamic (above the idle baseline), with powers."""
        power_vals = [s.gpu_power_w for s in self.samples if s.phase == phase and s.gpu_power_w is not None]
        energy_j = dynamic_energy_j = avg_power_w = dynamic_power_w = None
        if power_vals:
            avg_power_w     = sum(power_vals) / len(power_vals)
            idle_w          = self.idle_power_w or 0.0
            dynamic_power_w = max(0.0, avg_power_w - idle_w)
            energy_j        = avg_power_w * elapsed_s
            dynamic_energy_j = dynamic_power_w * elapsed_s
        return {
            "gpu":                  self.phase_summary(phase),
            "gpu_energy_j":         energy_j,
            "gpu_dynamic_energy_j": dynamic_energy_j,
            "avg_power_w":          avg_power_w,
            "dynamic_power_w":      dynamic_power_w,
            "idle_power_w":         self.idle_power_w,
        }

    def start(self) -> None:
        self.t_origin = time.perf_counter()
        self.stop_event.clear()
        psutil.cpu_percent(interval=None)  # prime psutil's reference point
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        now = time.perf_counter() - (self.t_origin or 0.0)
        for v in self.open_violation.values():
            if v is not None:
                v.ended_t_s = now

    def loop(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            self.sample_once()

    def sample_once(self) -> None:
        t = time.perf_counter() - self.t_origin
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_gb  = psutil.virtual_memory().available / (1024 ** 3)

        gpu_util = gpu_power = gpu_clock = None
        if self.nvml_handle is not None:
            gpu_util  = float(pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu)
            gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
            gpu_clock = float(pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, pynvml.NVML_CLOCK_SM))

        self.samples.append(PipelineSample(
            t_s=t, phase=self.phase, cpu_percent=cpu_pct, ram_available_gb=ram_gb,
            gpu_util_pct=gpu_util, gpu_power_w=gpu_power, gpu_sm_clock_mhz=gpu_clock,
        ))

        # Not checked before the first phase: an idle GPU during setup and caching is expected.
        self.check_bound("gpu_idle", self.phase != "unset" and gpu_util is not None
                           and gpu_util < self.thresholds.gpu_idle_pct_below,
                           t, self.thresholds.gpu_idle_sustained_s)
        self.check_bound("cpu_saturated", cpu_pct > self.thresholds.cpu_saturated_pct_above,
                           t, self.thresholds.cpu_saturated_sustained_s)
        self.check_bound("ram_low", ram_gb < self.thresholds.ram_available_gb_below,
                           t, self.thresholds.ram_low_sustained_s)

    def check_bound(self, kind: str, in_violation: bool, t: float, sustained_s: float) -> None:
        """Open an episode when a bound is crossed, record it once it lasts sustained_s, close it when it clears."""
        open_v = self.open_violation[kind]
        if in_violation:
            if open_v is None:
                self.open_violation[kind] = BoundViolation(kind=kind, started_t_s=t, phase=self.phase)
            elif open_v not in self.violations and (t - open_v.started_t_s) >= sustained_s:
                self.violations.append(open_v)
                logger.debug(
                    f"[PIPELINE MONITOR] {kind} sustained for {t - open_v.started_t_s:.1f}s "
                    f"(phase='{open_v.phase}', started at t={open_v.started_t_s:.1f}s)"
                )
        else:
            if open_v is not None:
                if open_v in self.violations:
                    open_v.ended_t_s = t
                self.open_violation[kind] = None

    def summary(self) -> dict:
        """Utilisation and peak VRAM across the whole run, plus the violation log."""
        if not self.samples:
            return {"n_samples": 0, "violations": []}
        cpu = [s.cpu_percent for s in self.samples]
        gpu = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        peak_mem     = max(self.peak_mem_each) if self.peak_mem_each else 0.0
        peak_mem_pct = peak_mem / self.total_memory_gb * 100 if self.total_memory_gb > 0 else 0.0
        return {
            "n_samples":            len(self.samples),
            "avg_cpu_percent":      round(sum(cpu) / len(cpu), 1),
            "max_cpu_percent":      round(max(cpu), 1),
            "avg_gpu_util_pct":     round(sum(gpu) / len(gpu), 1) if gpu else None,
            "max_gpu_util_pct":     round(max(gpu), 1) if gpu else None,
            "overall_peak_mem_gb":  round(peak_mem, 2),
            "overall_peak_mem_pct": round(peak_mem_pct, 1),
            "total_vram_gb":        round(self.total_memory_gb, 2),
            "violations": [
                {"kind": v.kind, "started_t_s": round(v.started_t_s, 1),
                 "ended_t_s": round(v.ended_t_s, 1) if v.ended_t_s is not None else None,
                 "phase": v.phase}
                for v in self.violations
            ],
        }

    def memory_trend(self) -> dict:
        """Available host RAM at the first and last sample, to show whether usage drifted over the run."""
        if not self.samples:
            return {"n_samples": 0}
        ram_start = self.samples[0].ram_available_gb
        ram_end   = self.samples[-1].ram_available_gb
        ram_drift = ram_end - ram_start
        return {
            "n_samples":       len(self.samples),
            "ram_start_gb":    round(ram_start, 2),
            "ram_end_gb":      round(ram_end, 2),
            "ram_drift_gb":    round(ram_drift, 2),
            "ram_trend":       "stable" if abs(ram_drift) < 0.5 else ("declining" if ram_drift < 0 else "growing"),
            "gpu_peak_mem_gb": round(max(self.peak_mem_each), 2) if self.peak_mem_each else 0.0,
        }
