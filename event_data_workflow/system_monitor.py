"""
Probes CPU, RAM, GPU, and disk availability for cache/worker decisions, and
(via PipelineMonitor below) continuous background monitoring of the same
resources during a real run.
"""
from __future__ import annotations
import shutil
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import psutil
import torch
import pynvml

logger = logging.getLogger(__name__)

GPU_PRESSURE_THRESHOLD = 0.75


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
    cpu_percent: float      # 0 on the very first call ever (psutil has no prior
                             # reference point yet) -- meaningful from the second
                             # call onward. See SystemResourceMonitor.snapshot().
    cpu_count: int


def gpu_pressure(metrics: CacheMetrics) -> float:
    """VRAM utilisation as 0–1. 0 when no GPU is present."""
    if metrics.gpu_memory_gb == 0:
        return 0.0
    return 1.0 - (metrics.gpu_available_gb / metrics.gpu_memory_gb)


def is_gpu_under_pressure(metrics: CacheMetrics, threshold: float = GPU_PRESSURE_THRESHOLD) -> bool:
    return gpu_pressure(metrics) > threshold


# GPU VRAM safety margin per phase — diagnostic only, see
# SystemResourceMonitor.enter_phase(). "training" carries the larger margin
# since the backward pass and optimizer state compete hardest for VRAM;
# "testing" (inference, no gradients/optimizer state active) needs less
# headroom.
GPU_PHASE_MARGINS: dict[str, float] = {
    "training": 0.20,
    "testing":  0.10,
}


def gpu_total_memory_gb(device_idx: int = 0) -> float:
    """Total VRAM for one CUDA device, in GB. 0.0 if CUDA isn't available.
    The single place that reads device capacity — SystemResourceMonitor.snapshot()
    and GPUStats both call this instead of each independently calling
    torch.cuda.get_device_properties()."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 3)


class SystemResourceMonitor:
    """Live RAM/GPU/disk probe, plus phase-aware VRAM headroom diagnostics
    at the training/testing boundary (see enter_phase()). One instance,
    `monitor` below, is built once at import time and shared: every file
    that needs live resource state imports that instance directly
    (`from .system_monitor import monitor`) and calls its methods, instead
    of each building its own probe and re-querying the same RAM/disk/VRAM
    state independently."""

    def __init__(self, cache_path: str = "./cache", device_idx: int = 0, cuda_enabled: bool = True):
        self.cache_path = Path(cache_path)
        self.device_idx = device_idx
        # False forces GPU fields to 0 even when CUDA is physically present —
        # set when the run is explicitly CPU-only, so a GPU that exists but
        # isn't requested never affects cache/worker decisions.
        self.cuda_enabled = cuda_enabled
        self.phase = "training"  # see enter_phase()

    def configure(self, cache_path: str = "./cache", device_idx: int = 0, cuda_enabled: bool = True) -> None:
        """Set the run's device/cache_path once, early, as soon as the run's
        device is known (NeuromorphicEncoder.__init__). Unconfigured, the
        constructor defaults above apply."""
        self.cache_path = Path(cache_path)
        self.device_idx = device_idx
        self.cuda_enabled = cuda_enabled

    def snapshot(self) -> CacheMetrics:
        vm = psutil.virtual_memory()
        # interval=None: non-blocking, returns usage since the LAST call to
        # cpu_percent() on this process (0.0 on the very first-ever call,
        # since there's no prior reference point yet). A blocking interval
        # would give a more precise single reading but stalls the caller for
        # that long, which snapshot() callers (cache/worker decisions,
        # potentially per-batch monitoring) shouldn't pay.
        cpu_percent = psutil.cpu_percent(interval=None)

        try:
            disk_stat = shutil.disk_usage(self.cache_path)
            disk_available = disk_stat.free / (1024 ** 3)
            disk_exists = True
        except Exception:
            disk_available = 0.0
            disk_exists = False

        if self.cuda_enabled and torch.cuda.is_available():
            gpu_total = gpu_total_memory_gb(self.device_idx)

            # Most conservative available-VRAM estimate: driver-reported free
            # space vs. total minus PyTorch's own reserved pool, whichever is smaller.
            free_driver    = torch.cuda.mem_get_info(self.device_idx)[0]
            reserved       = torch.cuda.memory_reserved(self.device_idx)
            total_bytes    = torch.cuda.get_device_properties(self.device_idx).total_memory
            gpu_available  = min(free_driver, total_bytes - reserved) / (1024 ** 3)
        else:
            gpu_total     = 0.0
            gpu_available = 0.0

        return CacheMetrics(
            total_ram_gb=vm.total / (1024 ** 3),
            available_ram_gb=vm.available / (1024 ** 3),
            ram_usage_percent=vm.percent,
            disk_available_gb=disk_available,
            disk_exists=disk_exists,
            gpu_memory_gb=gpu_total,
            gpu_available_gb=gpu_available,
            cpu_percent=cpu_percent,
            cpu_count=psutil.cpu_count(logical=True) or 1,
        )

    def enter_phase(self, phase: str) -> None:
        """Diagnostic-only VRAM headroom check at the training/testing phase
        boundary. Logs whether free VRAM is above or below that phase's
        safety margin (GPU_PHASE_MARGINS), so a run's logs show whether the
        model's own weights, activations, gradients, or optimizer state are
        running close to the edge. Takes no corrective action. Call once at
        the real train/test boundary (SNNTrainer.train(), SNNTester.run())."""
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


# The one shared instance — every file imports this directly instead of
# constructing its own SystemResourceMonitor.
monitor = SystemResourceMonitor()


@dataclass
class PipelineSample:
    """One timestamped reading from PipelineMonitor's background thread."""
    t_s: float               # seconds since PipelineMonitor.start()
    phase: str                # caller-set label — e.g. "fetch"/"transfer"/"compute",
                               # or "train"/"eval" for a real run. Purely descriptive.
    cpu_percent: float
    ram_available_gb: float
    gpu_util_pct: float | None    # None when no CUDA device / NVML unavailable
    gpu_power_w: float | None
    gpu_sm_clock_mhz: float | None


@dataclass
class BoundThresholds:
    """Sustained-violation thresholds for PipelineMonitor's "out of bounds"
    flags. A single instantaneous sample crossing a threshold is never
    flagged on its own — only a violation that persists for `*_sustained_s`
    is, so a momentary blip (a GC pause, another process waking up) doesn't
    spam warnings. Defaults are conservative starting points, not tuned for
    any specific dataset/model — override per use case."""
    gpu_idle_pct_below:        float = 5.0
    gpu_idle_sustained_s:      float = 5.0
    cpu_saturated_pct_above:   float = 95.0
    cpu_saturated_sustained_s: float = 10.0
    ram_available_gb_below:    float = 1.0
    ram_low_sustained_s:       float = 5.0


@dataclass
class BoundViolation:
    """One sustained out-of-bounds episode, logged once when it first crosses
    `sustained_s` (not once per sample) and again when it clears."""
    kind: str              # "gpu_idle" | "cpu_saturated" | "ram_low"
    started_t_s: float
    phase: str             # phase active when the violation started
    ended_t_s: float | None = None


class PipelineMonitor:
    """
    Continuous background sampler for CPU + GPU utilization/power/clock +
    RAM, shared by the offline diagnostics harness
    (diagnostics/gpu_utilization_harness.py) and, optionally, real training/
    inference runs — the same probing SystemResourceMonitor does
    point-in-time for cache decisions, extended into a running, phase-
    taggable trace with sustained-violation ("out of bounds") flagging.

    Sampling is NVML/psutil driver queries only — never touches a CUDA
    tensor or calls torch.cuda.synchronize() — so it never perturbs the
    training loop it's watching, the same non-blocking property GPUStats'
    existing sampler thread already relies on.

    Usage:
        monitor = PipelineMonitor(device_idx=0)
        monitor.start()
        monitor.set_phase("fetch")
        ...
        monitor.set_phase("compute")
        ...
        monitor.stop()
        for v in monitor.violations: ...   # sustained out-of-bounds episodes
    """

    def __init__(
        self,
        device_idx: int = 0,
        cuda_enabled: bool = True,
        interval_s: float = 0.2,
        thresholds: BoundThresholds | None = None,
    ):
        self.device_idx    = device_idx
        self.cuda_enabled  = cuda_enabled and torch.cuda.is_available()
        self.interval_s    = interval_s
        self.thresholds    = thresholds or BoundThresholds()

        self.nvml_handle = None
        if self.cuda_enabled:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            except Exception:
                self.nvml_handle = None

        self.samples: list[PipelineSample] = []
        self.violations: list[BoundViolation] = []
        self._phase = "unset"
        self._t_origin: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # in-progress violation start time per kind, or None if currently within bounds
        self._open_violation: dict[str, BoundViolation | None] = {
            "gpu_idle": None, "cpu_saturated": None, "ram_low": None,
        }

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    def start(self) -> None:
        self._t_origin = time.perf_counter()
        self._stop_event.clear()
        psutil.cpu_percent(interval=None)  # prime the reference point (see snapshot()'s docstring note)
        self._thread = threading.Thread(target=self.loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # close out any still-open violation episodes at the moment of stopping
        now = time.perf_counter() - (self._t_origin or 0.0)
        for kind, v in self._open_violation.items():
            if v is not None:
                v.ended_t_s = now

    def loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                self.sample_once()
            except Exception:
                pass  # a monitor must never take down the run it's watching

    def sample_once(self) -> None:
        t = time.perf_counter() - self._t_origin
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_gb  = psutil.virtual_memory().available / (1024 ** 3)

        gpu_util = gpu_power = gpu_clock = None
        if self.nvml_handle is not None:
            try:
                gpu_util  = float(pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu)
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
                gpu_clock = float(pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, pynvml.NVML_CLOCK_SM))
            except Exception:
                pass

        self.samples.append(PipelineSample(
            t_s=t, phase=self._phase, cpu_percent=cpu_pct, ram_available_gb=ram_gb,
            gpu_util_pct=gpu_util, gpu_power_w=gpu_power, gpu_sm_clock_mhz=gpu_clock,
        ))

        self.check_bound("gpu_idle", gpu_util is not None and gpu_util < self.thresholds.gpu_idle_pct_below,
                           t, self.thresholds.gpu_idle_sustained_s)
        self.check_bound("cpu_saturated", cpu_pct > self.thresholds.cpu_saturated_pct_above,
                           t, self.thresholds.cpu_saturated_sustained_s)
        self.check_bound("ram_low", ram_gb < self.thresholds.ram_available_gb_below,
                           t, self.thresholds.ram_low_sustained_s)

    def check_bound(self, kind: str, in_violation: bool, t: float, sustained_s: float) -> None:
        open_v = self._open_violation[kind]
        if in_violation:
            if open_v is None:
                # Not flagged yet — just start tracking; only becomes a
                # logged BoundViolation once it's been sustained long enough.
                self._open_violation[kind] = BoundViolation(kind=kind, started_t_s=t, phase=self._phase)
            elif open_v not in self.violations and (t - open_v.started_t_s) >= sustained_s:
                self.violations.append(open_v)
                logger.warning(
                    f"[PIPELINE MONITOR] {kind} sustained for {t - open_v.started_t_s:.1f}s "
                    f"(phase='{open_v.phase}', started at t={open_v.started_t_s:.1f}s)"
                )
        else:
            if open_v is not None:
                if open_v in self.violations:
                    open_v.ended_t_s = t
                self._open_violation[kind] = None

    def summary(self) -> dict:
        """Aggregate stats across the whole run so far, plus the violation log."""
        if not self.samples:
            return {"n_samples": 0, "violations": []}
        cpu = [s.cpu_percent for s in self.samples]
        gpu = [s.gpu_util_pct for s in self.samples if s.gpu_util_pct is not None]
        return {
            "n_samples":        len(self.samples),
            "avg_cpu_percent":  round(sum(cpu) / len(cpu), 1),
            "max_cpu_percent":  round(max(cpu), 1),
            "avg_gpu_util_pct": round(sum(gpu) / len(gpu), 1) if gpu else None,
            "max_gpu_util_pct": round(max(gpu), 1) if gpu else None,
            "violations": [
                {"kind": v.kind, "started_t_s": round(v.started_t_s, 1),
                 "ended_t_s": round(v.ended_t_s, 1) if v.ended_t_s is not None else None,
                 "phase": v.phase}
                for v in self.violations
            ],
        }
