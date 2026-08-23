"""
Probes CPU, RAM, GPU, and disk availability for cache/worker decisions, and
(via PipelineMonitor below) continuous background monitoring of the same
resources during a real run.
"""
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

GPU_PRESSURE_THRESHOLD = 0.75

# Non-fork workers (spawn: Windows always, macOS default; also forkserver) re-import torch/CUDA per worker instead of inheriting the parent's already-loaded copy; measured ~1.6GB committed bytes per worker on Windows.
SPAWN_WORKER_RELOAD_OVERHEAD_GB = 1.6


def worker_start_method() -> str:
    return mp.get_context().get_start_method()


def scale_workers_by_fraction(num_workers: int, fraction: float) -> int:
    if fraction <= 0.0:
        return 0  # a deliberate 0 must stay 0, not floor to 1
    return max(1, int(num_workers * fraction)) if fraction < 1.0 else num_workers


def apply_worker_cap(num_workers: int, max_workers: int, reason: str) -> int:
    """Clamps num_workers to max_workers, logging `reason` only when this cap actually bound the result."""
    capped = max(0, min(num_workers, max_workers))
    if capped < num_workers:
        logger.info(f"[PIPELINE] Worker count capped by {reason} -> using {capped} workers")
    return capped


def shared_memory_budget_bytes(available_ram_gb: float, worker_ram_fraction: float) -> float:
    """Caps the RAM-fraction budget by real /dev/shm free space (small and RAM-independent in a default Docker container); falls back to the RAM budget where /dev/shm doesn't exist (Windows)."""
    ram_budget = available_ram_gb * worker_ram_fraction * (1024 ** 3)
    try:
        shm_free = shutil.disk_usage("/dev/shm").free
        return min(ram_budget, shm_free)
    except OSError:
        return ram_budget


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
    """VRAM utilisation as 0–1. 0 when no GPU is present."""
    if metrics.gpu_memory_gb == 0:
        return 0.0
    return 1.0 - (metrics.gpu_available_gb / metrics.gpu_memory_gb)


def is_gpu_under_pressure(metrics: CacheMetrics, threshold: float = GPU_PRESSURE_THRESHOLD) -> bool:
    return gpu_pressure(metrics) > threshold


# GPU VRAM safety margin per phase — diagnostic only, see
GPU_PHASE_MARGINS: dict[str, float] = {
    "training": 0.20,
    "testing":  0.10,
}


def gpu_total_memory_gb() -> float:
    """Total VRAM for the GPU"""
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


class SystemResourceMonitor:
    """Live RAM/GPU/disk probe, plus phase-aware VRAM headroom diagnostics
    at the training/testing boundary (see enter_phase()). One instance,
    `monitor` below, is built once at import time and shared: every file
    that needs live resource state imports that instance directly
    (`from .system_monitor import monitor`) and calls its methods, instead
    of each building its own probe and re-querying the same RAM/disk/VRAM
    state independently."""

    def __init__(self, cache_path: str = "./cache", cuda_enabled: bool = True):
        self.cache_path = Path(cache_path)
        # False forces GPU fields to 0 even when CUDA is physically present —
        # set when the run is explicitly CPU-only, so a GPU that exists but
        # isn't requested never affects cache/worker decisions.
        self.cuda_enabled = cuda_enabled and torch.cuda.is_available()
        self.phase = "training"  # see enter_phase()

    def configure(self, cache_path: str = "./cache", cuda_enabled: bool = True) -> None:
        """Set the run's cache_path once, early, as soon as it's known
        (NeuromorphicEncoder.__init__). Unconfigured, the constructor
        defaults above apply."""
        self.cache_path = Path(cache_path)
        self.cuda_enabled = cuda_enabled and torch.cuda.is_available()

    def snapshot(self) -> CacheMetrics:
        vm = psutil.virtual_memory()

        try:
            disk_stat = shutil.disk_usage(self.cache_path)
            disk_available = disk_stat.free / (1024 ** 3)
            disk_exists = True
        except Exception:
            disk_available = 0.0
            disk_exists = False

        if self.cuda_enabled:
            gpu_total = gpu_total_memory_gb()

            # Most conservative available-VRAM estimate: driver-reported free
            # space vs. total minus PyTorch's own reserved pool, whichever is smaller.
            free_driver    = torch.cuda.mem_get_info(0)[0]
            reserved       = torch.cuda.memory_reserved(0)
            total_bytes    = torch.cuda.get_device_properties(0).total_memory
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
        )

    def worker_count(self, device: torch.device, safety_margin_gb: float = 2.0, worker_fraction: float = 1.0,
                     worker_count_override: int | None = None, metrics: CacheMetrics | None = None) -> tuple[int, CacheMetrics, int]:
        """The bytes_per_batch-independent subset of dataloader_config()'s sizing (physical-core baseline, worker_fraction, spawn-reload cap, override) -- reusable wherever num_workers is needed before the real batch size is known, without building/logging a full DataLoader kwargs dict for it."""
        metrics = metrics or self.snapshot()
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        num_workers = physical_cores if metrics.available_ram_gb >= safety_margin_gb else 0
        if num_workers > 0:
            num_workers = scale_workers_by_fraction(num_workers, worker_fraction)

        if worker_count_override is not None:
            num_workers = max(0, min(physical_cores, scale_workers_by_fraction(worker_count_override, worker_fraction)))
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
                          worker_fraction: float = 1.0, worker_count_override: int | None = None,
                          metrics: CacheMetrics | None = None) -> dict:
        """num_workers/prefetch/pin_memory/persistent_workers, capped from physical-core count (scaled by worker_fraction) by spawn-reload overhead, /dev/shm, and bytes_per_batch, unless worker_count_override opts out of those RAM-based caps entirely."""
        cuda_enabled = device is not None and getattr(device, "type", "") == "cuda"
        num_workers, metrics, physical_cores = self.worker_count(device, safety_margin_gb, worker_fraction, worker_count_override, metrics)
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
            cfg = {
                "num_workers":        0,
                "prefetch_factor":    None,
                "pin_memory":         False,
                "persistent_workers": False,
                "timeout":            0,
            }
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
                "timeout":            worker_timeout_s if num_workers > 0 else 0,  # a stuck worker raises instead of hanging the run forever
            }

        logger.info(f"[PIPELINE] DataLoader config: {cfg}")
        return cfg

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
    Continuous background sampler for CPU + GPU utilization/power/clock/
    memory + RAM, shared by the offline diagnostics harness
    (diagnostics/gpu_utilization_harness.py) and by real training/inference
    runs (SNNTrainer, SNNTester) — the same probing SystemResourceMonitor
    does point-in-time for cache decisions, extended into a running, phase-
    taggable trace with sustained-violation ("out of bounds") flagging.

    Sampling is NVML/psutil driver queries only — never touches a CUDA
    tensor or calls torch.cuda.synchronize() — so it never perturbs the
    training loop it's watching.

    This project only ever runs single-GPU (training.device in
    SNN_module.yaml is cpu | cuda | auto — never an indexed cuda:N), so
    there's no device index to take here — it's always device 0.

    Usage — free-form phases (diagnostics harness):
        pm = PipelineMonitor()
        pm.start()
        pm.set_phase("fetch")
        ...
        pm.set_phase("compute")
        ...
        pm.stop()
        for v in pm.violations: ...   # sustained out-of-bounds episodes

    Usage — one phase per epoch (SNNTrainer/SNNTester), reusing the same
    phase-tagged sample stream to also report per-epoch VRAM/energy:
        pm = PipelineMonitor()
        pm.start()
        pm.measure_idle_baseline()
        pm.set_phase(f"epoch_{epoch}")
        pm.reset_epoch_memory()
        ... run the epoch ...
        gpu = pm.phase_summary(f"epoch_{epoch}")
        energy_j = pm.phase_energy_j(f"epoch_{epoch}", epoch_duration_s)
        pm.stop()
        overall = pm.summary()  # across every phase since start()
    """

    def __init__(
        self,
        cuda_enabled: bool = True,
        interval_s: float = 0.2,
        thresholds: BoundThresholds | None = None,
        enabled: bool = True,
    ):
        self.enabled        = enabled
        self.cuda_enabled  = cuda_enabled and torch.cuda.is_available()
        self.interval_s    = interval_s
        self.thresholds    = thresholds or BoundThresholds()

        self.total_memory_gb = gpu_total_memory_gb() if self.cuda_enabled else 0.0
        self.peak_mem_each: list[float] = []  # one entry per phase_summary() call
        self.idle_power_w: float | None = None

        self.nvml_handle = None
        if self.cuda_enabled:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.nvml_handle = None

        self.samples: list[PipelineSample] = []
        self.violations: list[BoundViolation] = []
        self.phase = "unset"
        self.t_origin: float | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        # in-progress violation start time per kind, or None if currently within bounds
        self.open_violation: dict[str, BoundViolation | None] = {
            "gpu_idle": None, "cpu_saturated": None, "ram_low": None,
        }

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def reset_epoch_memory(self) -> None:
        """Reset PyTorch's peak-VRAM counter. Call at the start of a phase
        (e.g. an epoch) so phase_summary()'s memory figures reflect only
        that phase, not everything since start()."""
        if self.cuda_enabled:
            torch.cuda.reset_peak_memory_stats(0)

    def epoch_memory_gb(self) -> float:
        """Peak VRAM (GB) since the last reset_epoch_memory() call."""
        if not self.cuda_enabled:
            return 0.0
        return torch.cuda.max_memory_allocated(0) / (1024 ** 3)

    def measure_idle_baseline(self, duration_s: float = 1.0) -> float | None:
        """Sample GPU power for a short window with no work scheduled,
        before a run starts. Lets phase_energy_j()/dynamic_power_w() report
        *dynamic* power/energy (above idle draw) instead of the raw total,
        which otherwise overstates what a workload actually costs — the GPU
        pulls non-zero power just sitting idle. No-op (returns None)
        without CUDA/NVML. Safe to call more than once; last call wins."""
        if not self.enabled or not self.cuda_enabled or self.nvml_handle is None:
            return None
        deadline = time.perf_counter() + duration_s
        samples = []
        while time.perf_counter() < deadline:
            try:
                samples.append(float(pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)))
            except Exception:
                pass
            time.sleep(self.interval_s / 2)
        if samples:
            self.idle_power_w = (sum(samples) / len(samples)) * 1e-3
        return self.idle_power_w

    def dynamic_power_w(self, avg_power_w: float) -> float:
        """avg_power_w with the idle baseline subtracted (clamped at 0).
        Falls back to avg_power_w unchanged if measure_idle_baseline() was
        never called."""
        if self.idle_power_w is None:
            return avg_power_w
        return max(0.0, avg_power_w - self.idle_power_w)

    def phase_summary(self, phase: str) -> dict:
        """Aggregate utilization for one phase's samples (e.g. one epoch),
        plus peak/current VRAM since the last reset_epoch_memory() call.
        Also records this call's peak VRAM into peak_mem_each, so summary()
        can report the overall peak across every phase later. gpu_idle_*
        rolls up every sustained gpu_idle episode logged for this phase
        (see check_bound()) into one count + total duration, instead of the
        caller having to read each episode out of self.violations."""
        samples  = [s for s in self.samples if s.phase == phase]
        gpu_vals = [s.gpu_util_pct for s in samples if s.gpu_util_pct is not None]
        peak_gb  = self.epoch_memory_gb()
        peak_pct = peak_gb / self.total_memory_gb * 100 if self.total_memory_gb > 0 else 0.0
        self.peak_mem_each.append(peak_gb)

        now = time.perf_counter() - (self.t_origin or 0.0)
        idle_episodes = [v for v in self.violations if v.kind == "gpu_idle" and v.phase == phase]
        open_idle = self.open_violation.get("gpu_idle")
        if open_idle is not None and open_idle.phase == phase and open_idle not in self.violations:
            idle_episodes = idle_episodes + [open_idle]  # still ongoing as of this call
        idle_total_s = sum((v.ended_t_s or now) - v.started_t_s for v in idle_episodes)

        return {
            "gpu_util_avg_pct":  round(sum(gpu_vals) / len(gpu_vals), 1) if gpu_vals else 0.0,
            "gpu_util_peak_pct": round(max(gpu_vals), 1) if gpu_vals else 0.0,
            "gpu_mem_peak_gb":   round(peak_gb, 2),
            "gpu_mem_peak_pct":  round(peak_pct, 1),
            "gpu_idle_episodes": len(idle_episodes),
            "gpu_idle_total_s":  round(idle_total_s, 1),
        }

    def phase_energy_j(self, phase: str, elapsed_s: float) -> float | None:
        """Estimated GPU energy in joules for one phase, from that phase's
        sampled power readings. None if no power data (no CUDA / no NVML)."""
        power_vals = [s.gpu_power_w for s in self.samples if s.phase == phase and s.gpu_power_w is not None]
        if not power_vals:
            return None
        avg_w = sum(power_vals) / len(power_vals)
        return avg_w * elapsed_s

    def runtime_diagnostics(self) -> dict:
        """Point-in-time GPU runtime diagnostics beyond phase_summary(): max
        memory *reserved* by PyTorch's caching allocator (the allocator's
        high-water mark, including memory held but not currently in use —
        distinct from max allocated), and — when NVML is available — GPU
        temperature and SM/memory clock speed. Single-GPU only (device 0) —
        see event_data_workflow/README.md's "Known Limitation" note.

        NVML/driver queries, not CUDA-stream operations, so unlike
        `.item()`/`.cpu()` this does NOT force a wait on kernel completion —
        safe to call once per epoch/test-run."""
        diag: dict = {
            "max_memory_reserved_gb": (
                torch.cuda.max_memory_reserved(0) / (1024 ** 3) if self.cuda_enabled else 0.0
            ),
        }
        if self.nvml_handle is not None:
            try:
                diag["gpu_temp_c"]    = pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                diag["sm_clock_mhz"]  = pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, pynvml.NVML_CLOCK_SM)
                diag["mem_clock_mhz"] = pynvml.nvmlDeviceGetClockInfo(self.nvml_handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                pass  # driver/permission hiccup — diagnostics are best-effort, never worth failing a run over
        return diag

    def phase_energy_report(self, phase: str, elapsed_s: float) -> dict:
        """One-call bundle of everything a caller needs to report a phase's
        GPU cost — utilization/memory summary, measured energy/power (idle
        baseline subtracted), and runtime diagnostics — instead of pulling
        phase_summary()/phase_energy_j()/dynamic_power_w()/
        runtime_diagnostics() together by hand at every call site."""
        gpu_energy_j = self.phase_energy_j(phase, elapsed_s)
        avg_power_w  = gpu_energy_j / elapsed_s if gpu_energy_j is not None else None
        return {
            "gpu":             self.phase_summary(phase),
            "gpu_energy_j":    gpu_energy_j,
            "avg_power_w":     avg_power_w,
            "dynamic_power_w": self.dynamic_power_w(avg_power_w) if avg_power_w is not None else None,
            "gpu_diag":        self.runtime_diagnostics(),
        }

    def start(self) -> None:
        self.t_origin = time.perf_counter()
        self.stop_event.clear()
        if not self.enabled:
            return
        psutil.cpu_percent(interval=None)  # prime the reference point (see snapshot()'s docstring note)
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        # close out any still-open violation episodes at the moment of stopping
        now = time.perf_counter() - (self.t_origin or 0.0)
        for kind, v in self.open_violation.items():
            if v is not None:
                v.ended_t_s = now

    def loop(self) -> None:
        while not self.stop_event.wait(self.interval_s):
            try:
                self.sample_once()
            except Exception:
                pass  # a monitor must never take down the run it's watching

    def sample_once(self) -> None:
        t = time.perf_counter() - self.t_origin
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
            t_s=t, phase=self.phase, cpu_percent=cpu_pct, ram_available_gb=ram_gb,
            gpu_util_pct=gpu_util, gpu_power_w=gpu_power, gpu_sm_clock_mhz=gpu_clock,
        ))

        # Skipped while phase=="unset": that's setup/caching before any real
        # phase has been entered, where an idle GPU is expected, not a fault.
        self.check_bound("gpu_idle", self.phase != "unset" and gpu_util is not None
                           and gpu_util < self.thresholds.gpu_idle_pct_below,
                           t, self.thresholds.gpu_idle_sustained_s)
        self.check_bound("cpu_saturated", cpu_pct > self.thresholds.cpu_saturated_pct_above,
                           t, self.thresholds.cpu_saturated_sustained_s)
        self.check_bound("ram_low", ram_gb < self.thresholds.ram_available_gb_below,
                           t, self.thresholds.ram_low_sustained_s)

    def check_bound(self, kind: str, in_violation: bool, t: float, sustained_s: float) -> None:
        open_v = self.open_violation[kind]
        if in_violation:
            if open_v is None:
                # Not flagged yet — just start tracking; only becomes a
                # logged BoundViolation once it's been sustained long enough.
                self.open_violation[kind] = BoundViolation(kind=kind, started_t_s=t, phase=self.phase)
            elif open_v not in self.violations and (t - open_v.started_t_s) >= sustained_s:
                # Recorded, not printed live — a real run can cross this every
                # few seconds, which floods the console without adding
                # information beyond what phase_summary()'s gpu_idle_episodes/
                # gpu_idle_total_s already rolls up once per epoch.
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
        """Aggregate stats across the whole run so far, plus the violation log."""
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
        """Host-RAM and peak-VRAM trend across every sample collected since
        start() — confirms usage stays flat/bounded across a run rather than
        climbing toward exhaustion. Reuses the same samples summary()/
        phase_summary() already collect; adds no new measurement."""
        if not self.samples:
            return {"n_samples": 0}
        ram_start = self.samples[0].ram_available_gb
        ram_end   = self.samples[-1].ram_available_gb
        ram_drift = ram_end - ram_start
        return {
            "n_samples":      len(self.samples),
            "ram_start_gb":   round(ram_start, 2),
            "ram_end_gb":     round(ram_end, 2),
            "ram_drift_gb":   round(ram_drift, 2),
            "ram_trend":      "stable" if abs(ram_drift) < 0.5 else ("declining" if ram_drift < 0 else "growing"),
            "gpu_peak_mem_gb": round(max(self.peak_mem_each), 2) if self.peak_mem_each else 0.0,
        }
