"""
Pipeline Memory Coordinator and Spike Buffer for Neuromorphic Pipelines

Responsibilities:
- PipelineMemoryBudget      : RAM fraction splits across pipeline stages
- PipelineMemoryCoordinator : enforces the budget, detects GPU pressure,
                               recommends DataLoader config
- SparseEventBuffer         : per-timestep spike accumulation during SNN forward passes
"""
from __future__ import annotations
import os
import sys
import logging
import threading
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton package
sys.path.insert(0, str(Path(__file__).parent))          # sibling modules

from skeleton.snn_config import Settings
from system_monitor import SystemResourceMonitor, gpu_pressure as _gpu_pressure, is_gpu_under_pressure

logger = logging.getLogger(__name__)


@dataclass
class PipelineMemoryBudget:
    """Explicit RAM split across pipeline stages."""
    total_gb: float                        # Available RAM budget for the pipeline (system RAM minus safety margin)
    recording_cache_fraction: float = 0.4  # BoundedRecordingCache
    worker_fraction: float = 0.3           # DataLoader worker buffers
    prefetch_fraction: float = 0.3         # AsyncGPUPrefetcher queue


class PipelineMemoryCoordinator:
    """
    Coordinates RAM budget across all pipeline components so no single stage
    can silently overflow into another's share.

    Automatically halves the recording-cache allocation when GPU memory usage
    exceeds gpu_pressure_threshold — CUDA pinned memory and the recording cache
    both draw from the same physical RAM, so allowing both to grow unchecked
    causes OOM or pipeline stalls.

    Note: tracks a single GPU only. For multi-GPU setups, create one coordinator
    per device. See event_data_workflow/README.md for details.

    Usage:
        coord    = PipelineMemoryCoordinator.from_system(settings)
        max_recs = coord.max_recordings(avg_recording_bytes=50_000)
        dl_cfg   = coord.dataloader_config(batch_bytes=2_000_000)
        q_size   = coord.prefetch_queue_size()
    """

    GPU_PRESSURE_THRESHOLD = 0.75
    # When the GPU has consumed 75% or more of its total VRAM, the recording cache allocation is halved — preventing the model and cache from competing for the remaining VRAM and crashing with an out-of-memory error

    def __init__(self, budget: PipelineMemoryBudget, settings: Settings, verbose: bool = True, device=None):
        self.budget = budget      # carries the RAM fractions for each pipeline stage and the total RAM budget
        self.settings = settings  # SNN_module.yaml config — used to read NUM_WORKERS and other training params
        self.verbose = verbose    # logging messages about allocation decisions and GPU pressure
        self.device = device

        # CUDA is always required — either with CPU workers (Scenario 1: CUDA + CPU)
        # or GPU-only with no CPU worker headroom (Scenario 2: CUDA only).
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PipelineMemoryCoordinator requires a CUDA device. "
                "No CUDA-capable GPU was detected."
            )

        self.device_idx = (
            (device.index or 0)
            if device is not None and getattr(device, "type", "") == "cuda"
            else 0
        )

    @classmethod
    def from_system(cls, settings: Settings, safety_margin_gb: float = 2.0, verbose: bool = True, device=None) -> "PipelineMemoryCoordinator":
        """Build coordinator from live system metrics via SystemResourceMonitor."""
        device_idx = (device.index or 0) if device is not None and getattr(device, "type", "") == "cuda" else 0
        metrics = SystemResourceMonitor(device_idx=device_idx).snapshot()
        total_gb = max(1.0, metrics.available_ram_gb - safety_margin_gb)
        coord = cls(PipelineMemoryBudget(total_gb=total_gb), settings=settings, verbose=verbose, device=device)
        if verbose:
            logger.info(f"[MEMORY COORDINATOR] Pipeline budget: {total_gb:.1f} GB")
            coord.log_allocation()
        return coord

    def gpu_pressure(self) -> float:
        """Live GPU VRAM utilisation as 0–1 via SystemResourceMonitor."""
        metrics = SystemResourceMonitor(device_idx=self.device_idx).snapshot()
        return _gpu_pressure(metrics)

    def max_cache_bytes(self) -> int:
        """Byte budget for BoundedRecordingCache derived from the effective cache GB."""
        return int(self.effective_cache_gb() * (1024 ** 3))

    def is_gpu_under_pressure(self) -> bool:
        metrics = SystemResourceMonitor(device_idx=self.device_idx).snapshot()
        return is_gpu_under_pressure(metrics)

    def effective_cache_gb(self) -> float:
        fraction = self.budget.recording_cache_fraction
        if self.is_gpu_under_pressure():
            fraction *= 0.5
        return self.budget.total_gb * fraction  # actual cache budget in GB for the model

    def max_recordings(self, avg_recording_bytes: int) -> int:
        """Compute max_recordings cap for BoundedRecordingCache."""
        result = max(50, int(self.effective_cache_gb() * (1024 ** 3) / avg_recording_bytes))
        if self.verbose:
            logger.info(
                f"[MEMORY COORDINATOR] max_recordings={result} "
                f"(cache={self.effective_cache_gb():.2f}GB, avg={avg_recording_bytes // 1024}KB/rec)"
            )
        return result

    def dataloader_config(self, batch_bytes: int = 0) -> dict:
        """Recommend num_workers, prefetch_factor, pin_memory, persistent_workers.

        num_workers is read from settings.NUM_WORKERS (training.num_workers in SNN_module.yaml).
        Always capped by os.cpu_count() and the available RAM worker budget.

        GPU-only mode (no CPU workers) is activated when the available RAM budget
        for workers is less than 500 MB — typical of GPU-only embedded deployments
        where there is no host memory headroom for multiprocessing DataLoader workers.
        In that case num_workers=0 avoids forking and pin_memory is disabled since
        there is no CPU→GPU transfer boundary to optimise.
        """
        gpu    = torch.cuda.is_available()
        on_gpu = (self.device is not None and getattr(self.device, "type", "") == "cuda")
        worker_budget_gb = self.budget.total_gb * self.budget.worker_fraction
        gpu_only = on_gpu and worker_budget_gb < 0.5  # <500 MB RAM → GPU-only embedded

        if gpu_only:
            cfg = {
                "num_workers":        0,
                "prefetch_factor":    None,
                "pin_memory":         False,  # no H2D copy boundary in GPU-only path
                "persistent_workers": False,
            }
            if self.verbose:
                logger.info("[MEMORY COORDINATOR] GPU-only mode — num_workers=0, pin_memory=False")
        else:
            worker_bytes = worker_budget_gb * (1024 ** 3)
            if batch_bytes > 0:
                max_workers = max(1, int(worker_bytes / (2 * batch_bytes)))
            else:
                max_workers = self.settings.NUM_WORKERS
            max_workers = min(max_workers, os.cpu_count() or 1)
            cfg = {
                "num_workers":        max_workers,
                "prefetch_factor":    2 if max_workers > 0 else None,
                # pin_memory allocates page-locked host memory so CPU→GPU transfers
                # can run asynchronously via non_blocking=True in the training loop.
                "pin_memory":         gpu,
                "persistent_workers": max_workers > 0,
            }

        if self.verbose:
            logger.info(f"[MEMORY COORDINATOR] DataLoader config: {cfg}")
        return cfg

    def prefetch_queue_size(self) -> int:
        """Queue depth for AsyncGPUPrefetcher — shrinks under GPU pressure."""
        return 1 if self.is_gpu_under_pressure() else 2

    def log_allocation(self):
        gp = self.gpu_pressure()
        logger.info(f"  Recording cache : {self.effective_cache_gb():.2f} GB")
        logger.info(f"  Worker buffers  : {self.budget.total_gb * self.budget.worker_fraction:.2f} GB")
        logger.info(f"  Prefetch queue  : {self.budget.total_gb * self.budget.prefetch_fraction:.2f} GB")
        if gp > 0:
            note = " (halved — GPU pressure)" if self.is_gpu_under_pressure() else ""
            logger.info(f"  GPU pressure    : {gp * 100:.0f}%{note}")


class DenseTimestepBuffer:
    """
    Per-timestep spike accumulation buffer for SNN forward passes.

    Stores detached dense tensors per timestep to avoid CUDA sync in the hot path.
    stack() reconstructs [T, B, ...] with a single torch.stack call for loss/metrics.

    Interface:
        buf.push(spk)  — call once per timestep in a forward hook
        buf.stack()    — reconstruct dense [T, B, ...] when needed for loss/metrics
        buf.clear()    — reset between forward passes

    Multiprocessing-safe: lock is rebuilt after unpickling.
    Works with CPU and CUDA tensors.
    """

    def __init__(self) -> None:
        self.events: List[torch.Tensor] = []     # per-timestep dense spike tensors
        self.step_shape: Optional[tuple] = None  # shape of one timestep e.g. (B, N) or (B, C, H, W)
        self.lock = threading.Lock()

    def push(self, spk: torch.Tensor) -> None:
        """Record one timestep's spike tensor. Stored dense to avoid CUDA sync in the hot path."""
        tensor = spk.detach()
        with self.lock:
            if self.step_shape is None:
                self.step_shape = tuple(spk.shape)
            self.events.append(tensor)

    def stack(self) -> Optional[torch.Tensor]:
        """Return [T, B, ...] dense tensor stacked from stored timesteps. Returns None if push() was never called."""
        with self.lock:
            if not self.events:
                return None
            return torch.stack(self.events)  # single CUDA op — no Python loop

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
        """Bytes used by stored spike tensors."""
        with self.lock:
            return sum(e.element_size() * e.numel() for e in self.events)

    @property
    def firing_rate(self) -> float:
        """Fraction of possible spike slots that fired (0.0 – 1.0). For diagnostics only — .item() forces a CUDA sync per timestep."""
        with self.lock:
            if not self.events or self.step_shape is None:
                return 0.0
            total_per_step = 1
            for d in self.step_shape:
                total_per_step *= d
            total = total_per_step * len(self.events)
            fired = int(sum(e.sum().item() for e in self.events))
            return fired / total if total > 0 else 0.0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()
