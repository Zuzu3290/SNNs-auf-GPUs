"""
Adaptive Cache Controller for Neuromorphic Data Pipelines
Dynamically balances between MemoryCachedDataset and DiskCachedDataset based on:
- Available RAM
- Disk availability
- Dataset size
- Access patterns
"""
from __future__ import annotations
import sys
import logging
import threading
import shutil
from abc import abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from tonic import DiskCachedDataset, MemoryCachedDataset

sys.path.insert(0, str(Path(__file__).parent))  # sibling modules (system_monitor)

from system_monitor import CacheMetrics, SystemResourceMonitor, is_gpu_under_pressure

logger = logging.getLogger(__name__)

# VRAM budget fractions per training phase — controls how aggressively the
# GPURecordingCache is allowed to grow relative to free VRAM.
# Lower fractions during computationally heavy phases (backward pass, warmup)
# prevent the cache from competing with activations, gradients, and optimizer state.
GPU_PHASE_CAPS: dict[str, float] = {
    "warmup":    0.05,  # activations and params not yet stable — keep cache tiny
    "train":     0.10,  # conservative: backward pass competes hard for VRAM
    "backward":  0.05,  # most dangerous moment — minimize cache footprint
    "eval":      0.25,  # no optimizer state or gradients active — more headroom
    "inference": 0.30,  # largest budget; no backward pass at all
}
GPU_EMERGENCY_MARGIN = 0.15  # fraction of total VRAM kept unconditionally free
GPU_MAX_CACHE_GB     = 2.0   # absolute ceiling regardless of free VRAM


def compute_gpu_cache_budget(free_vram_gb: float, total_vram_gb: float, phase: str = "train") -> float:
    """
    Strict VRAM budget for GPURecordingCache.

    VRAM owners in priority order (model/runtime take precedence):
        1. model parameters          5. temporary CUDA workspace
        2. forward activations       6. prefetch queue
        3. backward gradients        7. recording cache  ← lowest priority
        4. optimizer state

    Budget = min(free × phase_cap, free − emergency_margin, GPU_MAX_CACHE_GB)
    """
    emergency_gb    = total_vram_gb * GPU_EMERGENCY_MARGIN
    safe_cache_vram = free_vram_gb  - emergency_gb
    phase_cap       = GPU_PHASE_CAPS.get(phase, 0.10)
    budget          = min(free_vram_gb * phase_cap, safe_cache_vram, GPU_MAX_CACHE_GB)
    return max(0.0, budget)


@dataclass
class CacheStrategy:
    """Determined caching strategy based on system resources."""
    mode: Literal["memory", "disk", "hybrid", "gpu_memory", "no_cache"]
    memory_threshold_gb: float
    reason: str

    @property
    def memory_cache_enabled(self) -> bool:
        return self.mode in ("memory", "hybrid")

    @property
    def disk_cache_enabled(self) -> bool:
        return self.mode in ("disk", "hybrid")


def estimate_item_bytes(events) -> int:
    """Shared byte estimator for cached event tensors."""
    if hasattr(events, "element_size") and hasattr(events, "numel"):
        return int(events.element_size() * events.numel())
    if hasattr(events, "nbytes"):
        return int(events.nbytes)
    return 0


class BaseLRURecordingCache(Dataset):
    """
    LRU cache base for raw neuromorphic recordings.

    Subclasses override prepare_item() to transform a raw (events, target)
    before it enters the cache — CPU subclass is a no-op, GPU subclass moves
    tensors to CUDA.

    Eviction policy:
    - max_recordings: count-based LRU eviction (None = no count cap)
    - max_bytes: byte-based LRU eviction  (None = no byte cap)
    Both caps apply simultaneously when set.

    Multi-worker safety: lock rebuilds after multiprocessing unpickle.
    """

    def __init__(
        self,
        dataset: Dataset,
        max_recordings: Optional[int] = None,
        max_bytes: Optional[int] = None,
        transform=None,
    ):
        self.dataset = dataset
        self.max_recordings = max_recordings  # None = no count cap
        self.max_bytes = max_bytes            # None = no byte cap
        self.transform = transform
        self.cache: OrderedDict = OrderedDict()
        self.cache_bytes: int = 0
        self.lock = threading.Lock()

    @abstractmethod
    def prepare_item(self, raw):
        """Transform raw (events, target) before caching. Override in subclasses."""

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        with self.lock:
            if idx in self.cache:
                self.cache.move_to_end(idx)
                raw = self.cache[idx]
            else:
                raw = None

        if raw is None:
            raw = self.prepare_item(self.dataset[idx])
            events, _ = raw
            nb = estimate_item_bytes(events)
            with self.lock:
                if idx not in self.cache:
                    self.cache[idx] = raw
                    self.cache.move_to_end(idx)
                    self.cache_bytes += nb
                    if self.max_recordings is not None:
                        while len(self.cache) > self.max_recordings:
                            _, evicted = self.cache.popitem(last=False)
                            self.cache_bytes -= estimate_item_bytes(evicted[0])
                    if self.max_bytes is not None:
                        while self.cache_bytes > self.max_bytes and self.cache:
                            _, evicted = self.cache.popitem(last=False)
                            self.cache_bytes -= estimate_item_bytes(evicted[0])

        if self.transform is not None:
            events, target = raw
            return self.transform(events), target
        return raw

    @property
    def cache_size(self) -> int:
        with self.lock:
            return len(self.cache)

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.cache_bytes = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()


class BoundedRecordingCache(BaseLRURecordingCache):
    """
    CPU RAM LRU cache for raw recordings.

        cached_raw = BoundedRecordingCache(raw_dataset, max_recordings=500)
        sliced     = TemporalSlicedDataset(cached_raw, config)

    Why cache recordings, not slices:
    - N slices per recording → N cache hits from 1 stored entry (high reuse)
    - Memory is proportional to recordings, not recordings × slices
    - Avoids storing duplicate/overlapping event windows
    """

    def __init__(self, dataset: Dataset, max_recordings: int = 500, max_bytes: Optional[int] = None, transform=None):
        super().__init__(dataset, max_recordings=max_recordings, max_bytes=max_bytes, transform=transform)

    def prepare_item(self, raw):
        return raw  # CPU path — no transformation needed

    def __setstate__(self, state):
        super().__setstate__(state)
        self.cache_bytes = sum(estimate_item_bytes(v[0]) for v in self.cache.values())


class GPURecordingCache(BaseLRURecordingCache):
    """
    GPU VRAM LRU cache for GPU-only deployments (no disk, limited RAM).

    Budget policy — cache is the lowest-priority VRAM user:
        compute_gpu_cache_budget() enforces a phase cap and an emergency margin.
        set_phase() recalculates the budget immediately when the training phase
        changes (e.g. "train" → "eval") and evicts down to the new limit.
        evict_under_pressure() probes live VRAM every PRESSURE_CHECK_INTERVAL
        accesses and evicts LRU entries if free VRAM falls below the emergency
        margin (GPU_EMERGENCY_MARGIN × total VRAM).

    Design note for neuromorphic workloads:
        Full raw recordings may be too large for VRAM. Prefer applying this cache
        after temporal slicing (smaller chunks) or store encoded spike tensors
        rather than raw event arrays. See docs/Hardware/event_data_workflow.md.
    """

    PRESSURE_CHECK_INTERVAL = 50  # VRAM probe cadence (item accesses)

    def __init__(self, dataset: Dataset, device: torch.device, max_bytes: int, transform=None):
        super().__init__(dataset, max_recordings=None, max_bytes=max_bytes, transform=transform)
        self.device = device
        self.phase = "train"
        self.access_count = 0

    def prepare_item(self, raw):
        events, target = raw
        if isinstance(events, torch.Tensor):
            return events.to(self.device, non_blocking=True), target
        return torch.as_tensor(events, device=self.device), target

    def set_phase(self, phase: str) -> None:
        """
        Switch training phase and immediately enforce the new VRAM budget.

        Tighter phases (e.g. "backward") shrink max_bytes and evict LRU entries
        to free VRAM before the backward pass competes for it.
        Looser phases (e.g. "eval") expand max_bytes up to the new cap.

        Valid: warmup | train | backward | eval | inference
        """
        if phase not in GPU_PHASE_CAPS:
            raise ValueError(f"Unknown phase '{phase}'. Valid: {list(GPU_PHASE_CAPS)}")
        self.phase = phase
        if not torch.cuda.is_available():
            return
        device_idx    = self.device.index if self.device.index is not None else 0
        free_driver, total = torch.cuda.mem_get_info(device_idx)
        new_budget    = compute_gpu_cache_budget(
            free_driver / (1024 ** 3), total / (1024 ** 3), phase
        )
        new_max_bytes = int(new_budget * (1024 ** 3))
        with self.lock:
            self.max_bytes = new_max_bytes
            while self.cache_bytes > self.max_bytes and self.cache:
                _, evicted = self.cache.popitem(last=False)
                self.cache_bytes -= estimate_item_bytes(evicted[0])

    def __getitem__(self, idx: int):
        self.evict_under_pressure()
        return super().__getitem__(idx)

    def evict_under_pressure(self) -> None:
        """Evict LRU entries if VRAM free space falls below the emergency margin."""
        self.access_count += 1
        if self.access_count % self.PRESSURE_CHECK_INTERVAL != 0:
            return
        if not torch.cuda.is_available():
            return
        device_idx   = self.device.index if self.device.index is not None else 0
        free_driver, total = torch.cuda.mem_get_info(device_idx)
        free_gb      = free_driver / (1024 ** 3)
        emergency_gb = (total / (1024 ** 3)) * GPU_EMERGENCY_MARGIN
        if free_gb >= emergency_gb:
            return  # fast path — no pressure
        with self.lock:
            while self.cache and free_gb < emergency_gb:
                _, evicted = self.cache.popitem(last=False)
                self.cache_bytes -= estimate_item_bytes(evicted[0])
                free_driver, _ = torch.cuda.mem_get_info(device_idx)
                free_gb = free_driver / (1024 ** 3)


class AdaptiveCacheController:
    """
    Intelligent caching controller that monitors system resources and selects
    optimal caching strategy for neuromorphic datasets.

    Cache is applied to RAW recordings (before temporal slicing) so that
    all slices derived from one recording share a single cache entry.

    Strategy Selection Logic:
    1. If RAM > threshold and dataset fits → MemoryCachedDataset (full dataset in RAM)
    2. If RAM > 32GB and disk available → Hybrid (DiskCachedDataset primary + BoundedRecordingCache hot layer)
    3. If RAM < threshold and disk available → DiskCachedDataset (memory-safe, neuromorphic default)
    4. If no disk and insufficient RAM but VRAM >= 500MB → GPURecordingCache (compute_gpu_cache_budget, train-phase cap)
    5. If no resources available → No caching (on-the-fly)
    """

    def __init__(
        self,
        cache_path: str = "./cache",
        memory_safety_margin_gb: float = 2.0,
        memory_cache_threshold_gb: float = 8.0,
        max_cached_recordings: int = 500,
        verbose: bool = True,
        device=None,
    ):
        self.cache_path = Path(cache_path)
        self.memory_safety_margin = memory_safety_margin_gb
        self.memory_threshold = memory_cache_threshold_gb
        self.max_cached_recordings = max_cached_recordings
        self.verbose = verbose
        self.device = device
        self.device_idx = (
            (device.index or 0)
            if device is not None and getattr(device, "type", "") == "cuda"
            else 0
        )
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.monitor = SystemResourceMonitor(cache_path=str(self.cache_path), device_idx=self.device_idx)

    def get_system_metrics(self) -> CacheMetrics:
        return self.monitor.snapshot()

    def estimate_dataset_memory_footprint(self, dataset: Dataset, num_samples_to_probe: int = 10) -> float:
        """Estimate total dataset memory requirement by sampling. Returns estimated size in GB."""
        if len(dataset) == 0:
            return 0.0

        sample_indices = torch.randperm(len(dataset))[:min(num_samples_to_probe, len(dataset))]
        total_bytes = 0
        successful_probes = 0

        for idx in sample_indices:
            try:
                events, target = dataset[int(idx)]
                if hasattr(events, "nbytes"):
                    total_bytes += events.nbytes
                elif hasattr(events, "element_size"):
                    total_bytes += events.element_size() * events.numel()
                else:
                    total_bytes += sys.getsizeof(events)
                successful_probes += 1
            except Exception as e:
                if self.verbose:
                    logger.warning(f"[CACHE CONTROLLER] Could not probe sample {idx}: {e}")
                continue

        if successful_probes == 0:
            logger.warning("[CACHE CONTROLLER] All probes failed — assuming dataset size is 0")
            return 0.0

        return (total_bytes / successful_probes * len(dataset)) / (1024 ** 3)

    def determine_strategy(
        self,
        dataset: Dataset,
        force_mode: Optional[Literal["memory", "disk", "hybrid", "gpu_memory", "no_cache"]] = None,
    ) -> CacheStrategy:
        """Analyze system resources and dataset characteristics to determine optimal caching strategy."""
        metrics = self.monitor.snapshot()
        dataset_size_gb = self.estimate_dataset_memory_footprint(dataset)

        if self.verbose:
            self.log_diagnostics(metrics, dataset_size_gb)

        if force_mode:
            return self.create_forced_strategy(force_mode, metrics)

        available_for_cache = metrics.available_ram_gb - self.memory_safety_margin

        # GPU pressure guard: if GPU is using >75% of VRAM, CUDA's pinned-memory
        # allocator competes directly with the recording cache for physical RAM.
        if is_gpu_under_pressure(metrics) and metrics.disk_exists:
            return CacheStrategy(
                mode="disk",
                memory_threshold_gb=0.0,
                reason=(f"GPU under pressure ({gpu_pressure_pct(metrics):.0f}% VRAM used) — "
                        "disk cache chosen to avoid CUDA/RAM competition"),
            )

        if available_for_cache >= self.memory_threshold and dataset_size_gb < available_for_cache * 0.7:
            return CacheStrategy(
                mode="memory",
                memory_threshold_gb=available_for_cache,
                reason=f"Sufficient RAM available ({available_for_cache:.1f}GB free, dataset ~{dataset_size_gb:.1f}GB)",
            )

        if metrics.total_ram_gb >= 32.0 and metrics.disk_exists and metrics.disk_available_gb > dataset_size_gb * 1.5:
            return CacheStrategy(
                mode="hybrid",
                memory_threshold_gb=available_for_cache * 0.5,
                reason=f"Large RAM ({metrics.total_ram_gb:.1f}GB) with disk backup - using hybrid strategy",
            )

        if metrics.disk_exists and metrics.disk_available_gb > dataset_size_gb * 1.2:
            return CacheStrategy(
                mode="disk",
                memory_threshold_gb=0.0,
                reason=f"Limited RAM ({available_for_cache:.1f}GB) but disk available ({metrics.disk_available_gb:.1f}GB)",
            )

        # GPU-only fallback: no disk, insufficient RAM, but VRAM is available.
        # Budget is strictly computed — cache is the lowest-priority VRAM user.
        if metrics.gpu_available_gb >= 0.5:
            vram_budget_gb = compute_gpu_cache_budget(
                metrics.gpu_available_gb, metrics.gpu_memory_gb, phase="train"
            )
            return CacheStrategy(
                mode="gpu_memory",
                memory_threshold_gb=vram_budget_gb,
                reason=(
                    f"GPU-only: no disk, insufficient RAM — "
                    f"cache budget {vram_budget_gb:.2f} GB "
                    f"(train-phase cap {GPU_PHASE_CAPS['train']*100:.0f}%, "
                    f"{GPU_EMERGENCY_MARGIN*100:.0f}% emergency margin, "
                    f"from {metrics.gpu_available_gb:.1f} GB free)"
                ),
            )

        return CacheStrategy(
            mode="no_cache",
            memory_threshold_gb=0.0,
            reason=f"Insufficient resources - RAM: {available_for_cache:.1f}GB, Disk: {metrics.disk_available_gb:.1f}GB, GPU: {metrics.gpu_available_gb:.1f}GB",
        )

    def create_forced_strategy(self, mode: str, metrics: CacheMetrics) -> CacheStrategy:
        strategies = {
            "memory": CacheStrategy(
                mode="memory",
                memory_threshold_gb=metrics.available_ram_gb - self.memory_safety_margin,
                reason=f"Forced memory mode (available: {metrics.available_ram_gb:.1f}GB)",
            ),
            "disk": CacheStrategy(
                mode="disk",
                memory_threshold_gb=0.0,
                reason=f"Forced disk mode (available: {metrics.disk_available_gb:.1f}GB)",
            ),
            "hybrid": CacheStrategy(
                mode="hybrid",
                memory_threshold_gb=(metrics.available_ram_gb - self.memory_safety_margin) * 0.5,
                reason="Forced hybrid mode",
            ),
            "gpu_memory": CacheStrategy(
                mode="gpu_memory",
                memory_threshold_gb=compute_gpu_cache_budget(
                    metrics.gpu_available_gb, metrics.gpu_memory_gb, phase="train"
                ),
                reason=(
                    f"Forced GPU memory mode — train-phase cap "
                    f"({GPU_EMERGENCY_MARGIN*100:.0f}% emergency margin applied)"
                ),
            ),
            "no_cache": CacheStrategy(
                mode="no_cache",
                memory_threshold_gb=0.0,
                reason="Forced no-cache mode (on-the-fly processing)",
            ),
        }
        return strategies[mode]

    def wrap_dataset(
        self,
        dataset: Dataset,
        transform=None,
        split: str = "train",
        strategy: Optional[CacheStrategy] = None,
        num_workers: int = 1,
    ) -> Dataset:
        """
        Wrap dataset with appropriate caching mechanism based on strategy.
        Cache must be applied to raw recordings BEFORE temporal slicing.

        num_workers: number of DataLoader workers that will share this cache.
        With persistent_workers=True each worker holds its own copy of the cache,
        so the byte budget is divided by num_workers to prevent RAM overcommit.
        """
        if hasattr(dataset, "slice_map"):
            raise ValueError(
                "wrap_dataset() received an already-sliced dataset. "
                "Cache must be applied to raw recordings BEFORE slicing — "
                "use: cached_raw = wrap_dataset(raw_dataset); "
                "sliced = TemporalSlicedDataset(cached_raw, config)"
            )

        if strategy is None:
            strategy = self.determine_strategy(dataset)

        cache_dir = self.cache_path / split
        cache_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            logger.info(f"[CACHE CONTROLLER] Applying {strategy.mode.upper()} strategy for {split} split")
            logger.info(f"[CACHE CONTROLLER] Reason: {strategy.reason}")

        if strategy.mode == "memory":
            if self.verbose:
                logger.info(f"[CACHE CONTROLLER] Using MemoryCachedDataset → full dataset in RAM ({strategy.memory_threshold_gb:.1f} GB available)")
            return MemoryCachedDataset(dataset, transform=transform)

        elif strategy.mode == "disk":
            if self.verbose:
                logger.info(f"[CACHE CONTROLLER] Using DiskCachedDataset → Path: {cache_dir}")
            return DiskCachedDataset(dataset, transform=transform, cache_path=str(cache_dir))

        elif strategy.mode == "hybrid":
            # Each persistent DataLoader worker gets its own copy of the cache dict in its
            # subprocess, so true RAM consumption is max_bytes × num_workers. Divide the
            # budget so the total across all workers stays within the strategy's allocation.
            effective_workers = max(1, num_workers)
            max_bytes = int(strategy.memory_threshold_gb * (1024 ** 3)) // effective_workers
            if self.verbose:
                logger.info(
                    f"[CACHE CONTROLLER] Using Hybrid → DiskCachedDataset (primary) + "
                    f"BoundedRecordingCache ({strategy.memory_threshold_gb:.1f} GB ÷ {effective_workers} workers "
                    f"= {max_bytes / (1024**3):.2f} GB per worker hot layer)"
                )
            disk_cached = DiskCachedDataset(dataset, transform=transform, cache_path=str(cache_dir))
            return BoundedRecordingCache(disk_cached, max_recordings=self.max_cached_recordings, max_bytes=max_bytes)

        elif strategy.mode == "gpu_memory":
            if self.device is None or getattr(self.device, "type", "") != "cuda":
                logger.warning("[CACHE CONTROLLER] gpu_memory strategy selected but no CUDA device set — falling back to no_cache")
                return dataset
            max_bytes = int(strategy.memory_threshold_gb * (1024 ** 3))
            if self.verbose:
                logger.info(f"[CACHE CONTROLLER] Using GPURecordingCache → {strategy.memory_threshold_gb:.1f} GB VRAM budget on {self.device}")
            return GPURecordingCache(dataset, device=self.device, max_bytes=max_bytes, transform=transform)

        else:  # no_cache
            if self.verbose:
                logger.info("[CACHE CONTROLLER] No caching → On-the-fly processing (may be slower)")
            return dataset

    def log_diagnostics(self, metrics: CacheMetrics, dataset_size_gb: float):
        sep = "=" * 70
        logger.info(sep)
        logger.info("ADAPTIVE CACHE CONTROLLER - SYSTEM DIAGNOSTICS")
        logger.info(sep)
        logger.info(f"  Total RAM        : {metrics.total_ram_gb:.2f} GB")
        logger.info(f"  Available RAM    : {metrics.available_ram_gb:.2f} GB")
        logger.info(f"  RAM Usage        : {metrics.ram_usage_percent:.1f}%")
        logger.info(f"  Disk Available   : {'YES' if metrics.disk_exists else 'NO'}")
        if metrics.disk_exists:
            logger.info(f"  Free Disk Space  : {metrics.disk_available_gb:.2f} GB")
        if metrics.gpu_memory_gb > 0:
            logger.info(f"  GPU Memory       : {metrics.gpu_memory_gb:.2f} GB")
            logger.info(f"  GPU Available    : {metrics.gpu_available_gb:.2f} GB")
        else:
            logger.info("  GPU              : Not available or not detected")
        logger.info(f"  Est. Dataset     : ~{dataset_size_gb:.2f} GB")
        logger.info(f"  Safety Margin    : {self.memory_safety_margin:.2f} GB (reserved for system)")
        logger.info(sep)

    def clear_cache(self, split: Optional[str] = None):
        """Clear disk cache for specified split or all splits."""
        if split:
            cache_dir = self.cache_path / split
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
                if self.verbose:
                    logger.info(f"[CACHE CONTROLLER] Cleared cache for split: {split}")
        else:
            if self.cache_path.exists():
                shutil.rmtree(self.cache_path)
                self.cache_path.mkdir(parents=True, exist_ok=True)
                if self.verbose:
                    logger.info("[CACHE CONTROLLER] Cleared all cache directories")


def gpu_pressure_pct(metrics: CacheMetrics) -> float:
    """GPU VRAM usage as a percentage (0–100)."""
    if metrics.gpu_memory_gb == 0:
        return 0.0
    return (1.0 - metrics.gpu_available_gb / metrics.gpu_memory_gb) * 100


def auto_cache_dataset(
    dataset: Dataset,
    transform=None,
    split: str = "train",
    cache_path: str = "./cache",
    verbose: bool = True,
    device=None,
) -> Dataset:
    """
    Convenience function to automatically select and apply optimal caching.

    Pass device=torch.device("cuda:0") to enable GPURecordingCache fallback
    when no disk or RAM is available.

    Usage:
        trainset = tonic.datasets.NMNIST(save_to="./data", train=True)
        cached_trainset = auto_cache_dataset(trainset, transform=my_transform, split="train", device=device)
    """
    controller = AdaptiveCacheController(cache_path=cache_path, verbose=verbose, device=device)
    return controller.wrap_dataset(dataset, transform=transform, split=split)
