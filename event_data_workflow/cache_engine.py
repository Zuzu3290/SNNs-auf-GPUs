"""
Picks a caching strategy (RAM, disk, hybrid, or GPU VRAM) for neuromorphic
recordings based on live system resources, and provides the cache classes
that back each strategy.
"""
from __future__ import annotations
import sys
import logging
import threading
import shutil
from abc import abstractmethod
from collections import deque
from pathlib import Path
from typing import Optional, Literal
import torch
from torch.utils.data import Dataset
from tonic import DiskCachedDataset, MemoryCachedDataset
from .system_monitor import CacheMetrics, SystemResourceMonitor, is_gpu_under_pressure

logger = logging.getLogger(__name__)


def measure_event_bytes(events) -> int:
    """Byte size of one sample, for cache byte-budget accounting."""
    if hasattr(events, "nbytes"):
        return events.nbytes
    if hasattr(events, "element_size"):
        return events.element_size() * events.numel()
    return sys.getsizeof(events)


class _ComposedTransform:
    """Picklable stand-in for a closure — Windows' spawn-based multiprocessing
    can't pickle nested functions (only importable module-level classes/functions),
    which broke DataLoader workers (num_workers > 0) the moment a DiskCachedDataset
    holding a closure-based transform got sent to a worker process."""
    def __init__(self, fns):
        self.fns = fns

    def __call__(self, x):
        for f in self.fns:
            x = f(x)
        return x


def compose_transforms(*fns):
    """Chain callables left to right, skipping any that are None."""
    fns = [f for f in fns if f is not None]
    if not fns:
        return None
    if len(fns) == 1:
        return fns[0]
    return _ComposedTransform(fns)


# How large a slice of free VRAM the GPU cache may use, per training phase.
# Smaller during backward/warmup (competing hard for VRAM), larger during
# eval/inference (no gradients or optimizer state active).
GPU_PHASE_CAPS: dict[str, float] = {
    "warmup":    0.05,
    "train":     0.10,
    "backward":  0.05,
    "eval":      0.25,
    "inference": 0.30,
}
GPU_EMERGENCY_MARGIN = 0.15  # fraction of total VRAM always kept free
GPU_MAX_CACHE_GB     = 2.0   # hard ceiling regardless of free VRAM


def compute_gpu_cache_budget(free_vram_gb: float, total_vram_gb: float, phase: str = "train") -> float:
    """VRAM budget for the GPU cache: the smallest of the phase cap, the
    emergency margin, and the hard ceiling."""
    emergency_gb    = total_vram_gb * GPU_EMERGENCY_MARGIN
    safe_cache_vram = free_vram_gb  - emergency_gb
    phase_cap       = GPU_PHASE_CAPS.get(phase, 0.10)
    budget          = min(free_vram_gb * phase_cap, safe_cache_vram, GPU_MAX_CACHE_GB)
    return max(0.0, budget)


class BaseRecordingCache(Dataset):
    """
    Bounded FIFO cache for raw recordings: caches up to max_recordings /
    max_bytes, evicting the oldest entry once full. Subclasses implement
    prepare_item() to decide what actually gets stored (CPU passthrough vs.
    GPU-resident tensor).
    """

    # True on subclasses holding device-resident state (live CUDA tensors)
    # that can't be shared with a separate DataLoader worker process.
    requires_single_process_loading = False

    def __init__(self, dataset: Dataset, max_recordings: Optional[int] = None, max_bytes: Optional[int] = None, transform=None):
        self.dataset        = dataset
        self.max_recordings = max_recordings
        self.max_bytes      = max_bytes
        self.transform      = transform  # applied fresh on every access, cache hit or miss

        self.cache: dict[int, tuple] = {}
        self.order: deque[int] = deque()  # insertion order, oldest at the left
        self.cache_bytes: int  = 0

        self.lock = threading.Lock()

    @abstractmethod
    def prepare_item(self, raw):
        """Transform a raw (events, target) pair before it's cached."""

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        with self.lock:
            raw = self.cache.get(idx)

        if raw is None:
            raw = self.prepare_item(self.dataset[idx])
            events, _ = raw
            nb = measure_event_bytes(events)
            with self.lock:
                if idx not in self.cache:
                    self.insert_item(idx, raw, nb)
                else:
                    raw = self.cache[idx]  # another worker inserted it first

        if self.transform is not None:
            events, target = raw
            return self.transform(events), target
        return raw

    def insert_item(self, idx: int, raw: tuple, nb: int) -> None:
        while self.over_capacity(nb) and self.order:
            self.evict_one()

        self.cache[idx]   = raw
        self.cache_bytes += nb
        self.order.append(idx)

    def over_capacity(self, incoming_bytes: int) -> bool:
        if self.max_recordings is not None and len(self.cache) >= self.max_recordings:
            return True
        if self.max_bytes is not None and self.cache_bytes + incoming_bytes > self.max_bytes:
            return True
        return False

    def evict_one(self) -> bool:
        if not self.order:
            return False
        idx = self.order.popleft()
        nb = measure_event_bytes(self.cache[idx][0])
        del self.cache[idx]
        self.cache_bytes -= nb
        return True

    @property
    def cache_size(self) -> int:
        with self.lock:
            return len(self.cache)

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.cache_bytes = 0
            self.order.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["lock"] = None  # locks aren't picklable — rebuilt on unpickle
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()


class BoundedRecordingCache(BaseRecordingCache):
    """CPU RAM cache tier (hybrid mode): sits in front of a DiskCachedDataset
    as a bounded hot layer. Stores whatever it's given, unchanged."""

    def __init__(self, dataset: Dataset, max_recordings: int = 500, max_bytes: Optional[int] = None, transform=None):
        super().__init__(dataset, max_recordings=max_recordings, max_bytes=max_bytes, transform=transform)

    def prepare_item(self, raw):
        return raw


class GPURecordingCache(BaseRecordingCache):
    """
    GPU VRAM cache tier: stores the encoded frame tensor directly on the
    device. encode_transform (deterministic, e.g. Denoise+ToFrame) runs once
    per recording and is baked into the cached value. live_transform
    (stochastic, e.g. random rotation) runs fresh on every access instead,
    so it doesn't get frozen into the cache after a recording's first touch.
    """

    requires_single_process_loading = True  # holds live CUDA tensors

    PRESSURE_CHECK_INTERVAL = 50  # VRAM probe cadence, in accesses

    def __init__(self, dataset: Dataset, device: torch.device, max_bytes: int, encode_transform=None, live_transform=None):
        super().__init__(dataset, max_recordings=None, max_bytes=max_bytes, transform=live_transform)
        self.device = device
        self.phase = "train"
        self.access_count = 0
        self._encode = encode_transform

    def prepare_item(self, raw):
        events, target = raw
        if self._encode is not None:
            events = self._encode(events)
        if isinstance(events, torch.Tensor):
            return events.to(self.device, non_blocking=True), target
        return torch.as_tensor(events, device=self.device), target

    def set_phase(self, phase: str) -> None:
        """Switch training phase and immediately shrink/grow the VRAM budget to match."""
        if phase not in GPU_PHASE_CAPS:
            raise ValueError(f"Unknown phase '{phase}'. Valid: {list(GPU_PHASE_CAPS)}")
        self.phase = phase
        device_idx    = self.device.index if self.device.index is not None else 0
        free_driver, total = torch.cuda.mem_get_info(device_idx)
        new_budget    = compute_gpu_cache_budget(
            free_driver / (1024 ** 3), total / (1024 ** 3), phase
        )
        new_max_bytes = int(new_budget * (1024 ** 3))
        with self.lock:
            self.max_bytes = new_max_bytes
            while self.cache_bytes > self.max_bytes and self.order:
                self.evict_one()

    def __getitem__(self, idx: int):
        self.evict_under_pressure()
        return super().__getitem__(idx)

    def evict_under_pressure(self) -> None:
        """Evict down to the emergency margin if free VRAM has dropped below it."""
        self.access_count += 1
        if self.access_count % self.PRESSURE_CHECK_INTERVAL != 0:
            return
        device_idx   = self.device.index if self.device.index is not None else 0
        free_driver, total = torch.cuda.mem_get_info(device_idx)
        free_gb      = free_driver / (1024 ** 3)
        emergency_gb = (total / (1024 ** 3)) * GPU_EMERGENCY_MARGIN
        if free_gb >= emergency_gb:
            return
        with self.lock:
            while self.cache and free_gb < emergency_gb:
                self.evict_one()
                free_driver, _ = torch.cuda.mem_get_info(device_idx)
                free_gb = free_driver / (1024 ** 3)


class AdaptiveCacheController:
    """Probes live RAM/disk/VRAM and picks memory, disk, hybrid, or GPU
    caching for a dataset — whichever tier actually fits."""

    def __init__(self, cache_path: str = "./cache", memory_safety_margin_gb: float = 2.0, memory_cache_threshold_gb: float = 6.0, max_cached_recordings: int = 500, device=None):
        self.cache_path = Path(cache_path)
        self.memory_safety_margin = memory_safety_margin_gb
        self.memory_threshold = memory_cache_threshold_gb
        self.max_cached_recordings = max_cached_recordings
        self.device = device
        self.cuda_enabled = device is not None and getattr(device, "type", "") == "cuda"
        self.device_idx = (device.index or 0) if device is not None and self.cuda_enabled else 0
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.monitor = SystemResourceMonitor(
            cache_path=str(self.cache_path), device_idx=self.device_idx, cuda_enabled=self.cuda_enabled
        )

    def estimate_dataset_memory_footprint(self, dataset: Dataset, num_samples_to_probe: int = 10) -> float:
        """Estimate total dataset size in GB by sampling a few items."""
        sample_indices = torch.randperm(len(dataset))[:min(num_samples_to_probe, len(dataset))]
        total_bytes = 0
        successful_probes = 0

        for idx in sample_indices:
            try:
                events, target = dataset[int(idx)]
                total_bytes += measure_event_bytes(events)
                successful_probes += 1
            except Exception as e:
                logger.warning(f"[CACHE CONTROLLER] Could not probe sample {idx}: {e}")
                continue

        if successful_probes == 0:
            logger.warning("[CACHE CONTROLLER] All probes failed — assuming dataset size is 0")
            return 0.0

        return (total_bytes / successful_probes * len(dataset)) / (1024 ** 3)

    def determine_dataset_strategy(self, dataset: Dataset, transform=None, live_transform=None, split: str = "train", num_workers: int = 1,
        force_mode: Optional[Literal["memory", "disk", "hybrid", "gpu_memory", "no_cache"]] = None,
    ) -> Dataset:
        """
        Pick a cache tier from live resources and wrap dataset in it.

        transform: deterministic preprocessing (same output every time) —
            safe to bake into whichever cache is chosen.
        live_transform: stochastic augmentation that must vary every access.
            For memory/disk it's composed after transform, since those
            caches already re-run their transform on every read. For
            hybrid/GPU it's kept out of the cached value and applied via
            the cache's own per-access hook instead.
        num_workers: workers that will share this cache — the hybrid byte
            budget is divided across them to avoid RAM overcommit.
        force_mode: the adaptive on/off switch. None probes live resources
            and picks a strategy; any other value forces that strategy
            instead (still reads live resources once, only to size the
            cache budget within it, not to choose it).
        """
        if hasattr(dataset, "slice_map"):
            raise ValueError(
                "determine_dataset_strategy() received an already-sliced dataset. "
                "Cache must be applied to raw recordings BEFORE slicing — "
                "use: cached_raw = determine_dataset_strategy(raw_dataset); "
                "sliced = TemporalSlicedDataset(cached_raw, config)"
            )

        metrics             = self.monitor.snapshot()
        dataset_size_gb     = self.estimate_dataset_memory_footprint(dataset)
        available_for_cache = metrics.available_ram_gb - self.memory_safety_margin
        self.log_diagnostics(metrics, dataset_size_gb)

        cache_dir = self.cache_path / split
        cache_dir.mkdir(parents=True, exist_ok=True)

        if force_mode:
            mode = force_mode
            threshold_gb = {
                "memory":     available_for_cache,
                "hybrid":     available_for_cache * 0.5,
                "disk":       0.0,
                "gpu_memory": compute_gpu_cache_budget(metrics.gpu_available_gb, metrics.gpu_memory_gb, phase="train"),
                "no_cache":   0.0,
            }[mode]
        elif is_gpu_under_pressure(metrics) and metrics.disk_exists:
            mode, threshold_gb = "disk", 0.0
        elif available_for_cache >= self.memory_threshold and dataset_size_gb < available_for_cache * 0.7:
            mode, threshold_gb = "memory", available_for_cache
        elif metrics.total_ram_gb >= 32.0 and metrics.disk_exists and metrics.disk_available_gb > dataset_size_gb * 1.5:
            mode, threshold_gb = "hybrid", available_for_cache * 0.5
        elif metrics.disk_exists and metrics.disk_available_gb > dataset_size_gb * 1.2:
            mode, threshold_gb = "disk", 0.0
        elif metrics.gpu_available_gb >= 0.5:
            threshold_gb = compute_gpu_cache_budget(metrics.gpu_available_gb, metrics.gpu_memory_gb, phase="train")
            mode = "gpu_memory"
        else:
            raise RuntimeError(
                f"[CACHE CONTROLLER] System halt: insufficient resources. "
                f"RAM: {available_for_cache:.1f}GB, Disk: {metrics.disk_available_gb:.1f}GB, GPU: {metrics.gpu_available_gb:.1f}GB"
            )

        logger.info(f"[CACHE CONTROLLER] {split.upper()} → {mode.upper()} ({available_for_cache:.1f}GB RAM free, dataset ~{dataset_size_gb:.1f}GB)")

        # Only insert the numpy→tensor bridge ahead of live_transform when
        # there actually is one, to match the exact pipeline used when
        # augmentation is present.
        numpy_bridge = torch.from_numpy if live_transform is not None else None

        if mode == "memory":
            return MemoryCachedDataset(dataset, transform=compose_transforms(transform, numpy_bridge, live_transform))

        if mode == "disk":
            return DiskCachedDataset(dataset, transform=compose_transforms(transform, numpy_bridge, live_transform), cache_path=str(cache_dir))

        if mode == "hybrid":
            effective_workers = max(1, num_workers)
            max_bytes = int(threshold_gb * (1024 ** 3)) // effective_workers
            logger.info(
                f"[CACHE CONTROLLER] Hybrid hot layer: {threshold_gb:.1f}GB ÷ {effective_workers} workers "
                f"= {max_bytes / (1024**3):.2f}GB per worker"
            )
            # Cache only the deterministic transform; live_transform runs via
            # BoundedRecordingCache's own per-access hook, not baked in.
            disk_cached = DiskCachedDataset(dataset, transform=transform, cache_path=str(cache_dir))
            return BoundedRecordingCache(
                disk_cached, max_recordings=self.max_cached_recordings, max_bytes=max_bytes,
                transform=compose_transforms(numpy_bridge, live_transform),
            )

        if mode == "gpu_memory":
            if self.device is None or getattr(self.device, "type", "") != "cuda":
                logger.warning("[CACHE CONTROLLER] gpu_memory selected but no CUDA device — no_cache fallback")
                return dataset
            if transform is None:
                raise ValueError(
                    "[CACHE CONTROLLER] gpu_memory cache mode requires a transform (Denoise+ToFrame) "
                    "to encode raw events into a cacheable tensor before caching — it cannot cache raw "
                    "structured event arrays directly. This is why gpu_memory does not support temporal "
                    "slicing: the sliced pipeline path caches raw recordings with no transform (slicing "
                    "needs raw event timestamps), and defers encoding to per-slice processing. Use a "
                    "different cache tier (memory/disk/hybrid) when temporal_slicing is enabled."
                )
            logger.info(f"[CACHE CONTROLLER] GPURecordingCache: {threshold_gb:.2f}GB VRAM budget on {self.device}")
            # live_transform runs against a tensor already on `device` here —
            # no numpy bridge needed, unlike the hybrid/memory/disk tiers.
            return GPURecordingCache(
                dataset, device=self.device, max_bytes=int(threshold_gb * (1024 ** 3)),
                encode_transform=transform, live_transform=live_transform,
            )

        logger.info(f"[CACHE CONTROLLER] {split.upper()} → NO_CACHE (on-the-fly processing)")
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
        """Delete the disk cache for one split, or all splits if none is given."""
        if split:
            cache_dir = self.cache_path / split
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                logger.info(f"[CACHE CONTROLLER] Cleared cache for split: {split}")
        else:
            if self.cache_path.exists():
                shutil.rmtree(self.cache_path)
                logger.info("[CACHE CONTROLLER] Cleared all cache directories")
