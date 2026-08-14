"""
Picks a caching strategy (RAM, disk, or hybrid RAM/disk) for neuromorphic
recordings based on live system resources, and provides the cache classes
that back each strategy.

Hardware topology is fixed and singular: data loading/caching always runs
on CPU, training always runs on GPU. This controller does not choose
between hardware configurations — it only chooses where one dataset's
cache lives (RAM, disk, or both). VRAM is intentionally never a cache
storage target: it is reserved for the model's own parameters,
activations, and gradients during training.
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
from .system_monitor import CacheMetrics, monitor, is_gpu_under_pressure

logger = logging.getLogger(__name__)


def measure_event_bytes(events) -> int:
    """Byte size of one sample, for cache byte-budget accounting."""
    if hasattr(events, "nbytes"):
        return events.nbytes
    if hasattr(events, "element_size"):
        return events.element_size() * events.numel()
    return sys.getsizeof(events)


class ComposedTransform:
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


class PreTransformedDataset(Dataset):
    """Applies a deterministic transform once, inside __getitem__, so that a
    wrapping MemoryCachedDataset/DiskCachedDataset caches the POST-transform
    result rather than the raw sample. Without this, tonic's cache classes
    only skip re-fetching the raw item on a cache hit — they unconditionally
    re-run their own `transform` argument on every single access (cache hit
    or miss), which defeats caching entirely for an expensive deterministic
    transform like Denoise+ToFrame. hybrid mode avoids this by wrapping a
    DiskCachedDataset in BoundedRecordingCache, whose __getitem__ genuinely
    memoizes the result; this gives memory/disk mode the same property."""

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return self.transform(data), target

    def __len__(self):
        return len(self.dataset)


def compose_transforms(*fns):
    """Chain callables left to right, skipping any that are None."""
    fns = [f for f in fns if f is not None]
    if not fns:
        return None
    if len(fns) == 1:
        return fns[0]
    return ComposedTransform(fns)


class BaseRecordingCache(Dataset):
    """
    Bounded FIFO cache for raw recordings: caches up to max_recordings /
    max_bytes, evicting the oldest entry once full. Subclasses implement
    prepare_item() to decide what actually gets stored.
    """

    # True on subclasses holding state that can't be shared with a separate
    # DataLoader worker process (e.g. a single mutable cache slot).
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


class AdaptiveCacheController:
    """Probes live RAM/disk and picks memory, disk, or hybrid caching for a
    dataset — whichever tier actually fits. VRAM is never a cache target:
    the GPU is only ever the training device, never a dataset storage
    location."""

    def __init__(self, cache_path: str = "./cache", memory_safety_margin_gb: float = 2.0, memory_cache_threshold_gb: float = 6.0, max_cached_recordings: int = 500):
        self.cache_path = Path(cache_path)
        self.memory_safety_margin = memory_safety_margin_gb
        self.memory_threshold = memory_cache_threshold_gb
        self.max_cached_recordings = max_cached_recordings
        self.cache_path.mkdir(parents=True, exist_ok=True)

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
        force_mode: Optional[Literal["memory", "disk", "hybrid", "no_cache"]] = None) -> Dataset:
        """
        Pick a cache tier from live resources and wrap dataset in it.

        transform: deterministic preprocessing (same output every time) —
            safe to bake into whichever cache is chosen.
        live_transform: stochastic augmentation that must vary every access.
            For memory/disk it's composed after transform, since those
            caches already re-run their transform on every read. For
            hybrid it's kept out of the cached value and applied via the
            cache's own per-access hook instead.
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

        metrics             = monitor.snapshot()
        dataset_size_gb     = self.estimate_dataset_memory_footprint(dataset)
        available_for_cache = metrics.available_ram_gb - self.memory_safety_margin
        self.log_diagnostics(metrics, dataset_size_gb)

        cache_dir = self.cache_path / split
        cache_dir.mkdir(parents=True, exist_ok=True)

        if force_mode:
            mode = force_mode
            threshold_gb = {
                "memory":   available_for_cache,
                "hybrid":   available_for_cache * 0.5,
                "disk":     0.0,
                "no_cache": 0.0,
            }[mode]
        elif is_gpu_under_pressure(metrics) and metrics.disk_exists:
            # A busy GPU means CUDA's pinned-memory allocator is competing
            # for the same physical RAM a memory/hybrid cache would use —
            # disk sidesteps that contention entirely.
            mode, threshold_gb = "disk", 0.0
        elif available_for_cache >= self.memory_threshold and dataset_size_gb < available_for_cache * 0.7:
            mode, threshold_gb = "memory", available_for_cache
        elif metrics.total_ram_gb >= 32.0 and metrics.disk_exists and metrics.disk_available_gb > dataset_size_gb * 1.5:
            mode, threshold_gb = "hybrid", available_for_cache * 0.5
        elif metrics.disk_exists and metrics.disk_available_gb > dataset_size_gb * 1.2:
            mode, threshold_gb = "disk", 0.0
        else:
            raise RuntimeError(
                f"[CACHE CONTROLLER] System halt: insufficient resources. "
                f"RAM: {available_for_cache:.1f}GB, Disk: {metrics.disk_available_gb:.1f}GB"
            )

        logger.info(f"[CACHE CONTROLLER] {split.upper()} → {mode.upper()} ({available_for_cache:.1f}GB RAM free, dataset ~{dataset_size_gb:.1f}GB)")

        # Only insert the numpy→tensor bridge ahead of live_transform when
        # there actually is one, to match the exact pipeline used when
        # augmentation is present.
        numpy_bridge = torch.from_numpy if live_transform is not None else None

        # Bake the deterministic transform in BEFORE handing to tonic's cache
        # classes: MemoryCachedDataset/DiskCachedDataset only skip re-fetching
        # the raw sample on a cache hit, they unconditionally re-run whatever
        # `transform` they're given on every access. Wrapping first means
        # what gets cached is already the (expensive) post-transform result;
        # only numpy_bridge/live_transform run fresh per access, matching how
        # hybrid mode's BoundedRecordingCache already behaves (see its own
        # __getitem__: raw is memoized, self.transform runs fresh on top).
        pre_transformed = PreTransformedDataset(dataset, transform) if transform is not None else dataset

        if mode == "memory":
            return MemoryCachedDataset(pre_transformed, transform=compose_transforms(numpy_bridge, live_transform))

        if mode == "disk":
            return DiskCachedDataset(pre_transformed, transform=compose_transforms(numpy_bridge, live_transform), cache_path=str(cache_dir), compress=False)

        if mode == "hybrid":
            effective_workers = max(1, num_workers)
            max_bytes = int(threshold_gb * (1024 ** 3)) // effective_workers
            logger.info(
                f"[CACHE CONTROLLER] Hybrid hot layer: {threshold_gb:.1f}GB ÷ {effective_workers} workers "
                f"= {max_bytes / (1024**3):.2f}GB per worker"
            )
            # Cache only the deterministic transform; live_transform runs via
            # BoundedRecordingCache's own per-access hook, not baked in.
            disk_cached = DiskCachedDataset(dataset, transform=transform, cache_path=str(cache_dir), compress=False)
            return BoundedRecordingCache(
                disk_cached, max_recordings=self.max_cached_recordings, max_bytes=max_bytes,
                transform=compose_transforms(numpy_bridge, live_transform),
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
