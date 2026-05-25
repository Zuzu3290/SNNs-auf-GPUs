"""
Adaptive Cache Controller for Neuromorphic Data Pipelines
Dynamically balances between MemoryCachedDataset and DiskCachedDataset based on:
- Available RAM
- Disk availability
- Dataset size
- Access patterns
"""
from __future__ import annotations

import os
import sys
import logging
import threading
import psutil
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from tonic import DiskCachedDataset

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """System resource metrics for cache decision-making"""
    total_ram_gb: float
    available_ram_gb: float
    ram_usage_percent: float
    disk_available_gb: float
    disk_exists: bool
    gpu_memory_gb: float
    gpu_available_gb: float


@dataclass
class CacheStrategy:
    """Determined caching strategy based on system resources"""
    mode: Literal["memory", "disk", "hybrid", "no_cache"]
    memory_cache_enabled: bool
    disk_cache_enabled: bool
    memory_threshold_gb: float
    reason: str


@dataclass
class PipelineMemoryBudget:
    """Explicit RAM split across pipeline stages."""
    total_gb: float
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

    Usage:
        coord    = PipelineMemoryCoordinator.from_system()
        max_recs = coord.max_recordings(avg_recording_bytes=50_000)
        dl_cfg   = coord.dataloader_config(batch_bytes=2_000_000)
        q_size   = coord.prefetch_queue_size()
    """

    GPU_PRESSURE_THRESHOLD = 0.75

    def __init__(self, budget: PipelineMemoryBudget, verbose: bool = True, device=None):
        self.budget = budget
        self.verbose = verbose
        self._device_idx = (
            (device.index or 0)
            if device is not None and getattr(device, "type", "") == "cuda"
            else 0
        )

    @classmethod
    def from_system(cls, safety_margin_gb: float = 2.0, verbose: bool = True, device=None) -> "PipelineMemoryCoordinator":
        """Build coordinator from live system metrics."""
        vm = psutil.virtual_memory()
        total_gb = max(1.0, vm.available / (1024 ** 3) - safety_margin_gb)
        coord = cls(PipelineMemoryBudget(total_gb=total_gb), verbose=verbose, device=device)
        if verbose:
            logger.info(f"[MEMORY COORDINATOR] Pipeline budget: {total_gb:.1f} GB")
            coord.log_allocation()
        return coord

    def gpu_pressure(self) -> float:
        """GPU memory utilisation as 0–1. Returns 0 if no GPU."""
        if not torch.cuda.is_available():
            return 0.0
        total = torch.cuda.get_device_properties(self._device_idx).total_memory
        return torch.cuda.memory_allocated(self._device_idx) / total

    def max_cache_bytes(self) -> int:
        """Byte budget for BoundedRecordingCache derived from the effective cache GB."""
        return int(self.effective_cache_gb() * (1024 ** 3))

    def is_gpu_under_pressure(self) -> bool:
        return self.gpu_pressure() > self.GPU_PRESSURE_THRESHOLD

    def effective_cache_gb(self) -> float:
        fraction = self.budget.recording_cache_fraction
        if self.is_gpu_under_pressure():
            fraction *= 0.5
        return self.budget.total_gb * fraction

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
        """Recommend num_workers, prefetch_factor, pin_memory, persistent_workers."""
        worker_bytes = self.budget.total_gb * self.budget.worker_fraction * (1024 ** 3)
        if batch_bytes > 0:
            max_workers = max(1, int(worker_bytes / (2 * batch_bytes)))
        else:
            max_workers = 4
        max_workers = min(max_workers, os.cpu_count() or 4)
        gpu = torch.cuda.is_available()
        cfg = {
            "num_workers": max_workers,
            "prefetch_factor": 2 if max_workers > 0 else None,
            # pin_memory allocates page-locked host memory so CPU→GPU transfers
            # can run asynchronously via non_blocking=True in the training loop.
            "pin_memory": gpu,
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


class BoundedRecordingCache(Dataset):
    """
        cached_raw = BoundedRecordingCache(raw_dataset, max_recordings=500)
        sliced     = TemporalSlicedDataset(cached_raw, config)

    Why cache recordings, not slices:
    - N slices per recording → N cache hits from 1 stored entry (high reuse)
    - Memory is proportional to recordings, not recordings × slices
    - Avoids storing duplicate/overlapping event windows

    Multi-worker safety:
    - threading.Lock guards the dict within a single process
    - __getstate__/__setstate__ rebuild the lock after multiprocessing spawn
    - Each DataLoader worker holds its own warm cache with persistent_workers=True
    """

    def __init__(self, dataset: Dataset, max_recordings: int = 500, max_bytes: int = 0, transform=None):
        self.dataset = dataset
        self.max_recordings = max_recordings
        self.max_bytes = max_bytes  # 0 = no byte cap (count-only eviction)
        self.transform = transform
        self._cache: OrderedDict = OrderedDict()
        self._cache_bytes: int = 0
        self._lock = threading.Lock()

    @staticmethod
    def estimate_item_bytes(raw) -> int:
        events, _ = raw
        if hasattr(events, "nbytes"):
            return int(events.nbytes)
        if hasattr(events, "element_size") and hasattr(events, "numel"):
            return int(events.element_size() * events.numel())
        return 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        with self._lock:
            if idx in self._cache:
                self._cache.move_to_end(idx)
                raw = self._cache[idx]
            else:
                raw = None

        if raw is None:
            raw = self.dataset[idx]
            item_bytes = self.estimate_item_bytes(raw)
            with self._lock:
                if idx not in self._cache:
                    self._cache[idx] = raw
                    self._cache.move_to_end(idx)
                    self._cache_bytes += item_bytes
                    while len(self._cache) > self.max_recordings:
                        _, evicted = self._cache.popitem(last=False)
                        self._cache_bytes -= self.estimate_item_bytes(evicted)
                    if self.max_bytes > 0:
                        while self._cache_bytes > self.max_bytes and self._cache:
                            _, evicted = self._cache.popitem(last=False)
                            self._cache_bytes -= self.estimate_item_bytes(evicted)

        if self.transform is not None:
            events, target = raw
            return self.transform(events), target
        return raw

    @property
    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_lock'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self._cache_bytes = sum(self.estimate_item_bytes(v) for v in self._cache.values())


class AdaptiveCacheController:
    """
    Intelligent caching controller that monitors system resources and selects
    optimal caching strategy for neuromorphic datasets.

    Cache is applied to RAW recordings (before temporal slicing) so that
    all slices derived from one recording share a single cache entry.

    Strategy Selection Logic:
    1. If RAM > threshold and dataset fits → BoundedRecordingCache (fastest)
    2. If RAM < threshold and disk available → DiskCachedDataset (memory-safe)
    3. If RAM > 32GB and disk available → Hybrid (recording cache + disk fallback)
    4. If RAM < 4GB and no disk → No caching (on-the-fly)
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
        self._device_idx = (
            (device.index or 0)
            if device is not None and getattr(device, "type", "") == "cuda"
            else 0
        )

        self.cache_path.mkdir(parents=True, exist_ok=True)

    def get_system_metrics(self) -> CacheMetrics:
        """Probe system resources and return diagnostic metrics"""
        vm = psutil.virtual_memory()
        total_ram = vm.total / (1024**3)
        available_ram = vm.available / (1024**3)
        ram_usage = vm.percent

        try:
            disk_stat = shutil.disk_usage(self.cache_path)
            disk_available = disk_stat.free / (1024**3)
            disk_exists = True
        except Exception:
            disk_available = 0.0
            disk_exists = False

        gpu_total = 0.0
        gpu_available = 0.0
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(self._device_idx).total_memory / (1024**3)
            gpu_available = (gpu_total - torch.cuda.memory_allocated(self._device_idx) / (1024**3))

        return CacheMetrics(
            total_ram_gb=total_ram,
            available_ram_gb=available_ram,
            ram_usage_percent=ram_usage,
            disk_available_gb=disk_available,
            disk_exists=disk_exists,
            gpu_memory_gb=gpu_total,
            gpu_available_gb=gpu_available
        )

    def estimate_dataset_memory_footprint(
        self,
        dataset: Dataset,
        num_samples_to_probe: int = 10
    ) -> float:
        """
        Estimate total dataset memory requirement by sampling.
        Returns estimated size in GB.
        """
        if len(dataset) == 0:
            return 0.0

        sample_indices = torch.randperm(len(dataset))[:min(num_samples_to_probe, len(dataset))]
        total_bytes = 0

        for idx in sample_indices:
            try:
                events, target = dataset[int(idx)]
                if hasattr(events, 'nbytes'):
                    total_bytes += events.nbytes
                elif hasattr(events, 'element_size'):
                    total_bytes += events.element_size() * events.numel()
                else:
                    total_bytes += sys.getsizeof(events)
            except Exception as e:
                if self.verbose:
                    logger.warning(f"[CACHE CONTROLLER] Could not probe sample {idx}: {e}")
                continue

        avg_bytes_per_sample = total_bytes / len(sample_indices)
        estimated_total_gb = (avg_bytes_per_sample * len(dataset)) / (1024**3)

        return estimated_total_gb

    def determine_strategy(
        self,
        dataset: Dataset,
        force_mode: Optional[Literal["memory", "disk", "hybrid", "no_cache"]] = None
    ) -> CacheStrategy:
        """
        Analyze system resources and dataset characteristics to determine
        optimal caching strategy.

        Args:
            dataset: The raw dataset to be cached
            force_mode: Override automatic detection (for testing/debugging)

        Returns:
            CacheStrategy with selected mode and configuration
        """
        metrics = self.get_system_metrics()
        dataset_size_gb = self.estimate_dataset_memory_footprint(dataset)

        if self.verbose:
            self._log_diagnostics(metrics, dataset_size_gb)

        if force_mode:
            return self._create_forced_strategy(force_mode, metrics, dataset_size_gb)

        available_for_cache = metrics.available_ram_gb - self.memory_safety_margin

        # GPU pressure guard: if GPU is using >75% of VRAM, CUDA's pinned-memory
        # allocator competes directly with the recording cache for physical RAM.
        # Skip all RAM-heavy strategies and go straight to disk.
        if metrics.gpu_memory_gb > 0:
            gpu_used_fraction = 1.0 - (metrics.gpu_available_gb / metrics.gpu_memory_gb)
            if gpu_used_fraction > 0.75 and metrics.disk_exists:
                return CacheStrategy(
                    mode="disk",
                    memory_cache_enabled=False,
                    disk_cache_enabled=True,
                    memory_threshold_gb=0.0,
                    reason=(f"GPU under pressure ({gpu_used_fraction*100:.0f}% VRAM used) — "
                            "disk cache chosen to avoid CUDA/RAM competition")
                )

        if (available_for_cache >= self.memory_threshold and
                dataset_size_gb < available_for_cache * 0.7):
            return CacheStrategy(
                mode="memory",
                memory_cache_enabled=True,
                disk_cache_enabled=False,
                memory_threshold_gb=available_for_cache,
                reason=f"Sufficient RAM available ({available_for_cache:.1f}GB free, dataset ~{dataset_size_gb:.1f}GB)"
            )

        if (metrics.total_ram_gb >= 32.0 and
                metrics.disk_exists and
                metrics.disk_available_gb > dataset_size_gb * 1.5):
            return CacheStrategy(
                mode="hybrid",
                memory_cache_enabled=True,
                disk_cache_enabled=True,
                memory_threshold_gb=available_for_cache * 0.5,
                reason=f"Large RAM ({metrics.total_ram_gb:.1f}GB) with disk backup - using hybrid strategy"
            )

        if metrics.disk_exists and metrics.disk_available_gb > dataset_size_gb * 1.2:
            return CacheStrategy(
                mode="disk",
                memory_cache_enabled=False,
                disk_cache_enabled=True,
                memory_threshold_gb=0.0,
                reason=f"Limited RAM ({available_for_cache:.1f}GB) but disk available ({metrics.disk_available_gb:.1f}GB)"
            )

        return CacheStrategy(
            mode="no_cache",
            memory_cache_enabled=False,
            disk_cache_enabled=False,
            memory_threshold_gb=0.0,
            reason=f"Insufficient resources - RAM: {available_for_cache:.1f}GB, Disk: {metrics.disk_available_gb:.1f}GB, Dataset: ~{dataset_size_gb:.1f}GB"
        )

    def _create_forced_strategy(
        self,
        mode: str,
        metrics: CacheMetrics,
        dataset_size_gb: float
    ) -> CacheStrategy:
        """Create strategy based on forced mode"""
        strategies = {
            "memory": CacheStrategy(
                mode="memory",
                memory_cache_enabled=True,
                disk_cache_enabled=False,
                memory_threshold_gb=metrics.available_ram_gb - self.memory_safety_margin,
                reason=f"Forced memory mode (available: {metrics.available_ram_gb:.1f}GB)"
            ),
            "disk": CacheStrategy(
                mode="disk",
                memory_cache_enabled=False,
                disk_cache_enabled=True,
                memory_threshold_gb=0.0,
                reason=f"Forced disk mode (available: {metrics.disk_available_gb:.1f}GB)"
            ),
            "hybrid": CacheStrategy(
                mode="hybrid",
                memory_cache_enabled=True,
                disk_cache_enabled=True,
                memory_threshold_gb=(metrics.available_ram_gb - self.memory_safety_margin) * 0.5,
                reason="Forced hybrid mode"
            ),
            "no_cache": CacheStrategy(
                mode="no_cache",
                memory_cache_enabled=False,
                disk_cache_enabled=False,
                memory_threshold_gb=0.0,
                reason="Forced no-cache mode (on-the-fly processing)"
            )
        }
        return strategies[mode]

    def wrap_dataset(
        self,
        dataset: Dataset,
        transform: Optional[object] = None,
        split: str = "train",
        strategy: Optional[CacheStrategy] = None
    ) -> Dataset:
        """
        Wrap dataset with appropriate caching mechanism based on strategy.

        Args:
            dataset: Raw dataset to wrap
            transform: Optional transforms to apply
            split: Dataset split name (for cache path organization)
            strategy: Pre-determined strategy (if None, will auto-detect)

        Returns:
            Wrapped dataset with optimal caching
        """
        if hasattr(dataset, 'slice_map'):
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
            max_bytes = int(strategy.memory_threshold_gb * (1024 ** 3))
            if self.verbose:
                logger.info(
                    f"[CACHE CONTROLLER] Using BoundedRecordingCache → "
                    f"max {self.max_cached_recordings} recordings, {strategy.memory_threshold_gb:.1f} GB"
                )
            return BoundedRecordingCache(
                dataset,
                max_recordings=self.max_cached_recordings,
                max_bytes=max_bytes,
                transform=transform,
            )

        elif strategy.mode == "disk":
            if self.verbose:
                logger.info(f"[CACHE CONTROLLER] Using DiskCachedDataset → Path: {cache_dir}")
            return DiskCachedDataset(
                dataset,
                transform=transform,
                cache_path=str(cache_dir)
            )

        elif strategy.mode == "hybrid":
            max_bytes = int(strategy.memory_threshold_gb * (1024 ** 3))
            if self.verbose:
                logger.info("[CACHE CONTROLLER] Using Hybrid → disk cache + bounded recording buffer")
            disk_cached = DiskCachedDataset(
                dataset,
                transform=transform,
                cache_path=str(cache_dir)
            )
            return BoundedRecordingCache(
                disk_cached,
                max_recordings=self.max_cached_recordings,
                max_bytes=max_bytes,
            )

        else:  # no_cache
            if self.verbose:
                logger.info("[CACHE CONTROLLER] No caching → On-the-fly processing (may be slower)")
            return dataset

    def _log_diagnostics(self, metrics: CacheMetrics, dataset_size_gb: float):
        """Log detailed system diagnostics"""
        sep = "=" * 70
        logger.info(sep)
        logger.info("ADAPTIVE CACHE CONTROLLER - SYSTEM DIAGNOSTICS")
        logger.info(sep)
        logger.info("RAM Status:")
        logger.info(f"  Total RAM        : {metrics.total_ram_gb:.2f} GB")
        logger.info(f"  Available RAM    : {metrics.available_ram_gb:.2f} GB")
        logger.info(f"  RAM Usage        : {metrics.ram_usage_percent:.1f}%")
        logger.info("Disk Status:")
        logger.info(f"  Disk Available   : {'YES' if metrics.disk_exists else 'NO'}")
        if metrics.disk_exists:
            logger.info(f"  Free Disk Space  : {metrics.disk_available_gb:.2f} GB")
        logger.info("GPU Status:")
        if metrics.gpu_memory_gb > 0:
            logger.info(f"  GPU Memory       : {metrics.gpu_memory_gb:.2f} GB")
            logger.info(f"  GPU Available    : {metrics.gpu_available_gb:.2f} GB")
        else:
            logger.info("  GPU              : Not available or not detected")
        logger.info("Dataset Estimation:")
        logger.info(f"  Est. Size        : ~{dataset_size_gb:.2f} GB")
        logger.info(f"  Safety Margin    : {self.memory_safety_margin:.2f} GB (reserved for system)")
        logger.info(sep)

    def clear_cache(self, split: Optional[str] = None):
        """
        Clear disk cache for specified split or all splits.

        Args:
            split: Specific split to clear, or None to clear all
        """
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


# Utility function for quick integration
def auto_cache_dataset(
    dataset: Dataset,
    transform: Optional[object] = None,
    split: str = "train",
    cache_path: str = "./cache",
    verbose: bool = True
) -> Dataset:
    """
    Convenience function to automatically select and apply optimal caching.

    Usage:
        trainset = tonic.datasets.NMNIST(save_to="./data", train=True)
        cached_trainset = auto_cache_dataset(trainset, transform=my_transform, split="train")
    """
    controller = AdaptiveCacheController(cache_path=cache_path, verbose=verbose)
    return controller.wrap_dataset(dataset, transform=transform, split=split)