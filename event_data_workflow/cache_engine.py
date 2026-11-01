"""Caches the framed samples in RAM or on disk, whichever the free resources allow, so framing runs once; VRAM is never used for the cache."""
from __future__ import annotations
import logging
import shutil
from pathlib import Path
import torch
from torch.utils.data import Dataset
import tonic.transforms as transforms
from tonic import DiskCachedDataset, MemoryCachedDataset
from .system_monitor import monitor

logger = logging.getLogger(__name__)

class FixedToFrame:
    """Wraps tonic's ToFrame: its empty-input zero-fill branch returns (T,C,W,H), swapped vs its real (T,C,H,W) output -- breaks batching on non-square sensors."""
    def __init__(self, to_frame):
        self.to_frame = to_frame

    def __call__(self, events):
        frame = self.to_frame(events)
        h, w = self.to_frame.sensor_size[1], self.to_frame.sensor_size[0]
        return frame.swapaxes(-1, -2) if frame.shape[-2:] != (h, w) else frame


class PreTransformedDataset(Dataset):
    """Frames each recording once in __getitem__, so the cache wrapped around it stores frames, not raw events."""

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return self.transform(data), target

    def __len__(self):
        return len(self.dataset)


def cache_folder_name(wf, denoise_filter_time_us) -> str:
    """Cache folder name spelling out every setting that changes the cached frames, e.g. bins16_dn10000, so other settings never reuse them."""
    name = f"bins{wf.N_TIME_BINS}" if wf.FRAME_MODE == "n_time_bins" else f"win{wf.TIME_WINDOW_US}"
    name += f"_dn{denoise_filter_time_us}"
    if wf.TEMPORAL_SLICING_ENABLED:
        name += f"_slice{wf.SLICE_DURATION_US}_ov{wf.SLICE_OVERLAP_US}"
    return name


class AdaptiveCacheController:
    """Probes live RAM/disk and picks memory or disk caching for a dataset — whichever tier actually fits. VRAM is never a cache target:
    the GPU is only ever the training device, never a dataset storage location."""

    def __init__(self, cache_path: str, memory_safety_margin_gb: float, memory_cache_threshold_gb: float,
                 memory_tier_headroom_fraction: float, disk_tier_headroom_multiple: float):
        self.cache_path = Path(cache_path)
        self.memory_safety_margin = memory_safety_margin_gb
        self.memory_threshold = memory_cache_threshold_gb
        self.memory_tier_headroom_fraction = memory_tier_headroom_fraction
        self.disk_tier_headroom_multiple = disk_tier_headroom_multiple
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def estimate_dataset_memory_footprint(self, dataset: Dataset, transform=None, num_samples_to_probe: int = 10) -> float:
        """Dataset size in GB from a few sampled items, measured after framing, since the cache holds frames, not the much smaller raw events."""
        sample_indices = torch.randperm(len(dataset))[:min(num_samples_to_probe, len(dataset))]
        total_bytes = 0

        for idx in sample_indices:
            events, _ = dataset[int(idx)]
            if transform is not None:
                events = transform(events)
            total_bytes += events.nbytes

        return (total_bytes / len(sample_indices) * len(dataset)) / (1024 ** 3)

    def determine_dataset_strategy(self, dataset: Dataset, transform=None, live_transform=None, split: str = "train", num_workers: int = 1) -> Dataset:
        """Picks a cache tier from live resources and wraps dataset in it; num_workers prices in that MemoryCachedDataset's per-instance dict gets duplicated once per DataLoader worker process."""
        metrics            = monitor.snapshot()
        dataset_size_gb     = self.estimate_dataset_memory_footprint(dataset, transform=transform)
        available_for_cache = metrics.available_ram_gb - self.memory_safety_margin
        effective_size_gb   = dataset_size_gb * max(1, num_workers)  # MemoryCachedDataset's per-instance dict gets duplicated once per DataLoader worker process

        cache_dir = self.cache_path / split
        cache_dir.mkdir(parents=True, exist_ok=True)

        if available_for_cache >= self.memory_threshold and effective_size_gb < available_for_cache * self.memory_tier_headroom_fraction:
            mode = "memory"
        elif metrics.disk_available_gb > dataset_size_gb * self.disk_tier_headroom_multiple:
            # No RAM hot layer over disk: training reshuffles every epoch, so it would not help.
            mode = "disk"
        else:
            raise RuntimeError(
                f"[CACHE CONTROLLER] System halt: insufficient resources. "
                f"RAM: {available_for_cache:.1f}GB, Disk: {metrics.disk_available_gb:.1f}GB"
            )

        logger.info(f"[CACHE CONTROLLER] {split.upper()} -> {mode.upper()} ({available_for_cache:.1f}GB RAM free, dataset ~{dataset_size_gb:.1f}GB x {num_workers} workers = {effective_size_gb:.1f}GB effective)")

        # live_transform (augmentation) runs after the cache on every access, on a tensor rather than the cached numpy frame.
        after_cache = transforms.Compose([torch.from_numpy, live_transform]) if live_transform is not None else None

        pre_transformed = PreTransformedDataset(dataset, transform) if transform is not None else dataset

        if mode == "memory":
            return MemoryCachedDataset(pre_transformed, transform=after_cache)

        # Disk: the framed sample (fetch + Denoise + ToFrame, ~136 ms) is paid once ever, not once per epoch.
        return DiskCachedDataset(pre_transformed, transform=after_cache, cache_path=str(cache_dir / "disk"), compress=False)

    def clear_cache(self, split=None):
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
