"""
Picks a caching strategy (RAM or disk) for neuromorphic recordings based on
live system resources, and provides the cache classes that back each
strategy.

Hardware topology is fixed and singular: data loading/caching always runs
on CPU, training always runs on GPU. This controller does not choose
between hardware configurations — it only chooses where one dataset's
cache lives (RAM or disk). VRAM is intentionally never a cache
storage target: it is reserved for the model's own parameters,
activations, and gradients during training.
"""
from __future__ import annotations
import sys
import logging
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from tonic import DiskCachedDataset, MemoryCachedDataset
from .system_monitor import monitor, is_gpu_under_pressure, GPU_PRESSURE_THRESHOLD

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


class FixedToFrame:
    """Wraps tonic's ToFrame: its empty-input zero-fill branch returns (T,C,W,H), swapped vs its real (T,C,H,W) output -- breaks batching on non-square sensors."""
    def __init__(self, to_frame):
        self.to_frame = to_frame

    def __call__(self, events):
        frame = self.to_frame(events)
        h, w = self.to_frame.sensor_size[1], self.to_frame.sensor_size[0]
        return frame.swapaxes(-1, -2) if frame.shape[-2:] != (h, w) else frame


class ClampToBinary:
    """Turn event COUNTS into 0/1 spikes.

    ToFrame SUMS every event landing in the same pixel, polarity and time bin, so raw
    frame values exceed 1 (measured max on N-MNIST at T=20: 8). This makes the network's
    input actual spikes rather than counts.

    ---------------------------------------------------------------------------
    IT IS A CLAMP, NOT A THRESHOLD.  min(x, 1)
    ---------------------------------------------------------------------------
    That distinction only matters once the values stop being integers:

        INTEGER counts (no augmentation) -- clamping IS binarising:
            0 -> 0    1 -> 1    5 -> 1    8 -> 1        every value ends up 0 or 1

        FRACTIONAL values (rotation interpolates) -- clamping is NOT binarising:
            0.0 -> 0.0    0.37 -> 0.37    1.4 -> 1.0    only >1 is touched

    So with `augmentation.random_rotation_enabled: true` the frames reaching this are
    already fractional and it merely bounds them: the result is NOT a spike train. If you
    want a genuine 0/1 input, set binarize true AND rotation false. A real threshold
    (x > 0) would binarise either way, but it is not what this class does, and changing
    that would silently alter what every past run's input meant.

    Applied on the way OUT of the cache, so switching it on or off never invalidates the
    cache. Handles both an ndarray and a tensor because it may sit either side of the
    numpy->tensor bridge. Module-level class so DataLoader workers can pickle it.
    """

    def __call__(self, frame):
        if isinstance(frame, torch.Tensor):
            return frame.clamp(max=1)
        return np.minimum(frame, 1)


class PreTransformedDataset(Dataset):
    """Applies a deterministic transform once, inside __getitem__, so that a
    wrapping MemoryCachedDataset/DiskCachedDataset caches the POST-transform
    result rather than the raw sample. Without this, tonic's cache classes
    only skip re-fetching the raw item on a cache hit — they unconditionally
    re-run their own `transform` argument on every single access (cache hit
    or miss), which defeats caching entirely for an expensive deterministic
    transform like Denoise+ToFrame."""

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


def cache_identity(wf, denoise_filter_time_us) -> tuple[str, dict]:
    """(directory name, manifest) for everything that changes the CACHED BYTES.

    Returns a readable hint plus a short hash, e.g. `bins20_dn10000_9f3a2b`. The name
    only has to be unique and recognisable; what actually guarantees you never read
    16-bin frames when you asked for 20 is the manifest, checked on load.

    Framing and denoising were previously absent from the cache path entirely -- it was
    just <dataset>/<split> -- so changing n_time_bins reused frames built under the old
    setting. Measured on an earlier branch: after 16 -> 20 the cache still served
    (16, 2, 34, 34) samples while the config said 20. A stale-cache bug produces
    plausible-looking results from the wrong data, which is worse than a crash.
    """
    if wf.FRAME_MODE == "n_time_bins":
        framing = {"mode": "n_time_bins", "n_time_bins": wf.N_TIME_BINS}
        hint = f"bins{wf.N_TIME_BINS}"
    else:
        framing = {"mode": "time_window", "time_window_us": wf.TIME_WINDOW_US}
        hint = f"win{wf.TIME_WINDOW_US}"

    identity = {"framing": framing, "denoise_filter_time_us": denoise_filter_time_us}
    if wf.TEMPORAL_SLICING_ENABLED:
        identity["slicing"] = {
            "events_per_slice": wf.EVENTS_PER_SLICE,
            "calibrate_events_per_slice": wf.CALIBRATE_EVENTS_PER_SLICE,
            "slice_duration_us": (None if wf.EVENTS_PER_SLICE else wf.SLICE_DURATION_US),
        }
        hint += "_sliced"

    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:6]
    return f"{hint}_dn{denoise_filter_time_us}_{digest}", identity


class CacheMismatch(Exception):
    """A cache directory holds samples built with different settings than requested."""


def check_manifest(cache_dir: Path, expected: dict) -> bool:
    """True if a matching cache already exists; False if the directory is fresh.

    RAISES if samples are present but the settings differ, or are present with no
    manifest at all -- both mean the bytes on disk are not what the config asked for, and
    using them silently would corrupt the run.
    """
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        has_samples = cache_dir.is_dir() and any(cache_dir.rglob("*.hdf5"))
        if has_samples:
            raise CacheMismatch(
                f"{cache_dir} holds cached samples but no manifest.json, so the settings "
                "that produced them are unknown. Delete the directory to rebuild."
            )
        return False

    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded != expected:
        raise CacheMismatch(
            f"cache at {cache_dir} was built with different settings.\n"
            f"  on disk: {recorded}\n"
            f"  config : {expected}\n"
            "Delete the directory to rebuild it, or point cache.path elsewhere."
        )
    return True


def write_manifest(cache_dir: Path, settings: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "manifest.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


class AdaptiveCacheController:
    """Probes live RAM/disk and picks memory or disk caching for a
    dataset — whichever tier actually fits. VRAM is never a cache target:
    the GPU is only ever the training device, never a dataset storage
    location."""

    def __init__(self, cache_path: str = "./cache", memory_safety_margin_gb: float = 2.0, memory_cache_threshold_gb: float = 6.0,
                 gpu_pressure_threshold: float = GPU_PRESSURE_THRESHOLD, memory_tier_headroom_fraction: float = 0.7,
                 disk_tier_headroom_multiple: float = 1.2):
        self.cache_path = Path(cache_path)
        self.memory_safety_margin = memory_safety_margin_gb
        self.memory_threshold = memory_cache_threshold_gb
        self.gpu_pressure_threshold = gpu_pressure_threshold
        self.memory_tier_headroom_fraction = memory_tier_headroom_fraction
        self.disk_tier_headroom_multiple = disk_tier_headroom_multiple
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def estimate_dataset_memory_footprint(self, dataset: Dataset, transform=None, num_samples_to_probe: int = 10) -> float:
        """Estimate total dataset size in GB by sampling a few items, measuring
        the POST-transform size when a deterministic transform is given.
        Memory mode caches PreTransformedDataset's output (the
        already-ToFrame'd dense array), not the raw sparse event stream —
        measuring the raw sample here would price a few dozen bytes per
        event and silently undercount what actually ends up resident in RAM
        once the dense per-bin frames are what's held."""
        sample_indices = torch.randperm(len(dataset))[:min(num_samples_to_probe, len(dataset))]
        total_bytes = 0
        successful_probes = 0

        for idx in sample_indices:
            try:
                events, _ = dataset[int(idx)]
                if transform is not None:
                    events = transform(events)
                total_bytes += measure_event_bytes(events)
                successful_probes += 1
            except Exception as e:
                logger.warning(f"[CACHE CONTROLLER] Could not probe sample {idx}: {e}")
                continue

        if successful_probes == 0:
            logger.warning("[CACHE CONTROLLER] All probes failed — assuming dataset size is 0")
            return 0.0

        return (total_bytes / successful_probes * len(dataset)) / (1024 ** 3)

    def determine_dataset_strategy(self, dataset: Dataset, transform=None, live_transform=None, split: str = "train", num_workers: int = 1, manifest: dict | None = None) -> Dataset:
        """Picks a cache tier from live resources and wraps dataset in it; num_workers prices in that MemoryCachedDataset's per-instance dict gets duplicated once per DataLoader worker process."""
        if hasattr(dataset, "slice_map"):
            raise ValueError(
                "determine_dataset_strategy() received an already-sliced dataset. "
                "Cache must be applied to raw recordings BEFORE slicing — "
                "use: cached_raw = determine_dataset_strategy(raw_dataset); "
                "sliced = create_sliced_dataset(cached_raw, ...)  (event_data_workflow/data_pipeline.py)"
            )

        metrics             = monitor.snapshot()
        dataset_size_gb     = self.estimate_dataset_memory_footprint(dataset, transform=transform)
        available_for_cache = metrics.available_ram_gb - self.memory_safety_margin
        effective_size_gb   = dataset_size_gb * max(1, num_workers)  # MemoryCachedDataset's per-instance dict gets duplicated once per DataLoader worker process

        cache_dir = self.cache_path / split
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Verified BEFORE a tier is chosen, so a mismatch fails immediately rather than
        # after the footprint probe has already read samples built under other settings.
        if manifest is not None:
            reused = check_manifest(cache_dir, manifest)
            logger.info(f"[CACHE CONTROLLER] {split} manifest: "
                        + ("matches, reusing" if reused else "fresh, will be built"))

        if is_gpu_under_pressure(metrics, threshold=self.gpu_pressure_threshold) and metrics.disk_exists:
            # A busy GPU means CUDA's pinned-memory allocator is competing
            # for the same physical RAM a memory cache would use —
            # disk sidesteps that contention entirely.
            mode = "disk"
        elif available_for_cache >= self.memory_threshold and effective_size_gb < available_for_cache * self.memory_tier_headroom_fraction:
            mode = "memory"
        elif metrics.disk_exists and metrics.disk_available_gb > dataset_size_gb * self.disk_tier_headroom_multiple:
            # Only two tiers exist (memory, disk): a bounded RAM hot layer on
            # top of disk has no measurable benefit here, since training
            # reshuffles every epoch — see System_Boundaries_and_Tuning.md.
            mode = "disk"
        else:
            raise RuntimeError(
                f"[CACHE CONTROLLER] System halt: insufficient resources. "
                f"RAM: {available_for_cache:.1f}GB, Disk: {metrics.disk_available_gb:.1f}GB"
            )

        logger.info(f"[CACHE CONTROLLER] {split.upper()} -> {mode.upper()} ({available_for_cache:.1f}GB RAM free, dataset ~{dataset_size_gb:.1f}GB x {num_workers} workers = {effective_size_gb:.1f}GB effective)")

        # Only insert the numpy→tensor bridge ahead of live_transform when
        # there actually is one, to match the exact pipeline used when
        # augmentation is present.
        numpy_bridge = torch.from_numpy if live_transform is not None else None

        pre_transformed = PreTransformedDataset(dataset, transform) if transform is not None else dataset

        if mode == "memory":
            return MemoryCachedDataset(pre_transformed, transform=compose_transforms(numpy_bridge, live_transform))

        # "disk": the already-transformed dense frame, cached permanently —
        # the expensive part (measured ~136ms/sample: fetch + Denoise +
        # ToFrame) paid once per sample, ever, not once per epoch.
        # Written for the disk tier only: a memory cache leaves nothing on disk that a
        # later run could mistake for another setting's frames.
        if manifest is not None:
            write_manifest(cache_dir, manifest)
        return DiskCachedDataset(pre_transformed, transform=compose_transforms(numpy_bridge, live_transform), cache_path=str(cache_dir / "disk"), compress=False)

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
