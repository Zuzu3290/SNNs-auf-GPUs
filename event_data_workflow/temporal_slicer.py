"""
Temporal Slicer for Neuromorphic Event-Based Data
Divides long event recordings into smaller temporal windows to:
- Increase number of training samples (data augmentation)
- Reduce memory consumption per batch
- Improve computational efficiency
- Enable models to process shorter, more focused sequences
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class TemporalSliceConfig:
    """Configuration for temporal slicing operations"""
    slice_duration_us: int           # Duration of each slice in microseconds
    overlap_us: int = 0              # Overlap between consecutive slices
    min_events_per_slice: int = 10   # Discard slices with fewer events than this
    discard_incomplete: bool = False  # Whether to discard the last partial slice
    events_per_slice: Optional[int] = None  # Event-driven mode: new slice every N events


class TemporalSlicedDataset(Dataset):
    """
    Wraps a neuromorphic dataset and provides temporal slicing on-the-fly.

    Slice index is built once at construction:
      slice_map[i] = (original_idx, start_idx, end_idx)
    where start_idx / end_idx are positions in the raw event array.
    __getitem__ then does events[start_idx:end_idx] — no mask, no timestamp scan.

    Two slicing modes:
      Time-driven  (default): fixed-duration windows located via np.searchsorted
      Event-driven (opt-in) : new slice every events_per_slice events regardless of time

    Example:
        Original: 80 recordings × 30ms each = 80 samples
        Sliced:   80 recordings × 2 slices (15ms each) = 160 samples
    """

    def __init__(
        self,
        dataset: Dataset,
        slice_config: TemporalSliceConfig,
        transform=None,
        verbose: bool = False
    ):
        self.dataset = dataset
        self.config = slice_config
        self.transform = transform
        self.verbose = verbose

        if (self.config.events_per_slice is None and
                self.config.overlap_us >= self.config.slice_duration_us):
            raise ValueError(
                f"[TEMPORAL SLICER] overlap_us ({self.config.overlap_us}µs) must be less than "
                f"slice_duration_us ({self.config.slice_duration_us}µs) — "
                "a zero or negative stride produces infinite or duplicate slices."
            )

        if self.config.overlap_us > 0 and self.verbose:
            overlap_pct = 100 * self.config.overlap_us / self.config.slice_duration_us
            logger.warning(
                f"[TEMPORAL SLICER] overlap={self.config.overlap_us/1000:.1f}ms "
                f"({overlap_pct:.0f}% of slice). Adjacent slices share events — "
                "set overlap_us=0 to eliminate duplicate training samples."
            )

        self.build_slice_index()

    # Index building
    def build_slice_index(self):
        """
        Pre-compute (original_idx, start_idx, end_idx) for every valid slice.
        Scans each sample once using binary search; sparse slices are dropped here
        so __getitem__ never needs to check.
        """
        # Each entry: (original_dataset_idx, start_pos_in_event_array, end_pos)
        self.slice_map: List[Tuple[int, int, int]] = []

        if self.verbose:
            mode = "event-driven" if self.config.events_per_slice else "time-driven"
            logger.info(f"[TEMPORAL SLICER] Building slice index ({mode})...")
            if self.config.events_per_slice:
                logger.info(f"  Events per slice : {self.config.events_per_slice}")
            else:
                logger.info(f"  Slice duration   : {self.config.slice_duration_us / 1000:.1f} ms")
                logger.info(f"  Overlap          : {self.config.overlap_us / 1000:.1f} ms")

        total_slices = 0
        discarded_slices = 0

        for idx in range(len(self.dataset)):
            try:
                events, _ = self.dataset[idx]

                if len(events) == 0:
                    continue

                t = events['t'] if hasattr(events, 'dtype') else events[:, 2]

                if self.config.events_per_slice is not None:
                    bounds = self.event_driven_bounds(len(t))
                else:
                    bounds = self.time_driven_bounds(t)

                for start_idx, end_idx in bounds:
                    if end_idx - start_idx < self.config.min_events_per_slice:
                        discarded_slices += 1
                        continue
                    self.slice_map.append((idx, start_idx, end_idx))
                    total_slices += 1

            except Exception as e:
                if self.verbose:
                    logger.error(f"  [TEMPORAL SLICER] Failed to process sample {idx}: {e}")
                continue

        if self.verbose:
            n_orig = len(self.dataset)
            expansion = total_slices / n_orig if n_orig > 0 else 0
            logger.info(f"  Original samples  : {n_orig}")
            logger.info(f"  Total slices      : {total_slices}")
            logger.info(f"  Discarded (sparse): {discarded_slices}")
            logger.info(f"  Expansion factor  : {expansion:.2f}x")
            logger.info("[TEMPORAL SLICER] Index built")

    def time_driven_bounds(self, t: np.ndarray) -> List[Tuple[int, int]]:
        """
        Locate slice boundaries with np.searchsorted — O(log N) per slice
        instead of scanning the full array for each window.
        """
        if len(t) > 1 and not np.all(t[:-1] <= t[1:]):
            raise ValueError(
                "[TEMPORAL SLICER] Non-monotonic timestamps in recording. "
                "np.searchsorted requires sorted input — pre-sort events by timestamp "
                "before dataset construction, or use event-driven slicing (events_per_slice=N)."
            )
        t_min = int(t[0])
        t_max = int(t[-1])
        duration = t_max - t_min
        stride = self.config.slice_duration_us - self.config.overlap_us
        if stride <= 0:
            stride = self.config.slice_duration_us

        n_slices = max(1, int(np.ceil((duration - self.config.slice_duration_us) / stride)) + 1)

        bounds: List[Tuple[int, int]] = []
        for i in range(n_slices):
            slice_start = t_min + i * stride
            slice_end   = slice_start + self.config.slice_duration_us

            if self.config.discard_incomplete and slice_end > t_max:
                continue

            start_idx = int(np.searchsorted(t, slice_start, side='left'))
            end_idx   = int(np.searchsorted(t, slice_end,   side='left'))
            bounds.append((start_idx, end_idx))

        return bounds

    def event_driven_bounds(self, n_events: int) -> List[Tuple[int, int]]:
        """
        Slice by event count rather than time.
        Creates a new window every events_per_slice events — no overlap,
        no inactive windows, no repeated background activity.
        """
        stride = self.config.events_per_slice
        bounds: List[Tuple[int, int]] = []
        start = 0
        while start < n_events:
            end = min(start + stride, n_events)
            bounds.append((start, end))
            start = end
        return bounds

    # Dataset interface
    def __len__(self) -> int:
        return len(self.slice_map)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Retrieve a temporal slice.
        Boundaries are pre-computed — this is a direct array index, no mask.
        """
        original_idx, start_idx, end_idx = self.slice_map[idx]

        events, target = self.dataset[original_idx]

        if hasattr(events, 'dtype'):  # structured array (x, y, t, p)
            sliced_events = events[start_idx:end_idx].copy()
            if len(sliced_events) > 0:
                sliced_events['t'] -= sliced_events['t'][0]
        else:  # tensor fallback — timestamps at column 2
            sliced_events = events[start_idx:end_idx].clone()
            if len(sliced_events) > 0:
                sliced_events[:, 2] -= sliced_events[0, 2]

        if self.transform is not None and len(sliced_events) > 0:
            sliced_events = self.transform(sliced_events)

        return sliced_events, target


class AdaptiveTemporalSlicer:
    """
    Analyzes dataset characteristics and recommends optimal slicing parameters.
    Useful for automatically tuning temporal window size based on event density.
    """

    def __init__(self, dataset: Dataset, num_samples_to_analyze: int = 50):
        self.dataset = dataset
        self.num_samples = min(num_samples_to_analyze, len(dataset))
        self.stats = None

    def analyze(self) -> dict:
        """
        Analyze temporal characteristics of the dataset.

        Returns:
            Dictionary with statistics:
            - mean_duration_ms, median_duration_ms
            - mean_event_count, median_event_count
            - mean_event_rate (events/ms)
        """
        durations = []
        event_counts = []
        event_rates = []

        indices = np.random.choice(len(self.dataset), self.num_samples, replace=False)
        logger.info(f"[ADAPTIVE SLICER] Analyzing {self.num_samples} samples...")

        for idx in indices:
            try:
                events, _ = self.dataset[int(idx)]
                if len(events) == 0:
                    continue

                if hasattr(events, 'dtype'):
                    t_min, t_max = events['t'][0], events['t'][-1]
                else:
                    t_min, t_max = events[0, 2], events[-1, 2]

                duration_ms = (t_max - t_min) / 1000.0
                num_events  = len(events)
                durations.append(duration_ms)
                event_counts.append(num_events)
                if duration_ms > 0:
                    event_rates.append(num_events / duration_ms)

            except Exception as e:
                logger.warning(f"  [ADAPTIVE SLICER] Failed to analyze sample {idx}: {e}")
                continue

        if not durations:
            raise RuntimeError(
                f"[ADAPTIVE SLICER] Could not collect valid statistics — "
                f"all {self.num_samples} sampled entries were empty or failed to load."
            )

        self.stats = {
            'mean_duration_ms':    np.mean(durations),
            'median_duration_ms':  np.median(durations),
            'std_duration_ms':     np.std(durations),
            'mean_event_count':    np.mean(event_counts),
            'median_event_count':  np.median(event_counts),
            'std_event_count':     np.std(event_counts),
            'mean_event_rate':     np.mean(event_rates)   if event_rates else float('nan'),
            'median_event_rate':   np.median(event_rates) if event_rates else float('nan'),
        }
        self.log_stats()
        return self.stats

    def log_stats(self):
        if self.stats is None:
            return
        s = self.stats
        sep = "=" * 60
        logger.info(sep)
        logger.info("DATASET TEMPORAL STATISTICS")
        logger.info(sep)
        logger.info(f"Duration  — mean: {s['mean_duration_ms']:.2f}ms  median: {s['median_duration_ms']:.2f}ms  std: {s['std_duration_ms']:.2f}ms")
        logger.info(f"Events    — mean: {s['mean_event_count']:.1f}  median: {s['median_event_count']:.1f}  std: {s['std_event_count']:.1f}")
        logger.info(f"Rate      — mean: {s['mean_event_rate']:.1f} ev/ms  median: {s['median_event_rate']:.1f} ev/ms")
        logger.info(sep)

    def suggest_slice_config(self, target_slices_per_sample: int = 2, min_events_per_slice: int = 50) -> TemporalSliceConfig:
        """
        Suggest optimal slice configuration based on analyzed statistics.
        """
        if self.stats is None:
            self.analyze()

        suggested_ms = self.stats['mean_duration_ms'] / target_slices_per_sample
        suggested_us = int(suggested_ms * 1000)

        expected_events = suggested_ms * self.stats['mean_event_rate']
        if expected_events < min_events_per_slice:
            logger.warning(f"[ADAPTIVE SLICER] {suggested_ms:.1f}ms slice yields ~{expected_events:.1f} events; adjusting up.")
            suggested_ms = min_events_per_slice / self.stats['mean_event_rate']
            suggested_us = int(suggested_ms * 1000)

        config = TemporalSliceConfig(
            slice_duration_us=suggested_us,
            overlap_us=0,
            min_events_per_slice=min_events_per_slice,
            discard_incomplete=False,
        )
        logger.info("[ADAPTIVE SLICER] Recommended config:")
        logger.info(f"  Slice duration : {suggested_ms:.2f} ms")
        logger.info("  Overlap        : 0 ms (no duplicate samples)")
        logger.info(f"  Min events     : {min_events_per_slice}")
        logger.info(f"  Expected slices: ~{target_slices_per_sample} per sample")

        return config


# Convenience helpers
def create_sliced_dataset(
    dataset: Dataset,
    slice_duration_ms: float = 15.0,
    overlap_ms: float = 0.0,
    min_events: int = 10,
    events_per_slice: Optional[int] = None,
    auto_tune: bool = False,
    transform=None,
    verbose: bool = True
) -> TemporalSlicedDataset:
    """
    Create a temporally sliced dataset.
    """
    if events_per_slice is not None:
        config = TemporalSliceConfig(
            slice_duration_us=int(slice_duration_ms * 1000),
            overlap_us=0,
            min_events_per_slice=min_events,
            events_per_slice=events_per_slice,
        )
    elif auto_tune:
        analyzer = AdaptiveTemporalSlicer(dataset, num_samples_to_analyze=50)
        config = analyzer.suggest_slice_config(
            target_slices_per_sample=2,
            min_events_per_slice=min_events,
        )
    else:
        config = TemporalSliceConfig(
            slice_duration_us=int(slice_duration_ms * 1000),
            overlap_us=int(overlap_ms * 1000),
            min_events_per_slice=min_events,
            discard_incomplete=False,
        )

    return TemporalSlicedDataset(dataset, config, transform=transform, verbose=verbose)


def create_cached_sliced_dataset(
    dataset: Dataset,
    slice_duration_ms: float = 15.0,
    cache_path: str = "./cache",
    split: str = "train",
    transform: Optional[object] = None,
    verbose: bool = True
) -> Dataset:
    """
    Combines adaptive caching with temporal slicing in strict layer order:

        Layer 1 — Cache raw recordings  (BoundedRecordingCache / DiskCachedDataset)
        Layer 2 — Stateless slicing     (TemporalSlicedDataset)

    Caching MUST come before slicing so that all slices from one recording
    share a single cache entry.  Reversing the order caches every slice
    individually and multiplies RAM usage by the slice expansion factor.
    """
    from .cache_engine import AdaptiveCacheController

    # Layer 1: cache raw recordings before any slicing
    controller = AdaptiveCacheController(cache_path=cache_path, verbose=verbose)
    cached_raw = controller.wrap_dataset(dataset, split=split)

    # Layer 2: stateless slice transformation from cached recordings
    return create_sliced_dataset(
        cached_raw,
        slice_duration_ms=slice_duration_ms,
        transform=transform,
        verbose=verbose,
    )