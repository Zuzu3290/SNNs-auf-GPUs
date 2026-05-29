"""
Temporal slicing for neuromorphic event-based data.

Slicing is delegated entirely to tonic.SlicedDataset + tonic.slicers.
AdaptiveTemporalSlicer is the only custom piece — it analyses the dataset
and recommends a slice duration when one is not known in advance.
"""
from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset
import tonic
import tonic.slicers as slicers

logger = logging.getLogger(__name__)


@dataclass
class TemporalSliceConfig:
    """Suggested slice parameters returned by AdaptiveTemporalSlicer."""
    slice_duration_us: int
    overlap_us: int = 0
    min_events_per_slice: int = 10
    events_per_slice: Optional[int] = None


def create_sliced_dataset(
    dataset: Dataset,
    slice_duration_ms: float = 15.0,
    overlap_ms: float = 0.0,
    events_per_slice: Optional[int] = None,
    auto_tune: bool = False,
    transform=None,
    metadata_path: Optional[str] = None,
) -> tonic.SlicedDataset:
    """
    Wrap a dataset with tonic's SlicedDataset.

    metadata_path: if provided, tonic stores the slice index as HDF5 so it
    is not rebuilt on subsequent runs.
    """
    if auto_tune:
        config = AdaptiveTemporalSlicer(dataset).suggest_slice_config()
        slice_duration_ms = config.slice_duration_us / 1000.0
        overlap_ms        = config.overlap_us / 1000.0

    if events_per_slice is not None:
        slicer = slicers.SliceByEventCount(event_count=events_per_slice)
    else:
        slicer = slicers.SliceByTime(
            time_window=slice_duration_ms * 1000,
            overlap=overlap_ms * 1000,
        )

    return tonic.SlicedDataset(dataset, slicer=slicer, transform=transform, metadata_path=metadata_path)  # type: ignore[arg-type]


class AdaptiveTemporalSlicer:
    """
    Analyses dataset characteristics and recommends optimal slicing parameters.
    Useful for automatically tuning temporal window size based on event density.
    """

    def __init__(self, dataset: Dataset, num_samples_to_analyze: int = 50):
        self.dataset = dataset
        self.num_samples = min(num_samples_to_analyze, len(dataset))
        self.stats = None

    def analyze(self) -> dict:
        durations    = []
        event_counts = []
        event_rates  = []

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
            'mean_duration_ms':   np.mean(durations),
            'median_duration_ms': np.median(durations),
            'std_duration_ms':    np.std(durations),
            'mean_event_count':   np.mean(event_counts),
            'median_event_count': np.median(event_counts),
            'std_event_count':    np.std(event_counts),
            'mean_event_rate':    np.mean(event_rates)   if event_rates else float('nan'),
            'median_event_rate':  np.median(event_rates) if event_rates else float('nan'),
        }
        self._log_stats()
        return self.stats

    def _log_stats(self):
        assert self.stats is not None
        s   = self.stats
        sep = "=" * 60
        logger.info(sep)
        logger.info("DATASET TEMPORAL STATISTICS")
        logger.info(sep)
        logger.info(f"Duration  — mean: {s['mean_duration_ms']:.2f}ms  median: {s['median_duration_ms']:.2f}ms  std: {s['std_duration_ms']:.2f}ms")
        logger.info(f"Events    — mean: {s['mean_event_count']:.1f}  median: {s['median_event_count']:.1f}  std: {s['std_event_count']:.1f}")
        logger.info(f"Rate      — mean: {s['mean_event_rate']:.1f} ev/ms  median: {s['median_event_rate']:.1f} ev/ms")
        logger.info(sep)

    def suggest_slice_config(
        self,
        target_slices_per_sample: int = 2,
        min_events_per_slice: int = 50,
    ) -> TemporalSliceConfig:
        if self.stats is None:
            self.analyze()

        suggested_ms = self.stats['mean_duration_ms'] / target_slices_per_sample
        suggested_us = int(suggested_ms * 1000)

        expected_events = suggested_ms * self.stats['mean_event_rate']
        if expected_events < min_events_per_slice:
            logger.warning(f"[ADAPTIVE SLICER] {suggested_ms:.1f}ms slice yields ~{expected_events:.1f} events; adjusting up.")
            suggested_ms = min_events_per_slice / self.stats['mean_event_rate']
            suggested_us = int(suggested_ms * 1000)

        logger.info("[ADAPTIVE SLICER] Recommended config:")
        logger.info(f"  Slice duration : {suggested_ms:.2f} ms")
        logger.info("  Overlap        : 0 ms")
        logger.info(f"  Min events     : {min_events_per_slice}")
        logger.info(f"  Expected slices: ~{target_slices_per_sample} per sample")

        return TemporalSliceConfig(
            slice_duration_us=suggested_us,
            overlap_us=0,
            min_events_per_slice=min_events_per_slice,
        )
