"""
Temporal Slicer for Neuromorphic Event-Based Data
Divides long event recordings into smaller temporal windows to:
- Increase number of training samples (data augmentation)
- Reduce memory consumption per batch
- Improve computational efficiency
- Enable models to process shorter, more focused sequences
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
import tonic
import tonic.transforms as transforms


@dataclass
class TemporalSliceConfig:
    """Configuration for temporal slicing operations"""
    slice_duration_us: int  # Duration of each slice in microseconds
    overlap_us: int = 0  # Overlap between consecutive slices (for temporal continuity)
    min_events_per_slice: int = 10  # Discard slices with fewer events than this
    discard_incomplete: bool = False  # Whether to discard the last partial slice
    

class TemporalSlicedDataset(Dataset):
    """
    Wraps a neuromorphic dataset and provides temporal slicing on-the-fly.
    
    Example:
        Original: 80 recordings × 30ms each = 80 samples
        Sliced:   80 recordings × 2 slices (15ms each) = 160 samples
        
    This effectively multiplies the dataset size while reducing per-sample
    memory footprint and improving temporal resolution.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        slice_config: TemporalSliceConfig,
        transform=None,
        verbose: bool = False
    ):
        """
        Args:
            dataset: Base neuromorphic dataset returning (events, target)
                     events should be numpy structured array with 'x','y','t','p' fields
            slice_config: Temporal slicing configuration
            transform: Optional transform applied to each slice after extraction
            verbose: Print diagnostic information
        """
        self.dataset = dataset
        self.config = slice_config
        self.transform = transform
        self.verbose = verbose
        
        # Build index mapping: (original_idx, slice_idx) for each output sample
        self._build_slice_index()
        
    def _build_slice_index(self):
        """
        Pre-compute slice index for efficient __len__ and __getitem__.
        Scans entire dataset once to determine slice boundaries.
        """
        self.slice_map: List[Tuple[int, int]] = []  # [(original_idx, slice_number), ...]
        
        if self.verbose:
            print(f"\n[TEMPORAL SLICER] Building slice index...")
            print(f"  Slice duration: {self.config.slice_duration_us / 1000:.1f} ms")
            print(f"  Overlap: {self.config.overlap_us / 1000:.1f} ms")
        
        total_slices = 0
        discarded_slices = 0
        
        for idx in range(len(self.dataset)):
            try:
                events, _ = self.dataset[idx]
                
                # Extract temporal bounds
                if len(events) == 0:
                    if self.verbose and idx < 5:
                        print(f"  [WARNING] Sample {idx} has no events, skipping")
                    continue
                
                # Events should be structured array with 't' field (time in microseconds)
                t_min = events['t'][0] if hasattr(events, 'dtype') else events[0, 2]  # Fallback for tensor
                t_max = events['t'][-1] if hasattr(events, 'dtype') else events[-1, 2]
                recording_duration = t_max - t_min
                
                # Calculate number of slices for this sample
                stride = self.config.slice_duration_us - self.config.overlap_us
                num_slices = max(1, int(np.ceil((recording_duration - self.config.slice_duration_us) / stride)) + 1)
                
                # Add each valid slice to the index
                for slice_idx in range(num_slices):
                    # Check if this is the last slice and we should discard incomplete slices
                    slice_start = t_min + slice_idx * stride
                    slice_end = slice_start + self.config.slice_duration_us
                    
                    if self.config.discard_incomplete and slice_end > t_max:
                        discarded_slices += 1
                        continue
                    
                    self.slice_map.append((idx, slice_idx))
                    total_slices += 1
                    
            except Exception as e:
                if self.verbose:
                    print(f"  [ERROR] Failed to process sample {idx}: {e}")
                continue
        
        if self.verbose:
            original_samples = len(self.dataset)
            expansion_factor = total_slices / original_samples if original_samples > 0 else 0
            print(f"  Original samples: {original_samples}")
            print(f"  Total slices: {total_slices}")
            print(f"  Discarded slices: {discarded_slices}")
            print(f"  Expansion factor: {expansion_factor:.2f}x")
            print(f"[TEMPORAL SLICER] Index built successfully\n")
    
    def __len__(self) -> int:
        """Return total number of temporal slices across all samples"""
        return len(self.slice_map)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Get a specific temporal slice.
        
        Returns:
            events: Numpy structured array with temporal slice
            target: Original target label
        """
        original_idx, slice_num = self.slice_map[idx]
        
        # Get original sample
        events, target = self.dataset[original_idx]
        
        # Calculate slice time boundaries
        if hasattr(events, 'dtype'):  # Structured array
            t_min = events['t'][0]
        else:  # Fallback for tensor
            t_min = events[0, 2]
            
        stride = self.config.slice_duration_us - self.config.overlap_us
        slice_start = t_min + slice_num * stride
        slice_end = slice_start + self.config.slice_duration_us
        
        # Extract events within this temporal window
        if hasattr(events, 'dtype'):  # Structured array
            mask = (events['t'] >= slice_start) & (events['t'] < slice_end)
            sliced_events = events[mask].copy()
            
            # Normalize timestamps to start at 0 for this slice
            if len(sliced_events) > 0:
                sliced_events['t'] = sliced_events['t'] - sliced_events['t'][0]
        else:  # Tensor fallback
            mask = (events[:, 2] >= slice_start) & (events[:, 2] < slice_end)
            sliced_events = events[mask].clone()
            if len(sliced_events) > 0:
                sliced_events[:, 2] = sliced_events[:, 2] - sliced_events[0, 2]
        
        # Validate minimum event count
        if len(sliced_events) < self.config.min_events_per_slice:
            if self.verbose and idx < 5:
                print(f"[WARNING] Slice {idx} has only {len(sliced_events)} events (min: {self.config.min_events_per_slice})")

        if self.transform is not None:
            sliced_events = self.transform(sliced_events)

        return sliced_events, target


class AdaptiveTemporalSlicer:
    """
    Analyzes dataset characteristics and recommends optimal slicing parameters.
    Useful for automatically tuning temporal window size based on event density.
    """
    
    def __init__(self, dataset: Dataset, num_samples_to_analyze: int = 50):
        """
        Args:
            dataset: Neuromorphic dataset to analyze
            num_samples_to_analyze: Number of samples to probe for statistics
        """
        self.dataset = dataset
        self.num_samples = min(num_samples_to_analyze, len(dataset))
        self.stats = None
        
    def analyze(self) -> dict:
        """
        Analyze temporal characteristics of the dataset.
        
        Returns:
            Dictionary with statistics:
            - mean_duration_ms: Average recording duration
            - median_duration_ms: Median recording duration
            - mean_event_count: Average events per recording
            - median_event_count: Median events per recording
            - mean_event_rate: Average events per millisecond
        """
        durations = []
        event_counts = []
        event_rates = []
        
        indices = np.random.choice(len(self.dataset), self.num_samples, replace=False)
        
        print(f"\n[ADAPTIVE SLICER] Analyzing {self.num_samples} samples...")
        
        for idx in indices:
            try:
                events, _ = self.dataset[int(idx)]
                
                if len(events) == 0:
                    continue
                
                # Extract temporal info
                if hasattr(events, 'dtype'):
                    t_min = events['t'][0]
                    t_max = events['t'][-1]
                else:
                    t_min = events[0, 2]
                    t_max = events[-1, 2]
                
                duration_us = t_max - t_min
                duration_ms = duration_us / 1000.0
                num_events = len(events)
                
                durations.append(duration_ms)
                event_counts.append(num_events)
                if duration_ms > 0:
                    event_rates.append(num_events / duration_ms)
                    
            except Exception as e:
                print(f"  [WARNING] Failed to analyze sample {idx}: {e}")
                continue
        
        if not durations:
            raise RuntimeError(
                f"[ADAPTIVE SLICER] Could not collect valid statistics — "
                f"all {self.num_samples} sampled dataset entries were empty or failed to load. "
                "Check that the dataset path is correct and events are non-empty."
            )

        self.stats = {
            'mean_duration_ms': np.mean(durations),
            'median_duration_ms': np.median(durations),
            'std_duration_ms': np.std(durations),
            'mean_event_count': np.mean(event_counts),
            'median_event_count': np.median(event_counts),
            'std_event_count': np.std(event_counts),
            'mean_event_rate': np.mean(event_rates) if event_rates else float('nan'),
            'median_event_rate': np.median(event_rates) if event_rates else float('nan'),
        }
        
        self._print_stats()
        return self.stats
    
    def _print_stats(self):
        """Print analysis results"""
        if self.stats is None:
            return
            
        print(f"\n{'='*60}")
        print("DATASET TEMPORAL STATISTICS")
        print(f"{'='*60}")
        print(f"Duration:")
        print(f"  Mean   : {self.stats['mean_duration_ms']:.2f} ms")
        print(f"  Median : {self.stats['median_duration_ms']:.2f} ms")
        print(f"  Std Dev: {self.stats['std_duration_ms']:.2f} ms")
        print(f"\nEvent Count:")
        print(f"  Mean   : {self.stats['mean_event_count']:.1f} events")
        print(f"  Median : {self.stats['median_event_count']:.1f} events")
        print(f"  Std Dev: {self.stats['std_event_count']:.1f} events")
        print(f"\nEvent Rate:")
        print(f"  Mean   : {self.stats['mean_event_rate']:.1f} events/ms")
        print(f"  Median : {self.stats['median_event_rate']:.1f} events/ms")
        print(f"{'='*60}\n")
    
    def suggest_slice_config(
        self,
        target_slices_per_sample: int = 2,
        min_events_per_slice: int = 50
    ) -> TemporalSliceConfig:
        """
        Suggest optimal slice configuration based on analyzed statistics.
        
        Args:
            target_slices_per_sample: Desired number of slices per original sample
            min_events_per_slice: Minimum events required per slice
            
        Returns:
            Recommended TemporalSliceConfig
        """
        if self.stats is None:
            self.analyze()
        
        # Calculate slice duration to achieve target number of slices
        suggested_slice_duration_ms = self.stats['mean_duration_ms'] / target_slices_per_sample
        suggested_slice_duration_us = int(suggested_slice_duration_ms * 1000)
        
        # Validate against event density
        expected_events = suggested_slice_duration_ms * self.stats['mean_event_rate']
        
        if expected_events < min_events_per_slice:
            print(f"[WARNING] Suggested slice ({suggested_slice_duration_ms:.1f}ms) may have too few events ({expected_events:.1f})")
            print(f"[WARNING] Adjusting to ensure minimum {min_events_per_slice} events per slice")
            suggested_slice_duration_ms = min_events_per_slice / self.stats['mean_event_rate']
            suggested_slice_duration_us = int(suggested_slice_duration_ms * 1000)
        
        # Suggest 10% overlap for temporal continuity
        overlap_us = int(suggested_slice_duration_us * 0.1)
        
        config = TemporalSliceConfig(
            slice_duration_us=suggested_slice_duration_us,
            overlap_us=overlap_us,
            min_events_per_slice=min_events_per_slice,
            discard_incomplete=False
        )
        
        print(f"\n[ADAPTIVE SLICER] Recommended Configuration:")
        print(f"  Slice duration: {suggested_slice_duration_ms:.2f} ms")
        print(f"  Overlap: {overlap_us / 1000:.2f} ms")
        print(f"  Min events: {min_events_per_slice}")
        print(f"  Expected slices per sample: ~{target_slices_per_sample}")
        print(f"  Expected events per slice: ~{expected_events:.1f}\n")
        
        return config


# Convenience function for quick integration
def create_sliced_dataset(
    dataset: Dataset,
    slice_duration_ms: float = 15.0,
    overlap_ms: float = 0.0,
    min_events: int = 10,
    auto_tune: bool = False,
    transform=None,
    verbose: bool = True
) -> TemporalSlicedDataset:
    """
    Quick creation of temporally sliced dataset.
    
    Args:
        dataset: Base neuromorphic dataset
        slice_duration_ms: Duration of each slice in milliseconds
        overlap_ms: Overlap between slices in milliseconds
        min_events: Minimum events required per slice
        auto_tune: If True, analyze dataset and auto-tune parameters
        verbose: Print diagnostic information
        
    Returns:
        TemporalSlicedDataset ready for DataLoader
        
    Example:
        trainset = tonic.datasets.NMNIST(save_to="./data", train=True)
        sliced_trainset = create_sliced_dataset(trainset, slice_duration_ms=15.0)
        # Now 80 samples → 160 slices (if each was 30ms)
    """
    if auto_tune:
        analyzer = AdaptiveTemporalSlicer(dataset, num_samples_to_analyze=50)
        config = analyzer.suggest_slice_config(
            target_slices_per_sample=2,
            min_events_per_slice=min_events
        )
    else:
        config = TemporalSliceConfig(
            slice_duration_us=int(slice_duration_ms * 1000),
            overlap_us=int(overlap_ms * 1000),
            min_events_per_slice=min_events,
            discard_incomplete=False
        )
    
    return TemporalSlicedDataset(dataset, config, transform=transform, verbose=verbose)


# Example integration with caching
def create_cached_sliced_dataset(
    dataset: Dataset,
    slice_duration_ms: float = 15.0,
    cache_path: str = "./cache",
    split: str = "train",
    transform: Optional[object] = None,
    verbose: bool = True
) -> Dataset:
    """
    Combines temporal slicing with adaptive caching for optimal performance.
    
    Pipeline:
        1. Apply temporal slicing (expands dataset)
        2. Apply adaptive caching (memory or disk based on resources)
        3. Return optimized dataset ready for DataLoader
        
    Example:
        from adaptive_cache_controller import AdaptiveCacheController
        
        trainset = tonic.datasets.NMNIST(save_to="./data", train=True)
        optimized = create_cached_sliced_dataset(
            trainset,
            slice_duration_ms=15.0,
            split="train"
        )
        loader = DataLoader(optimized, batch_size=32, shuffle=True)
    """
    # Import here to avoid circular dependency
    from cache_engine import AdaptiveCacheController
    
    # Step 1: Apply temporal slicing
    sliced = create_sliced_dataset(
        dataset,
        slice_duration_ms=slice_duration_ms,
        verbose=verbose
    )
    
    # Step 2: Apply adaptive caching
    controller = AdaptiveCacheController(cache_path=cache_path, verbose=verbose)
    cached_sliced = controller.wrap_dataset(
        sliced,
        transform=transform,
        split=split
    )
    
    return cached_sliced