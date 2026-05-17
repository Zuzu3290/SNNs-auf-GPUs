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
import psutil
import shutil
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import tonic
from tonic import MemoryCachedDataset, DiskCachedDataset


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


class AdaptiveCacheController:
    """
    Intelligent caching controller that monitors system resources and selects
    optimal caching strategy for neuromorphic datasets.
    
    Strategy Selection Logic:
    1. If RAM > 16GB and dataset fits → MemoryCachedDataset (fastest)
    2. If RAM < 16GB and disk available → DiskCachedDataset (memory-efficient)
    3. If RAM > 32GB and disk available → Hybrid (hot data in RAM, cold on disk)
    4. If RAM < 4GB and no disk → No caching (process on-the-fly)
    """
    
    def __init__(
        self,
        cache_path: str = "./cache",
        memory_safety_margin_gb: float = 2.0,
        memory_cache_threshold_gb: float = 8.0,
        verbose: bool = True
    ):
        """
        Args:
            cache_path: Base directory for disk cache
            memory_safety_margin_gb: RAM to keep free for system stability
            memory_cache_threshold_gb: Minimum free RAM required for memory caching
            verbose: Print diagnostic information
        """
        self.cache_path = Path(cache_path)
        self.memory_safety_margin = memory_safety_margin_gb
        self.memory_threshold = memory_cache_threshold_gb
        self.verbose = verbose
        
        # Create cache directories if needed
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
    def get_system_metrics(self) -> CacheMetrics:
        """Probe system resources and return diagnostic metrics"""
        # RAM metrics
        vm = psutil.virtual_memory()
        total_ram = vm.total / (1024**3)  # Convert to GB
        available_ram = vm.available / (1024**3)
        ram_usage = vm.percent
        
        # Disk metrics
        try:
            disk_stat = shutil.disk_usage(self.cache_path)
            disk_available = disk_stat.free / (1024**3)
            disk_exists = True
        except Exception:
            disk_available = 0.0
            disk_exists = False
        
        # GPU metrics (if available)
        gpu_total = 0.0
        gpu_available = 0.0
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_available = (gpu_total - torch.cuda.memory_allocated(0) / (1024**3))
        
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
        
        # Sample a few data points to estimate average size
        sample_indices = torch.randperm(len(dataset))[:min(num_samples_to_probe, len(dataset))]
        total_bytes = 0
        
        for idx in sample_indices:
            try:
                events, target = dataset[int(idx)]
                # Estimate memory for events (assuming numpy array or tensor)
                if hasattr(events, 'nbytes'):
                    total_bytes += events.nbytes
                elif hasattr(events, 'element_size'):
                    total_bytes += events.element_size() * events.numel()
                else:
                    # Fallback estimate
                    total_bytes += sys.getsizeof(events)
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Could not probe sample {idx}: {e}")
                continue
        
        # Calculate average and extrapolate
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
            self._print_diagnostics(metrics, dataset_size_gb)
        
        # Override if force_mode is specified
        if force_mode:
            return self._create_forced_strategy(force_mode, metrics, dataset_size_gb)
        
        # Decision tree for automatic strategy selection
        available_for_cache = metrics.available_ram_gb - self.memory_safety_margin
        
        # Strategy 1: Pure memory cache (fastest, requires sufficient RAM)
        if (available_for_cache >= self.memory_threshold and 
            dataset_size_gb < available_for_cache * 0.7):  # Use max 70% of available
            return CacheStrategy(
                mode="memory",
                memory_cache_enabled=True,
                disk_cache_enabled=False,
                memory_threshold_gb=available_for_cache,
                reason=f"Sufficient RAM available ({available_for_cache:.1f}GB free, dataset ~{dataset_size_gb:.1f}GB)"
            )
        
        # Strategy 2: Hybrid cache (large RAM + disk available)
        if (metrics.total_ram_gb >= 32.0 and 
            metrics.disk_exists and 
            metrics.disk_available_gb > dataset_size_gb * 1.5):
            return CacheStrategy(
                mode="hybrid",
                memory_cache_enabled=True,
                disk_cache_enabled=True,
                memory_threshold_gb=available_for_cache * 0.5,  # Use 50% for hot data
                reason=f"Large RAM ({metrics.total_ram_gb:.1f}GB) with disk backup - using hybrid strategy"
            )
        
        # Strategy 3: Disk cache (limited RAM but disk available)
        if metrics.disk_exists and metrics.disk_available_gb > dataset_size_gb * 1.2:
            return CacheStrategy(
                mode="disk",
                memory_cache_enabled=False,
                disk_cache_enabled=True,
                memory_threshold_gb=0.0,
                reason=f"Limited RAM ({available_for_cache:.1f}GB) but disk available ({metrics.disk_available_gb:.1f}GB)"
            )
        
        # Strategy 4: No caching (severe resource constraints)
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
        if strategy is None:
            strategy = self.determine_strategy(dataset)
        
        cache_dir = self.cache_path / split
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"\n[CACHE CONTROLLER] Applying {strategy.mode.upper()} strategy for {split} split")
            print(f"[CACHE CONTROLLER] Reason: {strategy.reason}")
        
        # Apply strategy
        if strategy.mode == "memory":
            if self.verbose:
                print(f"[CACHE CONTROLLER] Using MemoryCachedDataset → High-speed RAM access")
            return MemoryCachedDataset(
                dataset,
                transform=transform,
                num_copies=1  # Avoid unnecessary duplication
            )
        
        elif strategy.mode == "disk":
            if self.verbose:
                print(f"[CACHE CONTROLLER] Using DiskCachedDataset → Path: {cache_dir}")
            return DiskCachedDataset(
                dataset,
                transform=transform,
                cache_path=str(cache_dir)
            )
        
        elif strategy.mode == "hybrid":
            if self.verbose:
                print(f"[CACHE CONTROLLER] Using Hybrid strategy → Disk cache with memory buffer")
            # First apply disk cache, then wrap with limited memory cache
            disk_cached = DiskCachedDataset(
                dataset,
                transform=transform,
                cache_path=str(cache_dir)
            )
            # Memory cache acts as a hot-data buffer on top of disk
            return MemoryCachedDataset(
                disk_cached,
                num_copies=1
            )
        
        else:  # no_cache
            if self.verbose:
                print(f"[CACHE CONTROLLER] No caching → On-the-fly processing (may be slower)")
            # Return dataset as-is, transforms applied in DataLoader
            return dataset
    
    def _print_diagnostics(self, metrics: CacheMetrics, dataset_size_gb: float):
        """Print detailed system diagnostics"""
        print("\n" + "="*70)
        print("ADAPTIVE CACHE CONTROLLER - SYSTEM DIAGNOSTICS")
        print("="*70)
        print(f"RAM Status:")
        print(f"  Total RAM        : {metrics.total_ram_gb:.2f} GB")
        print(f"  Available RAM    : {metrics.available_ram_gb:.2f} GB")
        print(f"  RAM Usage        : {metrics.ram_usage_percent:.1f}%")
        print(f"\nDisk Status:")
        print(f"  Disk Available   : {'YES' if metrics.disk_exists else 'NO'}")
        if metrics.disk_exists:
            print(f"  Free Disk Space  : {metrics.disk_available_gb:.2f} GB")
        print(f"\nGPU Status:")
        if metrics.gpu_memory_gb > 0:
            print(f"  GPU Memory       : {metrics.gpu_memory_gb:.2f} GB")
            print(f"  GPU Available    : {metrics.gpu_available_gb:.2f} GB")
        else:
            print(f"  GPU              : Not available or not detected")
        print(f"\nDataset Estimation:")
        print(f"  Est. Size        : ~{dataset_size_gb:.2f} GB")
        print(f"  Safety Margin    : {self.memory_safety_margin:.2f} GB (reserved for system)")
        print("="*70 + "\n")
    
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
                    print(f"[CACHE CONTROLLER] Cleared cache for split: {split}")
        else:
            if self.cache_path.exists():
                shutil.rmtree(self.cache_path)
                self.cache_path.mkdir(parents=True, exist_ok=True)
                if self.verbose:
                    print(f"[CACHE CONTROLLER] Cleared all cache directories")


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