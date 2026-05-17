"""
Enhanced Universal Data Processing Pipeline for Spiking Neural Networks.
Integrates:
- NeuromorphicEncoder (base pipeline)
- AdaptiveCacheController (intelligent memory/disk caching)
- TemporalSlicer (temporal window augmentation)
"""
from __future__ import annotations
 
import sys
from pathlib import Path
# Add parent directory to path to find skeleton module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from skeleton.snn_config import Settings

import torch
from torch.utils.data import DataLoader
import tonic
from tonic import DiskCachedDataset
import tonic.transforms as transforms
import torchvision

# Import new components
from .cache_engine import AdaptiveCacheController, auto_cache_dataset
from .temporal_slicer import create_sliced_dataset, create_cached_sliced_dataset


class NeuromorphicEncoder:
    """
    Original encoder with optional adaptive caching and temporal slicing.
    
    New features:
    - use_adaptive_cache: Automatically selects memory vs disk caching based on system resources
    - use_temporal_slicing: Divides recordings into smaller windows for memory efficiency
    - slice_duration_ms: Duration of each temporal slice (default 15ms)
    """

    def __init__(
        self, 
        cfg: Settings, 
        use_cache: bool = True,
        use_adaptive_cache: bool = True,
        use_temporal_slicing: bool = False,
        slice_duration_ms: float = 15.0,
        auto_tune_slicing: bool = False
    ):
        """
        Args:
            cfg: Settings object with DATA_PATH, BATCH_SIZE, etc.
            use_cache: Enable caching (legacy parameter, kept for compatibility)
            use_adaptive_cache: Use intelligent cache controller (recommended)
            use_temporal_slicing: Enable temporal window slicing
            slice_duration_ms: Slice duration in milliseconds (if slicing enabled)
            auto_tune_slicing: Automatically determine optimal slice parameters
        """
        self.cfg = cfg
        self.use_cache = use_cache
        self.use_adaptive_cache = use_adaptive_cache
        self.use_temporal_slicing = use_temporal_slicing
        self.slice_duration_ms = slice_duration_ms
        self.auto_tune_slicing = auto_tune_slicing
        
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        
        # Initialize cache controller if adaptive caching is enabled
        if self.use_adaptive_cache:
            self.cache_controller = AdaptiveCacheController(
                cache_path="./cache",
                memory_safety_margin_gb=2.0,
                memory_cache_threshold_gb=8.0,
                verbose=True
            )
        
        self.build()
    
    def build(self):
        """Build the complete data pipeline with all enhancements"""
        train_data, test_data = self._prepare_datasets()
        self._create_loaders(train_data, test_data)

    def _prepare_datasets(self):
        """
        Load, validate, slice, and cache datasets.
        Returns (train_data, test_data) ready for DataLoader wrapping.
        """
        data_path = self.cfg.DATA_PATH

        # ========================================
        # STEP 1: Load base dataset
        # ========================================
        if not data_path:
            sensor_size = tonic.datasets.NMNIST.sensor_size
            frame_tf = transforms.Compose([
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(sensor_size=sensor_size, time_window=1000)
            ])
            if self.use_temporal_slicing:
                # Raw events needed so the temporal slicer can slice on timestamps;
                # frame_tf is applied per-slice inside TemporalSlicedDataset.
                trainset = tonic.datasets.NMNIST(save_to="./tmp/data", train=True)
                testset = tonic.datasets.NMNIST(save_to="./tmp/data", train=False)
            else:
                trainset = tonic.datasets.NMNIST(save_to="./tmp/data", transform=frame_tf, train=True)
                testset = tonic.datasets.NMNIST(save_to="./tmp/data", transform=frame_tf, train=False)
            self.dataset_label = "N-MNIST (default)"
        else:
            full_raw = tonic.datasets.FileDataset(save_to=data_path)
            sensor_size = full_raw.sensor_size
            frame_tf = transforms.Compose([
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(sensor_size=sensor_size, time_window=1000)
            ])
            n_train = int(0.8 * len(full_raw))
            n_test = len(full_raw) - n_train
            if self.use_temporal_slicing:
                trainset, testset = torch.utils.data.random_split(full_raw, [n_train, n_test])
            else:
                full_set = tonic.datasets.FileDataset(save_to=data_path, transform=frame_tf)
                trainset, testset = torch.utils.data.random_split(full_set, [n_train, n_test])
            self.dataset_label = getattr(self.cfg, "DATASET_NAME", "Custom Neuromorphic Dataset")

        self.sensor_size = sensor_size
        H, W, C = self.sensor_size
        print(f"\n{'='*70}")
        print(f"NEUROMORPHIC PIPELINE INITIALIZATION")
        print(f"{'='*70}")
        print(f"Sensor size  : H={H}  W={W}  C={C}  (polarity channels)")
        print(f"Dataset      : {self.dataset_label}")
        print(f"Train samples: {len(trainset)}")
        print(f"Test samples : {len(testset)}")

        self.validate_first_sample(trainset, "train")
        self.validate_first_sample(testset, "test")

        # ========================================
        # STEP 2: Apply temporal slicing (optional)
        # ========================================
        if self.use_temporal_slicing:
            print(f"\n[PIPELINE] Applying temporal slicing...")
            trainset = create_sliced_dataset(
                trainset,
                slice_duration_ms=self.slice_duration_ms,
                auto_tune=self.auto_tune_slicing,
                transform=frame_tf,
                verbose=True
            )
            testset = create_sliced_dataset(
                testset,
                slice_duration_ms=self.slice_duration_ms,
                auto_tune=False,
                transform=frame_tf,
                verbose=True
            )
            print(f"[PIPELINE] After slicing - Train: {len(trainset)}, Test: {len(testset)}")

        # ========================================
        # STEP 3: Define augmentation transforms
        # ========================================
        aug_tf = transforms.Compose([
            torch.from_numpy,
            torchvision.transforms.RandomRotation([-10, 10])
        ])

        # ========================================
        # STEP 4: Apply caching strategy
        # ========================================
        if self.use_adaptive_cache:
            print(f"\n[PIPELINE] Applying adaptive caching...")
            train_data = self.cache_controller.wrap_dataset(trainset, transform=aug_tf, split="train")
            test_data = self.cache_controller.wrap_dataset(testset, split="test")
        elif self.use_cache:
            print(f"\n[PIPELINE] Using disk caching...")
            train_data = DiskCachedDataset(trainset, transform=aug_tf, cache_path="./cache/train")
            test_data = DiskCachedDataset(testset, cache_path="./cache/test")
        else:
            print(f"\n[PIPELINE] No caching - on-the-fly processing")
            train_data, test_data = trainset, testset

        return train_data, test_data

    def _create_loaders(self, train_data, test_data):
        """Create DataLoaders from prepared datasets."""
        batch_size  = self.cfg.BATCH_SIZE
        num_workers = self.cfg.NUM_WORKERS
        pad = tonic.collation.PadTensors(batch_first=False)

        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=pad,
            shuffle=True,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=True,
        )

        self.test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=pad,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

        print(f"\n[PIPELINE] DataLoaders created successfully")
        print(f"  Batch size: {batch_size}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        print(f"{'='*70}\n")

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Return configured train and test dataloaders"""
        return self.train_loader, self.test_loader

    def validate_first_sample(self, dataset, split: str) -> None:
        """Index dataset[0] directly — no DataLoader overhead. Exits on failure."""
        try:
            events, target = dataset[0]
            if events is None or (hasattr(events, "numel") and events.numel() == 0):
                raise ValueError("first sample is empty")
            print(f"[VALIDATION] {split.capitalize()} dataset: ✓ First sample validated")
        except Exception as exc:
            print(f"[ERROR] {split} dataset validation failed — {exc}")
            sys.exit(1)
    
    def clear_cache(self):
        """Clear all cache directories"""
        if self.use_adaptive_cache and hasattr(self, 'cache_controller'):
            self.cache_controller.clear_cache()
        else:
            import shutil
            from pathlib import Path
            cache_path = Path("./cache")
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print("[PIPELINE] Cache cleared")


"""
First, sampler.set_epoch(epoch) must be called at the start of every epoch. The sampler uses the epoch number as a random seed for shuffling.
If you forget this, every epoch will iterate over data in the same order, which degrades generalisation.

Second, pin_memory=True in the DataLoader pre-allocates page-locked host memory, enabling asynchronous CPU-to-GPU transfers
when you call tensor.to(device, non_blocking=True). This overlap is where real throughput gains come from.

Third, persistent_workers=True avoids respawning worker processes every epoch — a significant overhead reduction when num_workers > 0.
"""
# def create_distributed_dataloader(dataset, config, ctx):
#     sampler = DistributedSampler(
#         dataset,
#         num_replicas=ctx.world_size,
#         rank=ctx.rank,
#         shuffle=True,
#     )
#     loader = DataLoader(
#         dataset,
#         batch_size=config.batch_size,
#         sampler=sampler,
#         num_workers=config.num_workers,
#         pin_memory=True,
#         drop_last=True,
#         persistent_workers=config.num_workers > 0,
#     )
#     return loader, sampler


# ========================================
# Usage Examples
# ========================================

def example_basic_usage():
    """Basic usage with default settings"""
    from skeleton import Settings
    
    cfg = Settings()
    cfg.DATA_PATH = None  # Will use N-MNIST
    cfg.BATCH_SIZE = 32
    
    encoder = NeuromorphicEncoder(cfg)
    train_loader, test_loader = encoder.get_dataloaders()
    
    # Training loop
    for epoch in range(10):
        for batch_idx, (events, targets) in enumerate(train_loader):
            # events shape: (time, batch, channels, height, width)
            # targets shape: (batch,)
            pass


def example_optimized_usage():
    """Recommended configuration with all optimizations"""
    from skeleton import Settings
    
    cfg = Settings()
    cfg.DATA_PATH = None
    cfg.BATCH_SIZE = 32
    
    encoder = NeuromorphicEncoder(
        cfg,
        use_adaptive_cache=True,      # Intelligent cache selection
        use_temporal_slicing=True,     # Increase dataset size
        slice_duration_ms=15.0,        # 15ms slices (30ms → 2 slices)
        auto_tune_slicing=True         # Auto-tune based on dataset stats
    )
    
    train_loader, test_loader = encoder.get_dataloaders()
    
    print(f"Optimized pipeline ready:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")




if __name__ == "__main__":
    # Run optimized example
    example_optimized_usage()