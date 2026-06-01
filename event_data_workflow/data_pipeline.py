"""
Neuromorphic Data Pipeline — single entry point for all event-based dataset loading.

Strict layer order (enforced):
    1. Raw dataset load        — no transforms, no caching
    2. Adaptive cache          — raw recordings only (BoundedRecordingCache / DiskCache)
    3. Temporal slicing        — stateless transform, optional
    4. DataLoader construction — coordinator-driven worker / prefetch config
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # project root → skeleton package
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "tmp" / "data"
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
import torchvision
import tqdm as t
from skeleton import Settings
from .cache_engine import AdaptiveCacheController, measure_event_bytes
from .pipeline_coordinator import PipelineMemoryCoordinator
from .temporal_slicer import create_sliced_dataset
from .workflow_config import WorkflowSettings

orig_tqdm_init = t.tqdm.__init__
def mb_init(self, *a, **kw):
    if kw.get("total", 0) > 1_000_000:
        kw.setdefault("unit", "B")
        kw.setdefault("unit_scale", True)
        kw.setdefault("unit_divisor", 1024)
    orig_tqdm_init(self, *a, **kw)
t.tqdm.__init__ = mb_init


class NeuromorphicEncoder:
    """
    Builds train and test DataLoaders for neuromorphic event-based datasets.

    Args:
        cfg                 : Settings object (DATA_PATH, BATCH_SIZE, etc.)
        use_temporal_slicing: Divide each recording into temporal windows
        slice_duration_ms   : Window size in ms (default: cfg.TEMPORAL_SLICE_DURATION / 1000)
        auto_tune_slicing   : Analyse dataset and auto-select slice parameters
        events_per_slice    : Switch to event-driven slicing (ignores slice_duration_ms)
    """

    def __init__(self, cfg: Settings, use_temporal_slicing: bool = False, slice_duration_ms: float = None, auto_tune_slicing: bool = False, events_per_slice: int = None ):

        self.cfg = cfg
        self.wf = WorkflowSettings()
        self.use_temporal_slicing = use_temporal_slicing
        self.slice_duration_ms = slice_duration_ms or (cfg.TEMPORAL_SLICE_DURATION / 1000.0)
        self.auto_tune_slicing = auto_tune_slicing
        self.events_per_slice = events_per_slice
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.coordinator = PipelineMemoryCoordinator.from_system(settings=cfg, device=torch.device(cfg.DEVICE))
        self.build()

    def build(self):
        raw_train, raw_test, frame_tf = self.load_raw()
        train_data, test_data = self.apply_pipeline(raw_train, raw_test, frame_tf)
        self.create_loaders(train_data, test_data)

    def load_raw(self):
        """Load raw event datasets with no transform — caching and slicing come later."""
        data_path = self.cfg.DATA_PATH
        time_window = self.cfg.TEMPORAL_SLICE_DURATION

        if not data_path:
            sensor_size = tonic.datasets.NMNIST.sensor_size
            raw_train = tonic.datasets.NMNIST(save_to=str(DATA_DIR), train=True)
            raw_test  = tonic.datasets.NMNIST(save_to=str(DATA_DIR), train=False)
            self.dataset_label = "N-MNIST"
        else:
            full_raw = tonic.datasets.FileDataset(save_to=data_path)
            sensor_size = full_raw.sensor_size
            n_train = int(0.8 * len(full_raw))
            raw_train, raw_test = torch.utils.data.random_split(
                full_raw, [n_train, len(full_raw) - n_train]
            )
            self.dataset_label = getattr(self.cfg, "DATASET_NAME", "Custom Dataset")

        self.sensor_size = sensor_size
        H, W, C = sensor_size
        logger.info(f"[PIPELINE] Dataset : {self.dataset_label}  |  Sensor H={H} W={W} C={C}")
        logger.info(f"[PIPELINE] Train   : {len(raw_train)} recordings")
        logger.info(f"[PIPELINE] Test    : {len(raw_test)} recordings")

        train_sample_bytes = self.validate_first_sample(raw_train, "train")
        self.validate_first_sample(raw_test, "test")
        logger.info(f"[PIPELINE] First train sample size: {train_sample_bytes / 1024:.1f} KB — dataset non-empty, proceeding to cache strategy")

        frame_tf = transforms.Compose([transforms.Denoise(filter_time=10000), transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)])
        return raw_train, raw_test, frame_tf

    def apply_pipeline(self, raw_train, raw_test, frame_tf):
        """
        Layer 1 — Cache raw recordings via AdaptiveCacheController.
        Layer 2 — Temporal slicing (optional, stateless).

        Caching always wraps raw datasets so all slices from one recording
        share a single cache entry.
        """
        controller = AdaptiveCacheController(
            cache_path=self.wf.CACHE_PATH,
            memory_safety_margin_gb=self.wf.MEMORY_SAFETY_MARGIN_GB,
            memory_cache_threshold_gb=self.wf.MEMORY_CACHE_THRESHOLD_GB,
            max_cached_recordings=self.wf.MAX_CACHED_RECORDINGS,
            device=torch.device(self.cfg.DEVICE),
        )

        train_tf = transforms.Compose([frame_tf, torch.from_numpy, torchvision.transforms.RandomRotation([-10, 10])])
        test_tf = frame_tf

        num_workers = self.cfg.NUM_WORKERS

        if self.use_temporal_slicing:
            # Layer 1: cache raw events (slicing needs raw timestamps)
            cached_train = controller.determine_dataset_strategy(raw_train, split="train", num_workers=num_workers, force_mode=self.wf.CACHE_FORCE_MODE)
            cached_test  = controller.determine_dataset_strategy(raw_test,  split="test",  num_workers=num_workers, force_mode=self.wf.CACHE_FORCE_MODE)

            # Layer 2: stateless slicing with transforms applied per slice
            metadata_dir = str(PROJECT_ROOT / "metadata")
            train_data = create_sliced_dataset(
                cached_train,
                slice_duration_ms=self.slice_duration_ms,
                auto_tune=self.auto_tune_slicing,
                events_per_slice=self.events_per_slice,
                transform=train_tf,
                metadata_path=f"{metadata_dir}/train",
            )
            test_data = create_sliced_dataset(
                cached_test,
                slice_duration_ms=self.slice_duration_ms,
                transform=test_tf,
                metadata_path=f"{metadata_dir}/test",
            )
            logger.info(f"[PIPELINE] After slicing — train: {len(train_data)}, test: {len(test_data)}")
            if len(train_data) == 0 or len(test_data) == 0:
                raise RuntimeError(
                    f"[PIPELINE] Temporal slicing produced an empty dataset — "
                    f"train: {len(train_data)} samples, test: {len(test_data)} samples. "
                    "Reduce min_events_per_slice or increase slice_duration_ms in your config."
                )
        else:
            # No slicing: cache with transforms baked in
            train_data = controller.determine_dataset_strategy(raw_train, transform=train_tf, split="train", num_workers=num_workers, force_mode=self.wf.CACHE_FORCE_MODE)
            test_data  = controller.determine_dataset_strategy(raw_test,  transform=test_tf,  split="test",  num_workers=num_workers, force_mode=self.wf.CACHE_FORCE_MODE)

        return train_data, test_data

    def create_loaders(self, train_data, test_data):
        """Build DataLoaders; worker count and prefetch driven by PipelineMemoryCoordinator."""
        batch_size = self.cfg.BATCH_SIZE
        dl_cfg = {
            k: v for k, v in self.coordinator.dataloader_config().items()
            if v is not None  # prefetch_factor must be omitted (not None) when num_workers=0
        }

        pad = tonic.collation.PadTensors(batch_first=False)

        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            collate_fn=pad,
            shuffle=True,
            drop_last=True,
            **dl_cfg,
        )
        self.test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            collate_fn=pad,
            **dl_cfg,
        )

        logger.info(f"[PIPELINE] Train batches : {len(self.train_loader)}")
        logger.info(f"[PIPELINE] Test batches  : {len(self.test_loader)}")
        logger.info(f"[PIPELINE] Batch size    : {batch_size}")

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader) for use in the training loop."""
        return self.train_loader, self.test_loader

    def clear_cache(self):
        """Clear all disk cache directories. Directory is recreated automatically on the next pipeline run."""
        import shutil
        cache_path = Path(self.wf.CACHE_PATH)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info("[PIPELINE] Cache cleared")

    def validate_first_sample(self, dataset, split: str) -> int:
        """Validate the first sample and return its byte size."""
        try:
            events, target = dataset[0]
            if events is None or (hasattr(events, "numel") and events.numel() == 0):
                raise ValueError("first sample is empty")
            sample_bytes = measure_event_bytes(events)
            logger.info(f"[VALIDATION] {split.capitalize()} dataset: first sample OK ({sample_bytes / 1024:.1f} KB)")
            return sample_bytes
        except Exception as exc:
            logger.error(f"[ERROR] {split} validation failed — {exc}")
            sys.exit(1)


def main() -> tuple[DataLoader, DataLoader]:
    """
    Standalone entry point — loads N-MNIST by default (cfg.DATA_PATH is empty).
    Returns (train_loader, test_loader) ready for the training loop.
    """
    cfg = Settings()
    encoder = NeuromorphicEncoder(cfg)
    return encoder.get_dataloaders()
