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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # src/ → skeleton
sys.path.insert(0, str(Path(__file__).parent))                # sibling modules

import yaml
import torch
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
import torchvision
from skeleton import Settings
from cache_engine import AdaptiveCacheController, PipelineMemoryCoordinator
from temporal_slicer import create_sliced_dataset

PROJECT_ROOT    = Path(__file__).parent.parent.parent.parent
DATA_DIR        = PROJECT_ROOT / "tmp" / "data"
_WORKFLOW_YAML  = Path(__file__).parent / "data_workflow.yaml"

import tqdm as t
orig_tqdm_init = t.tqdm.__init__
def _mb_init(self, *a, **kw):
    if kw.get("total", 0) > 1_000_000:
        kw.setdefault("unit", "B")
        kw.setdefault("unit_scale", True)
        kw.setdefault("unit_divisor", 1024)
    orig_tqdm_init(self, *a, **kw)
t.tqdm.__init__ = _mb_init

logger = logging.getLogger(__name__)


def _load_workflow_config() -> dict:
    with open(_WORKFLOW_YAML) as f:
        return yaml.safe_load(f)


class NeuromorphicEncoder:
    """
    Builds train and test DataLoaders for neuromorphic event-based datasets.

    High-level switches are available as constructor args; all fine-grained
    tuning parameters are read from data_workflow.yaml in this directory.

    Args:
        cfg                 : Settings object (DATA_PATH, BATCH_SIZE, etc.)
        use_temporal_slicing: Override the yaml enabled flag (None = read from yaml)
        slice_duration_ms   : Override yaml slice_duration_us (None = read from yaml)
        auto_tune_slicing   : Override yaml auto_tune flag
        events_per_slice    : Override yaml events_per_slice
    """

    def __init__(
        self,
        cfg: Settings,
        use_temporal_slicing: bool = None,
        slice_duration_ms: float = None,
        auto_tune_slicing: bool = None,
        events_per_slice: int = None,
    ):
        self.cfg = cfg
        self.wf  = _load_workflow_config()

        ts = self.wf["temporal_slicing"]
        mc = self.wf["memory_coordinator"]
        ca = self.wf["cache"]

        # Constructor args take precedence; fall back to yaml values
        self.use_temporal_slicing = use_temporal_slicing if use_temporal_slicing is not None else ts["enabled"]
        self.slice_duration_ms    = slice_duration_ms    or (ts["slice_duration_us"] / 1000.0)
        self.auto_tune_slicing    = auto_tune_slicing    if auto_tune_slicing    is not None else ts["auto_tune"]
        self.events_per_slice     = events_per_slice     if events_per_slice     is not None else ts["events_per_slice"]

        self.train_loader: DataLoader
        self.test_loader: DataLoader

        self.coordinator = PipelineMemoryCoordinator.from_system(
            safety_margin_gb=mc["safety_margin_gb"],
            recording_cache_fraction=mc["recording_cache_fraction"],
            worker_fraction=mc["worker_fraction"],
            prefetch_fraction=mc["prefetch_fraction"],
            gpu_pressure_threshold=mc["gpu_pressure_threshold"],
        )
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

        self.validate_first_sample(raw_train, "train")
        self.validate_first_sample(raw_test, "test")

        frame_tf = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
        ])
        return raw_train, raw_test, frame_tf

    def apply_pipeline(self, raw_train, raw_test, frame_tf):
        """
        Layer 1 — Cache raw recordings via AdaptiveCacheController.
        Layer 2 — Temporal slicing (optional, stateless).

        Caching always wraps raw datasets so all slices from one recording
        share a single cache entry.
        """
        ca  = self.wf["cache"]
        ts  = self.wf["temporal_slicing"]

        max_recs = self.coordinator.max_recordings(avg_recording_bytes=50_000)
        self.cache_controller = AdaptiveCacheController(
            cache_path=ca["cache_path"],
            memory_safety_margin_gb=ca["memory_safety_margin_gb"],
            memory_cache_threshold_gb=ca["memory_cache_threshold_gb"],
            max_cached_recordings=max_recs,
            verbose=True,
        )
        controller  = self.cache_controller
        force_mode  = ca["force_mode"] or None

        train_tf = transforms.Compose([
            frame_tf,
            torch.from_numpy,
            torchvision.transforms.RandomRotation([-10, 10]),
        ])
        test_tf = frame_tf

        if self.use_temporal_slicing:
            # Layer 1: cache raw events (slicing needs raw timestamps)
            strategy_train = controller.determine_strategy(raw_train, force_mode=force_mode)
            strategy_test  = controller.determine_strategy(raw_test,  force_mode=force_mode)
            cached_train = controller.wrap_dataset(raw_train, split="train", strategy=strategy_train)
            cached_test  = controller.wrap_dataset(raw_test,  split="test",  strategy=strategy_test)

            # Layer 2: stateless slicing with transforms applied per slice
            train_data = create_sliced_dataset(
                cached_train,
                slice_duration_ms=self.slice_duration_ms,
                overlap_ms=ts["overlap_us"] / 1000.0,
                min_events=ts["min_events_per_slice"],
                discard_incomplete=ts["discard_incomplete"],
                auto_tune=self.auto_tune_slicing,
                auto_tune_samples=ts["auto_tune_samples"],
                auto_tune_target_slices=ts["auto_tune_target_slices"],
                events_per_slice=self.events_per_slice,
                transform=train_tf,
                verbose=True,
            )
            test_data = create_sliced_dataset(
                cached_test,
                slice_duration_ms=self.slice_duration_ms,
                overlap_ms=ts["overlap_us"] / 1000.0,
                min_events=ts["min_events_per_slice"],
                discard_incomplete=ts["discard_incomplete"],
                events_per_slice=self.events_per_slice,
                transform=test_tf,
                verbose=True,
            )
            logger.info(f"[PIPELINE] After slicing — train: {len(train_data)}, test: {len(test_data)}")
        else:
            # No slicing: determine strategy once, cache with transforms baked in
            strategy_train = controller.determine_strategy(raw_train, force_mode=force_mode)
            strategy_test  = controller.determine_strategy(raw_test,  force_mode=force_mode)
            train_data = controller.wrap_dataset(raw_train, transform=train_tf, split="train", strategy=strategy_train)
            test_data  = controller.wrap_dataset(raw_test,  transform=test_tf,  split="test",  strategy=strategy_test)

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
        """Clear all disk cache directories."""
        self.cache_controller.clear_cache()
        logger.info("[PIPELINE] Cache cleared")

    def validate_first_sample(self, dataset, split: str) -> None:
        try:
            events, target = dataset[0]
            if events is None or (hasattr(events, "numel") and events.numel() == 0):
                raise ValueError("first sample is empty")
            logger.info(f"[VALIDATION] {split.capitalize()} dataset: first sample OK")
        except Exception as exc:
            logger.error(f"[ERROR] {split} validation failed — {exc}")
            raise RuntimeError(f"{split} dataset validation failed: {exc}") from exc


def main() -> tuple[DataLoader, DataLoader]:
    """
    Standalone entry point — loads N-MNIST by default (cfg.DATA_PATH is empty).
    Returns (train_loader, test_loader) ready for the training loop.
    """
    cfg = Settings()
    encoder = NeuromorphicEncoder(cfg)
    return encoder.get_dataloaders()


if __name__ == "__main__":
    train_loader, test_loader = main()
    logger.info(f"[PIPELINE] Ready — {len(train_loader)} train batches, {len(test_loader)} test batches")
