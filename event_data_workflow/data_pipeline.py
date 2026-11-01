"""
Builds train/test DataLoaders for neuromorphic event datasets: load raw recordings, cache them via AdaptiveCacheController, 
optionally slice into temporal windows, then wrap in DataLoaders.
"""
from __future__ import annotations
import inspect
import os
import logging
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "tmp" / "data"
logger = logging.getLogger(__name__)
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset
import tonic
import tonic.transforms as transforms
import tonic.slicers as slicers
import tqdm as t
from skeleton.seeding import split_generator
from .system_monitor import monitor
from torchvision.transforms import v2
from .cache_engine import AdaptiveCacheController, FixedToFrame, cache_folder_name
from .dataset_registry import resolve_dataset_entry


class RawDatasets:
    """Raw recordings in usable form: retried downloads, and string labels mapped to indices."""

    def __init__(self, labels):
        self.mapping = {label: index for index, label in enumerate(sorted(set(labels)))}

    def __call__(self, label):
        return self.mapping[label]

    @classmethod
    def index_targets(cls, *datasets: Dataset) -> None:
        """Give every dataset ONE shared label mapping, so train and test agree on class indices."""
        labels = [target for dataset in datasets for target in getattr(dataset, "targets", [])]
        if not labels or isinstance(labels[0], (int, bool)):
            return
        mapper = cls(labels)
        for dataset in datasets:
            dataset.target_transform = mapper


def create_sliced_dataset(dataset: Dataset, slice_duration_us, overlap_us, transform, metadata_path) -> tonic.SlicedDataset:
    """Wrap dataset with tonic's SlicedDataset; metadata_path stores the slice index as HDF5 so later runs reuse it."""
    slicer = slicers.SliceByTime(time_window=slice_duration_us, overlap=overlap_us)

    return tonic.SlicedDataset(dataset, slicer=slicer, transform=transform, metadata_path=metadata_path)

orig_tqdm_init = t.tqdm.__init__
def mb_init(self, *a, **kw):
    if (kw.get("total") or 0) > 1_000_000:
        kw.setdefault("unit", "B")
        kw.setdefault("unit_scale", True)
        kw.setdefault("unit_divisor", 1024)
    orig_tqdm_init(self, *a, **kw)
t.tqdm.__init__ = mb_init


class DeviceLoader:
    """Yields batches already on the device, so training and inference never move data themselves."""

    def __init__(self, loader, device: torch.device):
        self.loader = loader
        self.device = device

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for data, targets in self.loader:
            yield (data.to(self.device, non_blocking=True),
                   targets.to(self.device, non_blocking=True) if isinstance(targets, torch.Tensor) else targets)


class NeuromorphicEncoder:
    """Loads a dataset, caches it, and builds the train/test DataLoaders used by training."""

    def __init__(self, cfg, dataset_entry: dict, use_temporal_slicing=None, slice_duration_ms=None):

        self.cfg = cfg
        self.dataset_entry = dataset_entry  # resolved once by the caller, from dataset_registry

        cuda_enabled = torch.device(cfg.DEVICE).type == "cuda"
        monitor.configure(cache_path=self.cfg.CACHE_PATH, cuda_enabled=cuda_enabled)

        metrics = monitor.snapshot()
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        logger.info(
            f"[PIPELINE] System: {physical_cores} physical cores, "
            f"{metrics.available_ram_gb:.2f}GB RAM free, "
            f"{metrics.gpu_available_gb:.2f}GB VRAM free — before any dataset operation"
        )

        self.use_temporal_slicing = use_temporal_slicing if use_temporal_slicing is not None else self.cfg.TEMPORAL_SLICING_ENABLED
        self.slice_duration_ms = slice_duration_ms or (self.cfg.SLICE_DURATION_US / 1000.0)
        self.train_loader: DeviceLoader
        self.test_loader: DeviceLoader
        self.build()

    def build(self):
        raw_train, raw_test, frame_tf = self.load_raw()
        train_data, test_data = self.apply_pipeline(raw_train, raw_test, frame_tf)
        self.create_loaders(train_data, test_data)

    def load_raw(self):
        """Load the raw dataset (no transform, no cache yet) and build the frame transform for it."""
        entry = self.dataset_entry
        sensor_size = entry["sensor_size"]
        # tonic datasets that ship a train/test split take a `train` argument; the rest are split 80/20 here.
        if "train" in inspect.signature(entry["cls"]).parameters:
            raw_train = entry["cls"](save_to=str(DATA_DIR), train=True)
            raw_test  = entry["cls"](save_to=str(DATA_DIR), train=False)
            RawDatasets.index_targets(raw_train, raw_test)
        else:
            full_raw = entry["cls"](save_to=str(DATA_DIR))
            RawDatasets.index_targets(full_raw)
            n_train = int(0.8 * len(full_raw))
            raw_train, raw_test = torch.utils.data.random_split(
                full_raw, [n_train, len(full_raw) - n_train],
                generator=split_generator(getattr(self.cfg, "SEED", 0)),
            )
        self.dataset_label = entry["name"]
        self.sensor_size = sensor_size
        W, H, C = sensor_size
        logger.info(f"[PIPELINE] Dataset : {self.dataset_label}  |  Sensor H={H} W={W} C={C}")
        logger.info(f"[PIPELINE] Train   : {len(raw_train)} recordings")
        logger.info(f"[PIPELINE] Test    : {len(raw_test)} recordings")

        train_sample_bytes = self.validate_first_sample(raw_train, "train")
        self.validate_first_sample(raw_test, "test")
        logger.info(f"[PIPELINE] First train sample size: {train_sample_bytes / 1024:.1f} KB — dataset non-empty, proceeding to cache strategy")

        if entry.get("input") == "frames":
            self.denoise_filter_time_us = self.cfg.DENOISE_FILTER_TIME_US
            frame_tf = transforms.Compose([])
            self.batch_sample_bytes = frame_tf(raw_train[0][0]).nbytes
            logger.info("[PIPELINE] Frame input -- denoise and binning skipped")
            return raw_train, raw_test, frame_tf

        self.denoise_filter_time_us = self.cfg.DENOISE_FILTER_TIME_US
        denoise = transforms.Denoise(filter_time=self.denoise_filter_time_us)
        if self.cfg.FRAME_MODE == "n_time_bins":
            to_frame = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.cfg.N_TIME_BINS)
        else:
            to_frame = transforms.ToFrame(sensor_size=sensor_size, time_window=self.cfg.TIME_WINDOW_US)
        frame_tf = transforms.Compose([denoise, FixedToFrame(to_frame)])

        first_frame = frame_tf(raw_train[0][0])
        self.batch_sample_bytes = first_frame.nbytes

        return raw_train, raw_test, frame_tf

    def apply_pipeline(self, raw_train, raw_test, frame_tf):
        """Cache the raw recordings, then optionally slice them into temporal windows."""
        self.controller = AdaptiveCacheController(
            cache_path=self.cfg.CACHE_PATH,
            memory_safety_margin_gb=self.cfg.MEMORY_SAFETY_MARGIN_GB,
            memory_cache_threshold_gb=self.cfg.MEMORY_CACHE_THRESHOLD_GB,
            memory_tier_headroom_fraction=self.cfg.MEMORY_TIER_HEADROOM_FRACTION,
            disk_tier_headroom_multiple=self.cfg.DISK_TIER_HEADROOM_MULTIPLE,
        )
        controller = self.controller

        worker_estimate, _, _ = monitor.worker_count(
            torch.device(self.cfg.DEVICE), safety_margin_gb=self.cfg.MEMORY_SAFETY_MARGIN_GB,
            worker_count_override=self.cfg.WORKER_COUNT,
        )
        worker_estimate = max(1, worker_estimate)

        augment = v2.RandomRotation(self.cfg.AUGMENT_ROTATION_DEG) if self.cfg.AUGMENT_ENABLED else None
        train_tf = transforms.Compose([frame_tf, torch.from_numpy] + ([augment] if augment else []))
        test_tf = frame_tf

        dataset_prefix = f'{self.dataset_label.replace(" ", "_")}/{cache_folder_name(self.cfg, self.denoise_filter_time_us)}'
        logger.info(f"[PIPELINE] Cache folder: {dataset_prefix}")

        if self.use_temporal_slicing:

            cached_train = controller.determine_dataset_strategy(raw_train, split=f"{dataset_prefix}/train", num_workers=worker_estimate)
            cached_test  = controller.determine_dataset_strategy(raw_test,  split=f"{dataset_prefix}/test", num_workers=worker_estimate)

            metadata_dir = str(PROJECT_ROOT / "metadata" / dataset_prefix)
            train_data = create_sliced_dataset(cached_train, self.slice_duration_ms * 1000, self.cfg.SLICE_OVERLAP_US,
                                               train_tf, f"{metadata_dir}/train")
            test_data = create_sliced_dataset(cached_test, self.slice_duration_ms * 1000, self.cfg.SLICE_OVERLAP_US,
                                              test_tf, f"{metadata_dir}/test")
            logger.info(f"[PIPELINE] After slicing — train: {len(train_data)}, test: {len(test_data)}")
            if len(train_data) == 0 or len(test_data) == 0:
                raise RuntimeError(
                    f"[PIPELINE] Temporal slicing produced an empty dataset — "
                    f"train: {len(train_data)} samples, test: {len(test_data)} samples. "
                    "Increase temporal.slice_duration_us in data_workflow.yaml."
                )
        else:
            train_data = controller.determine_dataset_strategy( raw_train, transform=frame_tf, live_transform=augment, split=f"{dataset_prefix}/train", num_workers=worker_estimate)
            test_data = controller.determine_dataset_strategy( raw_test, transform=frame_tf, split=f"{dataset_prefix}/test", num_workers=worker_estimate)

        source = getattr(raw_train, "dataset", raw_train)
        if getattr(source, "requires_single_process_loading", False):
            setattr(train_data, "requires_single_process_loading", True)
            setattr(test_data, "requires_single_process_loading", True)
            self.warm_recording_cache(train_data, raw_train)
            self.warm_recording_cache(test_data, raw_test)

        return train_data, test_data

    def warm_recording_cache(self, cached_data, raw_subset) -> None:
        """Populates cached_data in recording order (not raw_subset's random_split order) so a multi-recording WindowedRecordingDataset's single-recording load cache doesn't thrash on the first pass."""
        for local_idx in np.argsort(raw_subset.indices):
            cached_data[int(local_idx)]

    def create_loaders(self, train_data, test_data):
        """Build the train/test DataLoaders, with worker config sized from live
        RAM, wrapped in DeviceLoader so batches arrive already device-resident."""
        batch_size = self.cfg.BATCH_SIZE
        if len(train_data) < batch_size:
            logger.info(f"[PIPELINE] Batch size {batch_size} > {len(train_data)} train samples -> clamping to {len(train_data)} so drop_last=True still yields a batch")
            batch_size = max(1, len(train_data))
        device = torch.device(self.cfg.DEVICE)
        loader_cfg_kwargs = dict(
            safety_margin_gb=self.cfg.MEMORY_SAFETY_MARGIN_GB,
            bytes_per_batch=self.batch_sample_bytes * batch_size,
            worker_ram_fraction=self.cfg.WORKER_RAM_FRACTION,
            worker_timeout_s=self.cfg.DATALOADER_WORKER_TIMEOUT_S,
            worker_count_override=self.cfg.WORKER_COUNT,
            metrics=monitor.snapshot(),
        )
        base_cfg = monitor.dataloader_config(device, **loader_cfg_kwargs)

        pad = tonic.collation.PadTensors(batch_first=False)
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=pad, shuffle=True, drop_last=True,**self.loader_kwargs(train_data, base_cfg),)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=pad,**self.loader_kwargs(test_data, base_cfg),)

        logger.info(f"[PIPELINE] Train batches : {len(train_loader)}")
        logger.info(f"[PIPELINE] Test batches  : {len(test_loader)}")
        logger.info(f"[PIPELINE] Batch size    : {batch_size}")

        self.train_loader = DeviceLoader(train_loader, device)
        self.test_loader  = DeviceLoader(test_loader,  torch.device(self.cfg.INFERENCE_DEVICE))

    def loader_kwargs(self, dataset, base_cfg: dict) -> dict:
        """Force single-process loading for a dataset that needs it (see requires_single_process_loading)."""
        if getattr(dataset, "requires_single_process_loading", False):
            cfg = {"num_workers": 0, "prefetch_factor": None, "pin_memory": False, "persistent_workers": False, "timeout": 0}
            logger.info("[PIPELINE] Single-process-only cache detected — forcing num_workers=0")
        else:
            cfg = base_cfg
        return {k: v for k, v in cfg.items() if v is not None}

    def get_dataloaders(self) -> tuple[DeviceLoader, DeviceLoader]:
        return self.train_loader, self.test_loader

    def clear_cache(self, split=None):
        self.controller.clear_cache(split)

    def validate_first_sample(self, dataset, split: str) -> int:
        events, _ = dataset[0]
        if events is None or (hasattr(events, "numel") and events.numel() == 0):
            raise ValueError(f"{split} dataset: first sample is empty")
        sample_bytes = events.nbytes
        logger.info(f"[VALIDATION] {split.capitalize()} dataset: first sample OK ({sample_bytes / 1024:.1f} KB)")
        return sample_bytes


def main() -> tuple[DeviceLoader, DeviceLoader]:
    from skeleton.config import Settings, load_config
    cfg = Settings(load_config())
    encoder = NeuromorphicEncoder(cfg, resolve_dataset_entry())
    return encoder.get_dataloaders()
