"""
Builds train/test DataLoaders for neuromorphic event datasets: load raw
recordings, cache them via AdaptiveCacheController, optionally slice into
temporal windows, then wrap in DataLoaders.
"""
from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "tmp" / "data"
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import DataLoader, Dataset
import tonic
import tonic.transforms as transforms
import tonic.slicers as slicers
import torchvision
import tqdm as t
from typing import Optional
from skeleton import Settings
from .cache_engine import AdaptiveCacheController, measure_event_bytes
from .system_monitor import SystemResourceMonitor
from .workflow_config import WorkflowSettings
from .dataset_registry import resolve_dataset_entry


class LabelToIndex:
    """Picklable string-label -> int-index mapper (see cache_engine.ComposedTransform
    for why a closure/lambda can't be used here — Windows' spawn-based multiprocessing
    can't pickle nested functions, only importable module-level classes/functions).

    Some tonic dataset classes (e.g. NCALTECH101) hand back the raw class-folder name
    as the target instead of an integer index — unlike ASLDVS/DVSGesture/NMNIST, which
    already map to ints internally. Nothing downstream (PadTensors' collate_fn calls
    torch.tensor(target)) can handle a string/bytes target, so it must be mapped to an
    int before caching/batching. Built from sorted(set(labels)) for a deterministic
    mapping regardless of directory-walk order."""

    def __init__(self, labels):
        classes = sorted(set(labels))
        self.mapping = {label: idx for idx, label in enumerate(classes)}

    def __call__(self, label):
        return self.mapping[label]


def apply_label_to_index(*datasets: Dataset) -> None:
    """If any given dataset's raw .targets are non-int labels (e.g. NCALTECH101's
    class-folder-name strings), build ONE LabelToIndex mapping from the union of
    every dataset's targets and set it as each dataset's target_transform. A single
    shared mapping (rather than one per dataset) keeps train/test class indices
    consistent even if one split happens to be missing a class the other has —
    matters for datasets passed here as two separately-constructed objects
    (has_train_split=True), not just one dataset split via random_split()."""
    all_targets = [target for dataset in datasets for target in getattr(dataset, "targets", [])]
    if not all_targets or isinstance(all_targets[0], (int, bool)):
        return
    mapper = LabelToIndex(all_targets)
    for dataset in datasets:
        dataset.target_transform = mapper


def pad_events_passthrough_target(batch):
    """Same event-frame padding/stacking as tonic.collation.PadTensors(batch_first=False).
    The target side stacks into a real tensor when every sample's target has the same
    shape (true for DSEC's per-window flow frames — (H, W, 3), fixed for a given
    dataset) — the trainer needs a real tensor, not a list, to move to device. Falls
    back to a plain list when shapes differ, since torch.tensor(target) would crash
    on a genuinely irregular/heterogeneous target (unused today, kept for safety —
    e.g. a future regression dataset that mixes recording-level target types)."""
    samples = [sample for sample, _ in batch]
    targets = [target for _, target in batch]

    max_length = max(s.shape[0] for s in samples)
    padded = []
    for sample in samples:
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample)
        sample = torch.cat((
            sample,
            torch.zeros(max_length - sample.shape[0], *sample.shape[1:], device=sample.device),
        ))
        padded.append(sample)

    samples_output = torch.stack(padded, 1)  # batch_first=False, matches PadTensors elsewhere in this pipeline

    target_shapes = {getattr(t, "shape", None) for t in targets}
    if len(target_shapes) == 1 and None not in target_shapes:
        targets_output = torch.stack([torch.as_tensor(t) for t in targets])
    else:
        targets_output = targets

    return samples_output, targets_output


def dataloader_config(settings: Settings, device: torch.device, safety_margin_gb: float = 2.0, worker_fraction: float = 0.3, batch_bytes: int = 0) -> dict:
    """num_workers/prefetch/pin_memory/persistent_workers, sized from live
    RAM. Drops to num_workers=0 when the worker RAM budget is under 500MB —
    a GPU-only embedded run with no host RAM headroom for multiprocessing workers."""
    cuda_enabled = device is not None and getattr(device, "type", "") == "cuda"
    device_idx = (device.index or 0) if device is not None and cuda_enabled else 0
    metrics = SystemResourceMonitor(device_idx=device_idx, cuda_enabled=cuda_enabled).snapshot()
    total_gb = max(1.0, metrics.available_ram_gb - safety_margin_gb)
    worker_budget_gb = total_gb * worker_fraction
    gpu_only = cuda_enabled and worker_budget_gb < 0.5

    if gpu_only:
        cfg = {
            "num_workers":        0,
            "prefetch_factor":    None,
            "pin_memory":         False,
            "persistent_workers": False,
        }
        logger.info("[PIPELINE] GPU-only mode — num_workers=0, pin_memory=False")
    else:
        worker_bytes = worker_budget_gb * (1024 ** 3)
        if batch_bytes > 0:
            max_workers = max(1, int(worker_bytes / (2 * batch_bytes)))
        else:
            max_workers = settings.NUM_WORKERS
        max_workers = min(max_workers, os.cpu_count() or 1)
        cfg = {
            "num_workers":        max_workers,
            "prefetch_factor":    2 if max_workers > 0 else None,
            "pin_memory":         True,
            "persistent_workers": max_workers > 0,
        }

    logger.info(f"[PIPELINE] DataLoader config: {cfg}")
    return cfg


def create_sliced_dataset(
    dataset: Dataset,
    slice_duration_ms: float = 15.0,
    overlap_ms: float = 0.0,
    events_per_slice: Optional[int] = None,
    transform=None,
    metadata_path: Optional[str] = None,
) -> tonic.SlicedDataset:
    """Wrap dataset with tonic's SlicedDataset. metadata_path, if given,
    stores the slice index as HDF5 so it isn't rebuilt on later runs."""
    if events_per_slice is not None:
        slicer = slicers.SliceByEventCount(event_count=events_per_slice)
    else:
        slicer = slicers.SliceByTime(
            time_window=slice_duration_ms * 1000,
            overlap=overlap_ms * 1000,
        )

    return tonic.SlicedDataset(dataset, slicer=slicer, transform=transform, metadata_path=metadata_path)  # type: ignore[arg-type]

# Show progress bars for large downloads in bytes instead of raw item counts.
orig_tqdm_init = t.tqdm.__init__
def mb_init(self, *a, **kw):
    if (kw.get("total") or 0) > 1_000_000:
        kw.setdefault("unit", "B")
        kw.setdefault("unit_scale", True)
        kw.setdefault("unit_divisor", 1024)
    orig_tqdm_init(self, *a, **kw)
t.tqdm.__init__ = mb_init


class NeuromorphicEncoder:
    """Loads a dataset, caches it, and builds the train/test DataLoaders used by training."""

    def __init__(self, cfg: Settings, use_temporal_slicing: bool | None = None, slice_duration_ms: float | None = None, events_per_slice: int | None = None, cache_force_mode: str | None = None):

        self.cfg = cfg
        self.wf  = WorkflowSettings()
        if cache_force_mode is not None:
            self.wf.CACHE_FORCE_MODE = cache_force_mode
        self.use_temporal_slicing = use_temporal_slicing if use_temporal_slicing is not None else self.wf.TEMPORAL_SLICING_ENABLED
        self.slice_duration_ms = slice_duration_ms or (cfg.TEMPORAL_SLICE_DURATION / 1000.0)
        self.events_per_slice = events_per_slice
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.build()

    def build(self):
        raw_train, raw_test, frame_tf = self.load_raw()
        train_data, test_data = self.apply_pipeline(raw_train, raw_test, frame_tf)
        self.create_loaders(train_data, test_data)

    def select_dataset(self) -> dict:
        """See module-level resolve_dataset_entry — kept as a method for callers that
        already have a NeuromorphicEncoder instance."""
        return resolve_dataset_entry(self.cfg)

    def load_raw(self):
        """Load the raw dataset (no transform, no cache yet) and build the frame transform for it."""
        entry = self.select_dataset()
        sensor_size = entry["sensor_size"]
        if entry.get("kind") == "regression":
            # DSEC's own "test" split has no local ground truth at all (it's held
            # out for their own leaderboard — tonic's DSEC class raises if you ask
            # for target_selection there). So: load DSEC's "train" split (the
            # recordings that actually have optical-flow/disparity ground truth),
            # then split across recordings ourselves, same as the other manually-
            # split datasets.
            full_raw = entry["loader"](str(DATA_DIR), split="train")
            n_train = int(0.8 * len(full_raw))
            raw_train, raw_test = torch.utils.data.random_split(
                full_raw, [n_train, len(full_raw) - n_train]
            )
        elif entry["has_train_split"]:
            raw_train = entry["cls"](save_to=str(DATA_DIR), train=True)
            raw_test  = entry["cls"](save_to=str(DATA_DIR), train=False)
            apply_label_to_index(raw_train, raw_test)
        else:
            full_raw = entry["cls"](save_to=str(DATA_DIR))
            apply_label_to_index(full_raw)
            n_train = int(0.8 * len(full_raw))
            raw_train, raw_test = torch.utils.data.random_split(
                full_raw, [n_train, len(full_raw) - n_train]
            )
        self.dataset_label = entry["name"]
        self.task_type = entry.get("kind", "classification")
        self.cfg.TASK_TYPE = self.task_type  # so SNNTrainer/SNNTester can branch without a separate cfg wiring step
        self.cfg.apply_dataset_shape(
            sensor_h=sensor_size[1], sensor_w=sensor_size[0],
            in_channels=sensor_size[2], num_classes=entry["num_classes"],
        )

        self.sensor_size = sensor_size
        W, H, C = sensor_size
        logger.info(f"[PIPELINE] Dataset : {self.dataset_label}  |  Sensor H={H} W={W} C={C}")
        logger.info(f"[PIPELINE] Train   : {len(raw_train)} recordings")
        logger.info(f"[PIPELINE] Test    : {len(raw_test)} recordings")

        train_sample_bytes = self.validate_first_sample(raw_train, "train")
        self.validate_first_sample(raw_test, "test")
        logger.info(f"[PIPELINE] First train sample size: {train_sample_bytes / 1024:.1f} KB — dataset non-empty, proceeding to cache strategy")

        if self.wf.FRAME_MODE == "n_time_bins":
            to_frame = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.wf.N_TIME_BINS)
        else:
            to_frame = transforms.ToFrame(sensor_size=sensor_size, time_window=self.wf.TIME_WINDOW_US)
        frame_tf = transforms.Compose([transforms.Denoise(filter_time=10000), to_frame])
        return raw_train, raw_test, frame_tf

    def apply_pipeline(self, raw_train, raw_test, frame_tf):
        """Cache the raw recordings, then optionally slice them into temporal windows."""
        self.controller = AdaptiveCacheController(
            cache_path=self.wf.CACHE_PATH,
            memory_safety_margin_gb=self.wf.MEMORY_SAFETY_MARGIN_GB,
            memory_cache_threshold_gb=self.wf.MEMORY_CACHE_THRESHOLD_GB,
            max_cached_recordings=self.wf.MAX_CACHED_RECORDINGS,
            device=torch.device(self.cfg.DEVICE),
        )
        controller = self.controller

        # train_augment is random, so it must run fresh every access rather
        # than get baked into a persistent cache — see determine_dataset_strategy.
        train_augment = torchvision.transforms.RandomRotation([-10, 10])
        train_tf = transforms.Compose([frame_tf, torch.from_numpy, train_augment])
        test_tf = frame_tf

        num_workers = self.cfg.NUM_WORKERS

        if self.use_temporal_slicing:
            # Cache raw recordings first — slicing needs the raw timestamps.
            cached_train = controller.determine_dataset_strategy(raw_train, split="train", num_workers=num_workers, force_mode=self.wf.CACHE_FORCE_MODE)
            cached_test  = controller.determine_dataset_strategy(raw_test,  split="test",  num_workers=num_workers, force_mode=self.wf.CACHE_FORCE_MODE)

            metadata_dir = str(PROJECT_ROOT / "metadata")
            train_data = create_sliced_dataset(cached_train,
                slice_duration_ms=self.slice_duration_ms,
                events_per_slice=self.events_per_slice, transform=train_tf,
                metadata_path=f"{metadata_dir}/train",
            )
            test_data = create_sliced_dataset(cached_test,
                slice_duration_ms=self.slice_duration_ms, transform=test_tf,
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
            # Cache the deterministic frame transform; keep the random
            # augmentation out of the cached value (transform/live_transform split).
            train_data = controller.determine_dataset_strategy(raw_train, transform=frame_tf, live_transform=train_augment, split="train", num_workers=num_workers, force_mode=self.wf.CACHE_FORCE_MODE)
            test_data  = controller.determine_dataset_strategy(raw_test,  transform=test_tf,  split="test",  num_workers=num_workers, force_mode=self.wf.CACHE_FORCE_MODE)

        # tonic's DiskCachedDataset/MemoryCachedDataset and torch's Subset (from
        # random_split) don't forward attribute access to the wrapped dataset, so a
        # raw dataset's requires_single_process_loading (e.g. MVSEC/TUM-VIE holding
        # rosbag/h5py file handles) would otherwise silently get lost by the time
        # create_loaders/loader_kwargs checks it. Carry it forward explicitly.
        source = getattr(raw_train, "dataset", raw_train)
        if getattr(source, "requires_single_process_loading", False):
            setattr(train_data, "requires_single_process_loading", True)
            setattr(test_data, "requires_single_process_loading", True)

        return train_data, test_data

    def create_loaders(self, train_data, test_data):
        """Build the train/test DataLoaders, with worker config sized from live RAM."""
        batch_size = self.cfg.BATCH_SIZE
        base_cfg = dataloader_config(self.cfg, torch.device(self.cfg.DEVICE))

        # Regression targets (MVSEC's tuple, TUM-VIE's dict) aren't torch.tensor()-able —
        # tonic's own PadTensors would crash on them. Only classification datasets get it.
        if getattr(self, "task_type", "classification") == "regression":
            pad = pad_events_passthrough_target
        else:
            pad = tonic.collation.PadTensors(batch_first=False)

        self.train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=pad, shuffle=True, drop_last=True,
            **self.loader_kwargs(train_data, base_cfg),
        )
        self.test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=pad,
            **self.loader_kwargs(test_data, base_cfg),
        )

        logger.info(f"[PIPELINE] Train batches : {len(self.train_loader)}")
        logger.info(f"[PIPELINE] Test batches  : {len(self.test_loader)}")
        logger.info(f"[PIPELINE] Batch size    : {batch_size}")

    def loader_kwargs(self, dataset, base_cfg: dict) -> dict:
        """Force single-process loading for a dataset that needs it (see requires_single_process_loading)."""
        if getattr(dataset, "requires_single_process_loading", False):
            cfg = {"num_workers": 0, "prefetch_factor": None, "pin_memory": False, "persistent_workers": False}
            logger.info("[PIPELINE] Single-process-only cache detected — forcing num_workers=0")
        else:
            cfg = base_cfg
        return {k: v for k, v in cfg.items() if v is not None}

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        return self.train_loader, self.test_loader

    def clear_cache(self, split: str | None = None):
        self.controller.clear_cache(split)

    def validate_first_sample(self, dataset, split: str) -> int:
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
    cfg = Settings()
    encoder = NeuromorphicEncoder(cfg)
    return encoder.get_dataloaders()
