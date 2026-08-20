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
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Dataset
import tonic
import tonic.transforms as transforms
import tonic.slicers as slicers
import torchvision
import tqdm as t
from typing import Optional
from skeleton import Settings, WorkflowSettings
from .cache_engine import AdaptiveCacheController, measure_event_bytes
from .system_monitor import monitor
from .dataset_registry import resolve_dataset_entry
from .prefetch import AsyncGPUPrefetcher, CudaPrefetcher


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


def calibrate_events_per_slice(dataset: Dataset, target_bins_per_recording: int = 8,
                                min_events_per_slice: int = 100, sample_recordings: int = 200) -> int:
    """Picks events_per_slice for SliceByEventCount from this dataset's own
    event-rate statistics, instead of a guessed constant.

    Anchors on the 10th percentile of sampled per-recording event counts
    (not the median), so a below-typical-length recording still clears
    target_bins_per_recording slices. SliceByEventCount itself clamps
    event_count = min(event_count, n_events), so a too-large constant
    doesn't drop a short recording, it just returns one whole-recording
    slice with no subdivision -- the failure mode this anchor avoids."""
    sample_size = min(sample_recordings, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size]
    event_counts = [len(dataset[int(i)][0]) for i in indices]

    anchor = int(np.percentile(event_counts, 10))
    return max(min_events_per_slice, anchor // target_bins_per_recording)

# Show progress bars for large downloads in bytes instead of raw item counts.
orig_tqdm_init = t.tqdm.__init__
def mb_init(self, *a, **kw):
    if (kw.get("total") or 0) > 1_000_000:
        kw.setdefault("unit", "B")
        kw.setdefault("unit_scale", True)
        kw.setdefault("unit_divisor", 1024)
    orig_tqdm_init(self, *a, **kw)
t.tqdm.__init__ = mb_init


class PrefetchedLoader:
    """The class training and testing actually use. Underneath, it just
    combines the two prefetchers in prefetch.py: one to fetch data on the
    CPU, one to move it onto the GPU ahead of time."""

    def __init__(self, loader, device: torch.device, depth: int = 1, queue_size: int | None = None):
        self.loader = loader
        self.device = device
        self.depth = max(1, depth)
        # Unset queue_size defaults to depth: the raw CPU-side buffer feeding
        # the CUDA-stream copies must be at least as deep as the device-
        # resident buffer it feeds, or it becomes the tighter bottleneck and
        # throttles the GPU below what `depth` was chosen to sustain.
        self.queue_size = max(1, queue_size if queue_size is not None else self.depth)
        self.current: CudaPrefetcher | None = None

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        if self.current is not None:
            self.current.stop()
        async_stage = AsyncGPUPrefetcher(self.loader, queue_size=self.queue_size)
        self.current = CudaPrefetcher(async_stage, self.device, depth=self.depth)
        yield from self.current


class NeuromorphicEncoder:
    """Loads a dataset, caches it, and builds the train/test DataLoaders used by training."""

    def __init__(self, cfg: Settings, use_temporal_slicing: bool | None = None, slice_duration_ms: float | None = None, events_per_slice: int | None = None, calibrate_events_per_slice: bool | None = None):

        self.cfg = cfg
        self.wf  = WorkflowSettings()

        # Configure the shared SystemResourceMonitor once, early, now that
        # the run's device and cache path are both known — every consumer
        # (AdaptiveCacheController, monitor.dataloader_config(), SNNTrainer,
        # SNNTester) reads live state through this same instance instead of
        # each building its own.
        cuda_enabled = torch.device(cfg.DEVICE).type == "cuda"
        monitor.configure(cache_path=self.wf.CACHE_PATH, cuda_enabled=cuda_enabled)

        metrics = monitor.snapshot()
        physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
        logger.info(
            f"[PIPELINE] System: {physical_cores} physical cores, "
            f"{metrics.available_ram_gb:.2f}GB RAM free, "
            f"{metrics.gpu_available_gb:.2f}GB VRAM free — before any dataset operation"
        )

        self.use_temporal_slicing = use_temporal_slicing if use_temporal_slicing is not None else self.wf.TEMPORAL_SLICING_ENABLED
        self.slice_duration_ms = slice_duration_ms or (cfg.TEMPORAL_SLICE_DURATION / 1000.0)
        self.events_per_slice = events_per_slice if events_per_slice is not None else self.wf.EVENTS_PER_SLICE
        self.calibrate_events_per_slice = (
            calibrate_events_per_slice if calibrate_events_per_slice is not None else self.wf.CALIBRATE_EVENTS_PER_SLICE
        )
        self.train_loader: PrefetchedLoader
        self.test_loader: PrefetchedLoader
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

        # One real transform, paid once here, so create_loaders() can size
        # DataLoader workers from the actual post-transform frame a batch is
        # made of — not the pre-transform raw event bytes (which can be
        # larger or smaller than the dense frame depending on sensor size).
        first_frame = frame_tf(raw_train[0][0])
        self.batch_sample_bytes = measure_event_bytes(first_frame)

        return raw_train, raw_test, frame_tf

    def apply_pipeline(self, raw_train, raw_test, frame_tf):
        """Cache the raw recordings, then optionally slice them into temporal windows."""
        self.controller = AdaptiveCacheController(
            cache_path=self.wf.CACHE_PATH,
            memory_safety_margin_gb=self.wf.MEMORY_SAFETY_MARGIN_GB,
            memory_cache_threshold_gb=self.wf.MEMORY_CACHE_THRESHOLD_GB,
            gpu_pressure_threshold=self.wf.GPU_PRESSURE_THRESHOLD,
            memory_tier_headroom_fraction=self.wf.MEMORY_TIER_HEADROOM_FRACTION,
            disk_tier_headroom_multiple=self.wf.DISK_TIER_HEADROOM_MULTIPLE,
        )
        controller = self.controller

        # train_augment is random, so it must run fresh every access rather
        # than get baked into a persistent cache — see determine_dataset_strategy.
        train_augment = torchvision.transforms.RandomRotation([-10, 10])
        train_tf = transforms.Compose([frame_tf, torch.from_numpy, train_augment])
        test_tf = frame_tf

        dataset_prefix = self.dataset_label.replace(" ", "_")

        if self.use_temporal_slicing:

            if self.calibrate_events_per_slice:
                self.events_per_slice = calibrate_events_per_slice(raw_train)
                logger.info(f"[PIPELINE] Case A calibration: events_per_slice={self.events_per_slice} (from raw_train)")

            # Cache raw recordings first — slicing needs the raw timestamps.
            cached_train = controller.determine_dataset_strategy(raw_train, split=f"{dataset_prefix}/train")
            cached_test  = controller.determine_dataset_strategy(raw_test,  split=f"{dataset_prefix}/test")

            metadata_dir = str(PROJECT_ROOT / "metadata" / dataset_prefix)
            train_data = create_sliced_dataset(cached_train,
                slice_duration_ms=self.slice_duration_ms,
                events_per_slice=self.events_per_slice, transform=train_tf,
                metadata_path=f"{metadata_dir}/train",
            )
            test_data = create_sliced_dataset(cached_test,
                slice_duration_ms=self.slice_duration_ms,
                events_per_slice=self.events_per_slice, transform=test_tf,
                metadata_path=f"{metadata_dir}/test",
            )
            logger.info(f"[PIPELINE] After slicing — train: {len(train_data)}, test: {len(test_data)}")
            if len(train_data) == 0 or len(test_data) == 0:
                raise RuntimeError(
                    f"[PIPELINE] Temporal slicing produced an empty dataset — "
                    f"train: {len(train_data)} samples, test: {len(test_data)} samples. "
                    "Increase architecture.temporal_slice_duration in SNN_module.yaml."
                )
        else:
            # Cache the deterministic frame transform; keep the random
            # augmentation out of the cached value (transform/live_transform split).
            train_data = controller.determine_dataset_strategy(raw_train, transform=frame_tf, live_transform=train_augment, split=f"{dataset_prefix}/train")
            test_data  = controller.determine_dataset_strategy(raw_test,  transform=test_tf,  split=f"{dataset_prefix}/test")

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
        """Build the train/test DataLoaders, with worker config sized from live
        RAM, wrapped in PrefetchedLoader so training/inference never need to
        wrap them a second time — batches arrive already device-resident."""
        batch_size = self.cfg.BATCH_SIZE
        device = torch.device(self.cfg.DEVICE)
        base_cfg = monitor.dataloader_config(device, safety_margin_gb=self.wf.MEMORY_SAFETY_MARGIN_GB)

        # Regression targets (MVSEC's tuple, TUM-VIE's dict) aren't torch.tensor()-able —
        # tonic's own PadTensors would crash on them. Only classification datasets get it.
        if getattr(self, "task_type", "classification") == "regression":
            pad = pad_events_passthrough_target
        else:
            pad = tonic.collation.PadTensors(batch_first=False)

        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=pad, shuffle=True, drop_last=True,
            **self.loader_kwargs(train_data, base_cfg),
        )
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=pad,
            **self.loader_kwargs(test_data, base_cfg),
        )

        logger.info(f"[PIPELINE] Train batches : {len(train_loader)}")
        logger.info(f"[PIPELINE] Test batches  : {len(test_loader)}")
        logger.info(f"[PIPELINE] Batch size    : {batch_size}")

        prefetch_depth = self.compute_prefetch_depth(batch_size)
        logger.info(f"[PIPELINE] Prefetch depth: {prefetch_depth}")
        self.train_loader = PrefetchedLoader(train_loader, device, depth=prefetch_depth)
        self.test_loader  = PrefetchedLoader(test_loader,  device, depth=prefetch_depth)

    def compute_prefetch_depth(self, batch_size: int) -> int:
        """How many batches to keep queued ahead of the GPU, sized from live
        VRAM and this dataset's real per-sample size — not a fixed constant,
        so a large-sensor dataset (bigger batches) or a smaller card (less
        headroom) both get a depth that actually fits, instead of one number
        tuned for whichever dataset/GPU it happened to be set on.

        This runs after calibrate_batch_size() has already freed its own
        probe allocations but before real training has claimed anything —
        a live VRAM snapshot at this point looks more available than it's
        about to be. Subtracting batch_vram_fraction of total VRAM (the
        share calibrate_batch_size already earmarked for the real training
        step) before sizing the queue keeps the two from double-booking the
        same memory. Fraction/min/max all read from resource_policy in
        data_workflow.yaml, not fixed here."""
        if not self.wf.CALIBRATE_PREFETCH_DEPTH:
            return self.wf.PREFETCH_DEPTH_FALLBACK
        batch_bytes = self.batch_sample_bytes * batch_size
        metrics = monitor.snapshot()
        reserved_for_training_gb = metrics.gpu_memory_gb * self.wf.BATCH_VRAM_FRACTION
        true_available_gb = max(0.0, metrics.gpu_available_gb - reserved_for_training_gb)
        if batch_bytes <= 0 or true_available_gb <= 0:
            return self.wf.PREFETCH_DEPTH_MIN
        budget_bytes = true_available_gb * self.wf.PREFETCH_VRAM_FRACTION * (1024 ** 3)
        return max(self.wf.PREFETCH_DEPTH_MIN, min(int(budget_bytes / batch_bytes), self.wf.PREFETCH_DEPTH_MAX))

    def loader_kwargs(self, dataset, base_cfg: dict) -> dict:
        """Force single-process loading for a dataset that needs it (see requires_single_process_loading)."""
        if getattr(dataset, "requires_single_process_loading", False):
            cfg = {"num_workers": 0, "prefetch_factor": None, "pin_memory": False, "persistent_workers": False}
            logger.info("[PIPELINE] Single-process-only cache detected — forcing num_workers=0")
        else:
            cfg = base_cfg
        return {k: v for k, v in cfg.items() if v is not None}

    def get_dataloaders(self) -> tuple[PrefetchedLoader, PrefetchedLoader]:
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


def main() -> tuple[PrefetchedLoader, PrefetchedLoader]:
    cfg = Settings()
    encoder = NeuromorphicEncoder(cfg)
    return encoder.get_dataloaders()
