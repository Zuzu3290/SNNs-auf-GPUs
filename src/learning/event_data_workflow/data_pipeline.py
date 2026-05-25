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
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # src/ → skeleton
sys.path.insert(0, str(Path(__file__).parent))                # sibling modules

import torch
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
import torchvision
from skeleton import Settings
from cache_engine import AdaptiveCacheController, PipelineMemoryCoordinator
from temporal_slicer import create_sliced_dataset

_WORKFLOW_YAML = Path(__file__).parent / "data_workflow.yaml"

# Maps framework name → tensor layout expected by that framework's forward()
_FORMAT_MAP = {
    "torch":        "TB",   # [T, B, C, H, W]
    "norse":        "TB",   # [T, B, C, H, W]
    "spikingjelly": "BT",   # [B, T, C, H, W]
}

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "tmp" / "data"

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


def _probe_sample_bytes(dataset, n: int = 5, fallback: int = 50_000) -> int:
    """
    Measure average bytes per item by accessing up to n real samples from the dataset.
    Works for any dataset/dtype/shape — no hardcoded assumptions.
    Falls back to `fallback` only if all probes fail (e.g. empty dataset).
    """
    total, count = 0, 0
    for i in range(min(n, len(dataset))):
        try:
            item, _ = dataset[i]
            if hasattr(item, 'nbytes'):            # numpy array
                total += int(item.nbytes)
            elif hasattr(item, 'element_size'):    # torch tensor
                total += item.element_size() * item.numel()
            else:
                total += sys.getsizeof(item)
            count += 1
        except Exception:
            continue
    measured = (total // count) if count > 0 else fallback
    logger.info(
        f"[PIPELINE] Sample size probe: {measured // 1024} KB/sample "
        f"(from {count} sample{'s' if count != 1 else ''})"
    )
    return measured


class _Collate:
    """
    Collate for n_time_bins mode — fixed T so all tensors are the same shape.
    torch.stack works directly; no padding needed.
    Must be a top-level class (not a closure) to be picklable by DataLoader workers.
    """
    def __init__(self, data_format: str):
        self.data_format = data_format

    def __call__(self, batch):
        samples, labels = zip(*batch)
        x = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in samples])
        y = torch.tensor(labels, dtype=torch.long)
        if self.data_format == "TB":
            x = x.permute(1, 0, 2, 3, 4).contiguous()
        return x, y


class _FramedDataset(torch.utils.data.Dataset):
    """
    Applies frame_tf inside __getitem__ so DiskCachedDataset stores the already-framed
    tensor, not raw event structs.  Matches old pipeline pattern:
        DiskCachedDataset(NMNIST(transform=frame_tf), transform=aug_tf)
    Must be a top-level class (not a closure) to be picklable by DataLoader workers.
    """
    def __init__(self, dataset, transform):
        self.dataset   = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        events, label = self.dataset[idx]
        return self.transform(events), label


class _PaddedCollate:
    """
    Collate for time_window mode — T varies per recording so we pad to the
    longest sequence in the batch, then permute to the requested layout.
    Must be a top-level class (not a closure) to be picklable by DataLoader workers.
    """
    def __init__(self, data_format: str):
        self.data_format = data_format

    def __call__(self, batch):
        samples, labels = zip(*batch)
        tensors = [torch.as_tensor(s, dtype=torch.float32) for s in samples]  # each [T_i, C, H, W]
        max_t = max(t.shape[0] for t in tensors)
        padded = torch.zeros(len(tensors), max_t, *tensors[0].shape[1:])
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t
        y = torch.tensor(labels, dtype=torch.long)
        if self.data_format == "TB":
            padded = padded.permute(1, 0, 2, 3, 4).contiguous()
        return padded, y


class NeuromorphicEncoder:
    """
    Builds train and test DataLoaders for neuromorphic event-based datasets.

    Framing mode and slicing are driven by data_workflow.yaml (same directory).
    Tensor layout is resolved automatically from the framework name:
        "torch" / "norse"  → [T, B, C, H, W]
        "spikingjelly"     → [B, T, C, H, W]

    Args:
        cfg       : Settings object from SNN_module.yaml (training/model params).
        framework : Name of the SNN framework being used — drives tensor layout.
    """

    def __init__(self, cfg: Settings, framework: str = "torch"):
        self.cfg         = cfg
        self.wf          = self._load_workflow()
        self.data_format = _FORMAT_MAP.get(framework.lower(), "TB")
        self.framework   = framework.lower()

        framing = self.wf.get("framing", {})
        self.frame_mode     = framing.get("mode", "n_time_bins")
        self.n_time_bins    = int(framing.get("n_time_bins", 16))
        self.time_window_us = int(framing.get("time_window_ms", 15.0) * 1000)

        preprocessing = self.wf.get("preprocessing", {})
        self.denoise_filter_time_us = int(preprocessing.get("denoise_filter_time_us", 10000))

        augmentation = self.wf.get("augmentation", {})
        self.augmentation_enabled = bool(augmentation.get("enabled", True))
        self.rotation_deg         = float(augmentation.get("rotation_deg", 10))

        cache_cfg = self.wf.get("cache", {})
        self.cache_path                 = cache_cfg.get("path", "./cache")
        self.avg_recording_bytes        = int(cache_cfg.get("avg_recording_bytes", 50_000))
        self.memory_safety_margin_gb    = float(cache_cfg.get("memory_safety_margin_gb", 2.0))
        self.memory_cache_threshold_gb  = float(cache_cfg.get("memory_cache_threshold_gb", 8.0))
        self.force_mode                 = cache_cfg.get("force_mode", None)  # null | disk | memory | hybrid

        slicing = self.wf.get("temporal_slicing", {})
        self.use_temporal_slicing = bool(slicing.get("enabled", False))
        self.slice_duration_ms    = float(slicing.get("slice_duration_ms", 15.0))
        self.slice_overlap_ms     = float(slicing.get("overlap_ms", 0.0))
        self.events_per_slice     = slicing.get("events_per_slice", None)
        self.auto_tune_slicing    = bool(slicing.get("auto_tune", False))

        self.train_loader: DataLoader
        self.test_loader: DataLoader

        self.coordinator = PipelineMemoryCoordinator.from_system()
        self.build()

    def _load_workflow(self) -> dict:
        with open(_WORKFLOW_YAML, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _print_config(self):
        cfg = self.cfg
        W = 64
        sep  = "=" * W
        sep2 = "-" * W

        print(sep)
        print(f"  PIPELINE CONFIG  —  {self.framework.upper()}  [{self.data_format}]")
        print(sep)

        # ── Dataset ─────────────────────────────────────────────────────────
        print(f"  Dataset            : {cfg.DATASET_NAME}")
        print(f"  Device             : {cfg.DEVICE}")

        # ── Framing ─────────────────────────────────────────────────────────
        print(sep2)
        if self.frame_mode == "n_time_bins":
            print(f"  Framing mode       : n_time_bins  →  fixed T = {self.n_time_bins}")
        else:
            print(f"  Framing mode       : time_window  →  {self.time_window_us / 1000:.1f} ms/frame  (variable T)")

        # ── Architecture ────────────────────────────────────────────────────
        print(sep2)
        print(f"  Sensor             : {cfg.SENSOR_H} × {cfg.SENSOR_W}  ({cfg.IN_CHANNELS} polarity channels)")
        print(f"  Conv1              : out={cfg.CONV1_OUT}  kernel={cfg.CONV1_KERNEL}×{cfg.CONV1_KERNEL}")
        print(f"  Conv2              : out={cfg.CONV2_OUT}  kernel={cfg.CONV2_KERNEL}×{cfg.CONV2_KERNEL}")
        print(f"  Pool kernel        : {cfg.POOL_KERNEL}×{cfg.POOL_KERNEL}")
        print(f"  FC_IN (auto)       : {cfg.FC_IN}   ← computed from conv/pool dims, not hardcoded")
        print(f"  Output classes     : {cfg.OUTPUT_SIZE}")
        print(f"  Threshold          : {cfg.THRESHOLD}")

        # ── Neuron params ────────────────────────────────────────────────────
        print(sep2)
        if self.framework == "torch":
            print(f"  SNNTorch  beta     : {cfg.BETA}   (membrane decay per timestep, 0–1)")
        elif self.framework == "norse":
            print(f"  Norse  tau_mem_inv : {cfg.TAU_MEM_INV} Hz  (inverse membrane time constant, valid 20–100)")
        elif self.framework == "spikingjelly":
            print(f"  SpikingJelly  tau  : {cfg.TAU}   (membrane time constant ratio)")

        # ── Training ────────────────────────────────────────────────────────
        print(sep2)
        eff_batch   = cfg.BATCH_SIZE * cfg.GRAD_ACCUM_STEPS
        total_itera = cfg.EPOCHS * cfg.ITERA
        print(f"  Epochs             : {cfg.EPOCHS}")
        print(f"  Iterations/epoch   : {cfg.ITERA}   (= 60k samples / batch_size {cfg.BATCH_SIZE})")
        print(f"  Total iterations   : {total_itera}")
        print(f"  Batch size         : {cfg.BATCH_SIZE}   effective = {eff_batch}  (×{cfg.GRAD_ACCUM_STEPS} grad_accum)")
        print(f"  Timesteps          : {cfg.TIMESTEPS}")
        print(f"  Learning rate      : {cfg.LEARNING_RATE}")
        print(f"  Weight decay       : {cfg.WEIGHT_DECAY}")
        print(f"  LR scheduler       : {cfg.LR_SCHEDULER}")
        print(f"  Loss               : {cfg.LOSS_FN}")
        print(f"  Optimizer          : {cfg.OPTIMIZER_TYPE}")
        print(f"  Surrogate          : {cfg.SURROGATE}")
        print(f"  AMP                : {cfg.USE_AMP}{'   ← safe for this framework' if not cfg.USE_AMP else '   ← watch for float16 underflow in deep SNNs'}")
        print(f"  num_workers        : {cfg.NUM_WORKERS}{'   ← single-process (Colab-safe)' if cfg.NUM_WORKERS == 0 else '   ← multiprocess prefetch'}")

        # ── Preprocessing / augmentation ─────────────────────────────────────
        print(sep2)
        print(f"  Denoise filter     : {self.denoise_filter_time_us} µs  (removes noise events closer than this)")
        aug_str = f"ON  ±{self.rotation_deg}° rotation (training only)" if self.augmentation_enabled else "OFF"
        print(f"  Augmentation       : {aug_str}")
        sli_str = f"ON  {self.slice_duration_ms} ms slices" if self.use_temporal_slicing else "OFF  (n_time_bins framing used)"
        print(f"  Temporal slicing   : {sli_str}")

        # ── Cache config ──────────────────────────────────────────────────────
        print(sep2)
        fm = self.force_mode if self.force_mode else "null  (adaptive — will probe RAM at startup)"
        print(f"  Cache force_mode   : {fm}")
        print(f"  Cache path         : {self.cache_path}")
        print(f"  Memory safety margin : {self.memory_safety_margin_gb} GB  (reserved for OS + model)")
        print(f"  Memory cache threshold: {self.memory_cache_threshold_gb} GB  (min free RAM to consider memory mode)")

        print(sep)

    def build(self):
        self._print_config()
        raw_train, raw_test, frame_tf = self.load_raw()
        train_data, test_data = self.apply_pipeline(raw_train, raw_test, frame_tf)
        self.create_loaders(train_data, test_data)
        self._print_ready_banner()

    def load_raw(self):
        """Load raw event datasets with no transform — caching and slicing come later."""
        data_path = self.cfg.DATA_PATH

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
        logger.info(f"[PIPELINE] Dataset    : {self.dataset_label}  |  Sensor H={H} W={W} C={C}")
        logger.info(f"[PIPELINE] Frame mode : {self.frame_mode}")
        logger.info(f"[PIPELINE] Framework  : {self.framework}  →  layout {self.data_format}")
        logger.info(f"[PIPELINE] Train      : {len(raw_train)} recordings")
        logger.info(f"[PIPELINE] Test       : {len(raw_test)} recordings")

        self.validate_first_sample(raw_train, "train")
        self.validate_first_sample(raw_test, "test")

        if self.frame_mode == "n_time_bins":
            to_frame = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.n_time_bins)
        else:  # time_window
            to_frame = transforms.ToFrame(sensor_size=sensor_size, time_window=self.time_window_us)

        frame_tf = transforms.Compose([transforms.Denoise(filter_time=self.denoise_filter_time_us), to_frame])
        return raw_train, raw_test, frame_tf

    def apply_pipeline(self, raw_train, raw_test, frame_tf):
        """
        Layer 1 — Cache raw recordings via AdaptiveCacheController.
        Layer 2 — Temporal slicing (optional, stateless).

        Caching always wraps raw datasets so all slices from one recording
        share a single cache entry.
        """
        if self.augmentation_enabled:
            aug_only_tf = transforms.Compose([
                torch.from_numpy,
                torchvision.transforms.RandomRotation([-self.rotation_deg, self.rotation_deg]),
            ])
        else:
            aug_only_tf = torch.from_numpy

        # Full transform (frame + aug) used only for temporal slicing, where the cache
        # stores raw events so slicing can operate on timestamps.
        if self.augmentation_enabled:
            full_train_tf = transforms.Compose([
                frame_tf,
                torch.from_numpy,
                torchvision.transforms.RandomRotation([-self.rotation_deg, self.rotation_deg]),
            ])
        else:
            full_train_tf = frame_tf

        # Build the dataset that the cache will actually wrap — before creating the controller —
        # so we can probe the real per-sample byte size and give the coordinator an accurate budget.
        # Slicing path: cache stores raw events (timestamps needed for slicing).
        # No-slicing path: _FramedDataset bakes ToFrame in; cache stores framed [T,C,H,W] tensors.
        if self.use_temporal_slicing:
            probe_dataset = raw_train
        else:
            framed_train = _FramedDataset(raw_train, frame_tf)
            framed_test  = _FramedDataset(raw_test,  frame_tf)
            probe_dataset = framed_train

        avg_bytes = _probe_sample_bytes(probe_dataset, fallback=self.avg_recording_bytes)
        max_recs  = self.coordinator.max_recordings(avg_recording_bytes=avg_bytes)

        cache_path = self.cache_path if Path(self.cache_path).is_absolute() else str(PROJECT_ROOT / self.cache_path.lstrip("./"))
        controller = AdaptiveCacheController(
            cache_path=cache_path,
            memory_safety_margin_gb=self.memory_safety_margin_gb,
            memory_cache_threshold_gb=self.memory_cache_threshold_gb,
            max_cached_recordings=max_recs,
            verbose=True,
        )

        if self.use_temporal_slicing:
            # Layer 1: cache raw events (slicing needs raw timestamps)
            cached_train = controller.wrap_dataset(raw_train, split="train", force_mode=self.force_mode)
            cached_test  = controller.wrap_dataset(raw_test,  split="test",  force_mode=self.force_mode)

            # Layer 2: stateless slicing with transforms applied per slice
            train_data = create_sliced_dataset(
                cached_train,
                slice_duration_ms=self.slice_duration_ms,
                auto_tune=self.auto_tune_slicing,
                events_per_slice=self.events_per_slice,
                transform=full_train_tf,
                verbose=True,
            )
            test_data = create_sliced_dataset(
                cached_test,
                slice_duration_ms=self.slice_duration_ms,
                transform=frame_tf,
                verbose=True,
            )
            logger.info(f"[PIPELINE] After slicing — train: {len(train_data)}, test: {len(test_data)}")
        else:
            # ToFrame (expensive) runs once per sample via _FramedDataset; cache stores the result.
            # Augmentation (RandomRotation) re-runs on every access — fresh rotation each epoch.
            train_data = controller.wrap_dataset(framed_train, transform=aug_only_tf, split="train", force_mode=self.force_mode)
            test_data  = controller.wrap_dataset(framed_test,  transform=None,        split="test",  force_mode=self.force_mode)

        return train_data, test_data

    def create_loaders(self, train_data, test_data):
        """Build DataLoaders; worker count capped by cfg.NUM_WORKERS (training.num_workers in YAML)."""
        batch_size = self.cfg.BATCH_SIZE
        dl_cfg = {
            k: v for k, v in self.coordinator.dataloader_config().items()
            if v is not None  # prefetch_factor must be omitted (not None) when num_workers=0
        }

        # YAML training.num_workers acts as a hard cap — set to 0 on Colab to avoid OOM.
        yaml_workers = getattr(self.cfg, "NUM_WORKERS", 4)
        if yaml_workers < dl_cfg.get("num_workers", 4):
            dl_cfg["num_workers"] = yaml_workers
            if yaml_workers == 0:
                dl_cfg.pop("prefetch_factor", None)
                dl_cfg["persistent_workers"] = False

        # n_time_bins → fixed T, stack directly.  time_window → variable T, pad to max in batch.
        if self.frame_mode == "n_time_bins":
            collate = _Collate(self.data_format)
        else:
            collate = _PaddedCollate(self.data_format)

        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            collate_fn=collate,
            shuffle=True,
            drop_last=True,
            **dl_cfg,
        )
        self.test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            collate_fn=collate,
            **dl_cfg,
        )

        logger.info(f"[PIPELINE] Train batches : {len(self.train_loader)}")
        logger.info(f"[PIPELINE] Test batches  : {len(self.test_loader)}")
        logger.info(f"[PIPELINE] Batch size    : {batch_size}")

        # Human-readable DataLoader summary
        W = 64
        print("-" * W)
        print("  DataLoader config (final — after YAML cap):")
        print(f"    Train batches      : {len(self.train_loader)}")
        print(f"    Test  batches      : {len(self.test_loader)}")
        print(f"    batch_size         : {batch_size}")
        print(f"    num_workers        : {dl_cfg.get('num_workers', 0)}")
        print(f"    pin_memory         : {dl_cfg.get('pin_memory', False)}")
        print(f"    persistent_workers : {dl_cfg.get('persistent_workers', False)}")
        if "prefetch_factor" in dl_cfg:
            print(f"    prefetch_factor    : {dl_cfg['prefetch_factor']}")
        collate_name = "Padded (variable T)" if self.frame_mode == "time_window" else "Fixed-T stack"
        print(f"    collate            : {collate_name}  →  layout [{self.data_format}]")
        if self.frame_mode == "n_time_bins":
            T = self.n_time_bins
        else:
            T = "variable"
        C, H, W_s = self.cfg.IN_CHANNELS, self.cfg.SENSOR_H, self.cfg.SENSOR_W
        if self.data_format == "TB":
            shape = f"[T={T}, B={batch_size}, C={C}, H={H}, W={W_s}]"
        else:
            shape = f"[B={batch_size}, T={T}, C={C}, H={H}, W={W_s}]"
        print(f"    batch tensor shape : {shape}")

    def _print_ready_banner(self):
        W = 64
        sep = "=" * W
        print(sep)
        print("  PIPELINE READY")
        print(sep)
        print(f"  Framework          : {self.framework.upper()}")
        print(f"  Train batches      : {len(self.train_loader)}  × batch {self.cfg.BATCH_SIZE}  = {len(self.train_loader) * self.cfg.BATCH_SIZE} samples/epoch")
        print(f"  Test  batches      : {len(self.test_loader)}")
        print(f"  Epochs planned     : {self.cfg.EPOCHS}  ({len(self.train_loader) * self.cfg.EPOCHS} total iterations)")
        fm = self.force_mode.upper() if self.force_mode else "ADAPTIVE"
        print(f"  Cache mode         : {fm}")
        if self.force_mode == "disk" or (not self.force_mode):
            print(f"  Epoch 1 speed      : SLOW  (ToFrame runs for every sample → disk write)")
            print(f"  Epoch 2+ speed     : FAST  (load framed tensor from cache → augment only)")
        elif self.force_mode == "memory":
            print(f"  Epoch 1 speed      : SLOW  (ToFrame runs for every cold-cache sample)")
            print(f"  Epoch 2+ speed     : VERY FAST  (framed tensors in RAM)")
        print(sep)

    def get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader) for use in the training loop."""
        return self.train_loader, self.test_loader

    def clear_cache(self):
        """Clear all disk cache directories."""
        import shutil
        cache_path = PROJECT_ROOT / "cache"
        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info("[PIPELINE] Cache cleared")

    def validate_first_sample(self, dataset, split: str) -> None:
        try:
            events, target = dataset[0]
            if events is None or (hasattr(events, "numel") and events.numel() == 0):
                raise ValueError("first sample is empty")
            logger.info(f"[VALIDATION] {split.capitalize()} dataset: first sample OK")
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


if __name__ == "__main__":
    train_loader, test_loader = main()
    logger.info(f"[PIPELINE] Ready — {len(train_loader)} train batches, {len(test_loader)} test batches")