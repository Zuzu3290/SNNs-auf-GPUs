import yaml
from pathlib import Path

DEFAULT_YAML = Path(__file__).parent.parent / "configuration" / "data_workflow.yaml"


class WorkflowSettings:
    def __init__(self, yaml_path: str = str(DEFAULT_YAML)):
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        framing = config.get("framing", {})
        self.FRAME_MODE     = framing.get("mode", "time_window")
        self.N_TIME_BINS    = int(framing.get("n_time_bins", 16))
        self.TIME_WINDOW_US = int(framing.get("time_window_ms", 15.0) * 1000)

        slicing = config.get("temporal_slicing", {})
        self.TEMPORAL_SLICING_ENABLED = bool(slicing.get("enabled", False))
        # null (default) -> SliceByTime, using cfg.TEMPORAL_SLICE_DURATION (the
        # "normal"/timing-window method, set in SNN_module.yaml). An int here
        # switches slicing to SliceByEventCount instead. See
        # CALIBRATE_EVENTS_PER_SLICE below for the third strategy.
        events_per_slice = slicing.get("events_per_slice", None)
        self.EVENTS_PER_SLICE = int(events_per_slice) if events_per_slice is not None else None
        # true -> SliceByEventCount with a value calibrated from the dataset's
        # own recordings (Case A) instead of a guessed constant; overrides
        # EVENTS_PER_SLICE above whenever both are set.
        self.CALIBRATE_EVENTS_PER_SLICE = bool(slicing.get("calibrate_events_per_slice", False))

        augmentation = config.get("augmentation", {})
        self.RANDOM_ROTATION_ENABLED = bool(augmentation.get("random_rotation_enabled", True))

        cache = config.get("cache", {})
        self.CACHE_PATH = cache.get("path", "./cache")

        rp = config.get("resource_policy", {})
        self.MEMORY_SAFETY_MARGIN_GB   = float(rp.get("memory_safety_margin_gb", 2.0))
        self.MEMORY_CACHE_THRESHOLD_GB = float(rp.get("memory_cache_threshold_gb", 6.0))
        self.GPU_PRESSURE_THRESHOLD    = float(rp.get("gpu_pressure_threshold", 0.75))
        self.BATCH_VRAM_FRACTION       = float(rp.get("batch_vram_fraction", 0.35))
        self.BATCH_VRAM_BAND_MIN       = float(rp.get("batch_vram_band_min", 0.30))
        self.BATCH_VRAM_BAND_MAX       = float(rp.get("batch_vram_band_max", 0.35))
        self.MAX_BATCH_SIZE            = int(rp.get("max_batch_size", 256))
        self.CALIBRATE_PREFETCH_DEPTH  = bool(rp.get("calibrate_prefetch_depth", True))
        self.PREFETCH_DEPTH_FALLBACK   = int(rp.get("prefetch_depth_fallback", 8))
        self.PREFETCH_VRAM_FRACTION    = float(rp.get("prefetch_vram_fraction", 0.10))
        self.PREFETCH_DEPTH_MIN        = int(rp.get("prefetch_depth_min", 1))
        self.PREFETCH_DEPTH_MAX        = int(rp.get("prefetch_depth_max", 32))
        self.MEMORY_TIER_HEADROOM_FRACTION = float(rp.get("memory_tier_headroom_fraction", 0.7))
        self.DISK_TIER_HEADROOM_MULTIPLE   = float(rp.get("disk_tier_headroom_multiple", 1.2))
        self.WORKER_RAM_FRACTION           = float(rp.get("worker_ram_fraction", 0.25))
        self.DATALOADER_WORKER_TIMEOUT_S   = float(rp.get("dataloader_worker_timeout_s", 60.0))
        self.CALIBRATE_WORKERS             = bool(rp.get("calibrate_workers", True))
        self.WORKER_COUNT_FALLBACK         = int(rp.get("worker_count_fallback", 4))

    @property
    def worker_count_override(self) -> int | None:
        """None when calibrate_workers is true (adaptive sizing); worker_count_fallback otherwise."""
        return None if self.CALIBRATE_WORKERS else self.WORKER_COUNT_FALLBACK
