import yaml
from pathlib import Path

DEFAULT_YAML = Path(__file__).parent.parent / "configuration" / "data_workflow.yaml"


class WorkflowSettings:
    def __init__(self, config: dict | None = None, overlay: str | None = None):
        """
        config   an already-merged config dict (from skeleton.config_loader.load_config)
        overlay  path to one experiment overlay to merge over the three base files

        Both omitted loads the three base files, matching the previous behaviour of
        reading data_workflow.yaml directly. Taking the merged dict is what lets an
        experiment overlay override framing, slicing, cache or resource_policy the same
        way it overrides anything else -- previously this class read its own file, so
        an overlay could not reach it.
        """
        from skeleton.config_loader import load_config

        if config is not None and overlay is not None:
            raise ValueError("pass either `config` or `overlay`, not both")
        config = config if config is not None else load_config(overlay)

        framing = config.get("framing", {})
        self.FRAME_MODE     = framing.get("mode", "time_window")
        self.N_TIME_BINS    = int(framing.get("n_time_bins", 16))
        self.TIME_WINDOW_US = int(framing.get("time_window_ms", 15.0) * 1000)
        # null = "not known"; Hz is then reported as unavailable rather than guessed.
        sample_duration = framing.get("sample_duration_us", None)
        # Part of the cache identity: it changes which events exist. null disables.
        _denoise = framing.get("denoise_filter_time_us", 10000)
        self.DENOISE_FILTER_TIME_US = None if _denoise is None else int(_denoise)
        # Applied AFTER the cache, so toggling it needs no rebuild and it is NOT part of
        # the cache identity.
        self.BINARIZE               = bool(framing.get("binarize", False))
        self.SAMPLE_DURATION_US = int(sample_duration) if sample_duration is not None else None

        slicing = config.get("temporal_slicing", {})
        self.TEMPORAL_SLICING_ENABLED = bool(slicing.get("enabled", False))
        # null (default) -> SliceByTime, using SLICE_DURATION_US below. An int here
        # switches to SliceByEventCount instead. See CALIBRATE_EVENTS_PER_SLICE for
        # the third strategy.
        events_per_slice = slicing.get("events_per_slice", None)
        self.EVENTS_PER_SLICE = int(events_per_slice) if events_per_slice is not None else None
        # true -> SliceByEventCount with a value calibrated from the dataset's
        # own recordings (Case A) instead of a guessed constant; overrides
        # EVENTS_PER_SLICE above whenever both are set.
        self.CALIBRATE_EVENTS_PER_SLICE = bool(slicing.get("calibrate_events_per_slice", False))
        # Slice length in microseconds, used when slicing by TIME. Deliberately not
        # called TEMPORAL_SLICE_DURATION_US: that spelling was read by code while never
        # existing, so getattr silently supplied 15 ms and every Hz figure came out ~20x
        # high. Tests assert that name stays absent.
        self.SLICE_DURATION_US          = int(slicing.get("slice_duration_us", 15000))

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
