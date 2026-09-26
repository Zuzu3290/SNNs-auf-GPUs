"""The data_workflow.yaml settings: binning, temporal slicing, cache, resources.

Every value is read through skeleton/strict.py -- see that module for why. In short: a
misspelled key used to fall through to a literal in this file, so `n_time_bin: 40` (one
character short) silently ran the experiment at the base file's 16 and said nothing.
"""
from pathlib import Path

from skeleton.config_loader import load_config
from skeleton.strict import section

DEFAULT_YAML = Path(__file__).parent.parent / "configuration" / "data_workflow.yaml"


class WorkflowSettings:
    def __init__(self, config: dict | None = None, overlay: str | None = None):
        """
        config   an already-merged config dict (from skeleton.config_loader.load_config)
        overlay  path to one experiment overlay to merge over the three base files

        Both omitted loads the three base files, matching the previous behaviour of
        reading data_workflow.yaml directly. Taking the merged dict is what lets an
        experiment overlay override binning, temporal, cache or resource_policy the same
        way it overrides anything else -- previously this class read its own file, so
        an overlay could not reach it.

        PASS `config` WHENEVER ONE EXISTS. Constructing this with no argument inside a
        run that HAS a merged config silently reverts every key here to the base files.
        That is exactly what happened in event_data_workflow/data_pipeline.py: a
        run asking for binning.n_time_bins = 20 was framed at the base file's 16, with
        nothing in the output saying so.
        """
        if config is not None and overlay is not None:
            raise ValueError("pass either `config` or `overlay`, not both")
        config = config if config is not None else load_config(overlay)
        self.config = config

        binning = section(config, "binning")
        self.FRAME_MODE     = binning.require_choice("mode", ["n_time_bins", "time_window"])
        self.N_TIME_BINS    = binning.require_int("n_time_bins")
        self.TIME_WINDOW_US = int(binning.require_float("time_window_ms") * 1000)
        # Part of the cache identity: it changes which events exist. null disables.
        self.DENOISE_FILTER_TIME_US = binning.optional_int("denoise_filter_time_us")

        temporal = section(config, "temporal")
        self.TEMPORAL_SLICING_ENABLED = temporal.require_bool("enabled")
        # Slice length in microseconds, used when slicing by TIME. Deliberately not
        # called TEMPORAL_SLICE_DURATION_US: that spelling was read by code while never
        # existing, so getattr silently supplied 15 ms and every Hz figure came out ~20x
        # high. Tests assert that name stays absent.
        self.SLICE_DURATION_US = temporal.require_int("slice_duration_us")


        self.CACHE_PATH = section(config, "cache").require_str("path")

        rp = section(config, "resource_policy")
        self.MEMORY_SAFETY_MARGIN_GB   = rp.require_float("memory_safety_margin_gb")
        self.MEMORY_CACHE_THRESHOLD_GB = rp.require_float("memory_cache_threshold_gb")
        self.GPU_PRESSURE_THRESHOLD    = rp.require_float("gpu_pressure_threshold")
        self.BATCH_VRAM_FRACTION       = rp.require_float("batch_vram_fraction")
        self.BATCH_VRAM_BAND_MIN       = rp.require_float("batch_vram_band_min")
        self.MAX_BATCH_SIZE            = rp.require_int("max_batch_size")
        self.MEMORY_TIER_HEADROOM_FRACTION = rp.require_float("memory_tier_headroom_fraction")
        self.DISK_TIER_HEADROOM_MULTIPLE   = rp.require_float("disk_tier_headroom_multiple")
        self.WORKER_RAM_FRACTION           = rp.require_float("worker_ram_fraction")
        self.DATALOADER_WORKER_TIMEOUT_S   = rp.require_float("dataloader_worker_timeout_s")
        self.CALIBRATE_WORKERS             = rp.require_bool("calibrate_workers")
        self.WORKER_COUNT_FALLBACK         = rp.require_int("worker_count_fallback")

    @property
    def worker_count_override(self) -> int | None:
        """None when calibrate_workers is true (adaptive sizing); worker_count_fallback otherwise."""
        return None if self.CALIBRATE_WORKERS else self.WORKER_COUNT_FALLBACK
