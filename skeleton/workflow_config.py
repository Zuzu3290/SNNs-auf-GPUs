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

        cache = config.get("cache", {})
        self.CACHE_PATH                = cache.get("path", "./cache")
        self.MEMORY_SAFETY_MARGIN_GB   = float(cache.get("memory_safety_margin_gb", 2.0))
        self.MEMORY_CACHE_THRESHOLD_GB = float(cache.get("memory_cache_threshold_gb", 6.0))
        self.MAX_CACHED_RECORDINGS     = int(cache.get("max_cached_recordings", 500))
