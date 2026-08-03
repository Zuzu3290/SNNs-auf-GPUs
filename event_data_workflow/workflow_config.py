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
        self.SLICE_DURATION_MS        = float(slicing.get("slice_duration_ms", 15.0))

        cache = config.get("cache", {})
        self.CACHE_PATH                = cache.get("path", "./cache")
        self.MEMORY_SAFETY_MARGIN_GB   = float(cache.get("memory_safety_margin_gb", 2.0))
        self.MEMORY_CACHE_THRESHOLD_GB = float(cache.get("memory_cache_threshold_gb", 6.0))
        self.MAX_CACHED_RECORDINGS     = int(cache.get("max_cached_recordings", 500))
        # Adaptive on/off switch: None → probe live resources and pick a
        # strategy; a value here forces that strategy instead.
        self.CACHE_FORCE_MODE          = cache.get("force_mode", None)

        realtime = config.get("realtime", {})
        # {dataset_name: deadline_ms} — see docs/frameworks/realtime_nir_evaluation.md
        self.REALTIME_DEADLINE_MS = {k: float(v) for k, v in realtime.get("deadline_ms", {}).items()}
