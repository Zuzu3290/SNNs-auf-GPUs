import yaml
from pathlib import Path

DEFAULT_YAML = Path(__file__).parent / "data_workflow.yaml"


class WorkflowSettings:
    def __init__(self, yaml_path: str = str(DEFAULT_YAML)):
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        cache = config.get("cache", {})
        self.CACHE_PATH               = cache.get("path", "./cache")
        self.MEMORY_SAFETY_MARGIN_GB  = float(cache.get("memory_safety_margin_gb", 2.0))
        self.MEMORY_CACHE_THRESHOLD_GB = float(cache.get("memory_cache_threshold_gb", 6.0))
        self.MAX_CACHED_RECORDINGS    = int(cache.get("max_cached_recordings", 500))
        self.CACHE_FORCE_MODE         = cache.get("force_mode", None)
