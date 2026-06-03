from .spike_rate_bus import SpikeRateBus
from .phase_manager  import PhaseManager, Phase

# Prefer the C++ extension — gives direct cuMemPool API control.
# Falls back to the pure-Python implementation when the extension is not built.
try:
    from snn_runtime import MemoryArbiter, Zone  # type: ignore[import]
    CPP_ARBITER = True
except ImportError:
    from .memory_arbiter import MemoryArbiter, Zone  # type: ignore[assignment]
    CPP_ARBITER = False

__all__ = ["MemoryArbiter", "Zone", "SpikeRateBus", "PhaseManager", "Phase"]
