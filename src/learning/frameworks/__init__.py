# Lazy imports — each framework loads only when first accessed.
# Importing this package does not pull in snntorch, norse, or spikingjelly
# until the specific class is actually used.

def __getattr__(name: str):
    if name == "SNN_TORCH":
        from .snn_torch import SNN_TORCH
        return SNN_TORCH
    if name == "SNN_NORSE":
        from .snn_norse import SNN_NORSE
        return SNN_NORSE
    if name == "SNN_SJ":
        from .snn_spikingjelly import SNN_SJ
        return SNN_SJ
    raise AttributeError(f"module 'learning.frameworks' has no attribute {name!r}")

__all__ = ["SNN_TORCH", "SNN_NORSE", "SNN_SJ"]