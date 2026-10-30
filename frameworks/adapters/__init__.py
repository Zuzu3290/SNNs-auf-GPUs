"""One LIF adapter per framework; each is imported only when chosen, so a run loads a single SNN library."""
from __future__ import annotations

from frameworks.adapters.base import BaseLIF


def adapter_class(framework: str) -> type[BaseLIF]:
    """The LIF adapter class for a cfg.FRAMEWORK value."""
    if framework == "torch":
        from frameworks.adapters.snntorch_lif import SnnTorchLIF
        return SnnTorchLIF
    if framework == "norse":
        from frameworks.adapters.norse_lif import NorseLIF
        return NorseLIF
    if framework == "sj":
        from frameworks.adapters.spikingjelly_lif import SpikingJellyLIF
        return SpikingJellyLIF
    if framework == "sinabs":
        from frameworks.adapters.sinabs_lif import SinabsLIF
        return SinabsLIF
    raise ValueError(f"unknown framework {framework!r}; options: torch, norse, sj, sinabs")


__all__ = ["BaseLIF", "adapter_class"]
