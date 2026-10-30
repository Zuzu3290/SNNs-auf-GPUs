"""snnTorch LIF, behind BaseLIF.

Neuron values come from the unified `neuron:` spec in network_architecture.yaml, read
via skeleton/config.py, which raises on a missing key rather than falling back to
an snnTorch default -- those defaults differ from the other three frameworks, which is
how the original per-framework config ended up with a 40x spread in firing threshold.
"""
from __future__ import annotations
import snntorch as snn
import torch
from snntorch import surrogate
from frameworks.adapters.base import BaseLIF, reconcile

SURROGATES = {"atan": surrogate.atan, "fast_sigmoid": surrogate.fast_sigmoid}


def _live_surrogate(spike_grad) -> tuple:
    """(type, alpha) of the surrogate the module is actually holding.

    snnTorch's `surrogate.atan(alpha=2.0)` returns a nested `inner` function rather than
    an object, so the type is in __qualname__ ("atan.<locals>.inner") and alpha survives
    only as a captured closure cell. Both are read defensively: a future snnTorch could
    return something else entirely, and an unreadable surrogate must degrade to "?"
    rather than take describe() down with it.
    """
    name = getattr(spike_grad, "__qualname__", type(spike_grad).__name__).split(".")[0]
    cells = [c.cell_contents for c in (getattr(spike_grad, "__closure__", None) or ())
             if isinstance(c.cell_contents, (int, float))]
    return name, (cells[0] if len(cells) == 1 else "?")


def build_lif(cfg) -> snn.Leaky:
    n = cfg.NEURON["snntorch"]
    stype, salpha = n["surrogate"]["type"], n["surrogate"]["alpha"]
    return snn.Leaky(
        beta=n["beta"],
        threshold=n["threshold"],
        spike_grad=SURROGATES[stype](salpha),
        # snnTorch defaults to "subtract" (soft reset). The other three hard-reset.
        reset_mechanism=n["reset_mechanism"],
        # snnTorch defaults reset_delay=True, applying a spike's reset on the FOLLOWING
        # timestep. The other three reset immediately. Left at the default, snnTorch's
        # membrane reads a different value than the others after every spike.
        reset_delay=n["reset_delay"],
        # init_hidden=False so state is passed explicitly -- see forward().
        init_hidden=False,
    )


class SnnTorchLIF(BaseLIF):
    """snnTorch hands state back to the caller; we hold it between timesteps.

    `init_hidden=True` would make snnTorch keep the membrane on the module instead,
    but it also registers the layer in a global instance list that `utils.reset()`
    walks PROCESS-WIDE -- resetting one model would reset every other snnTorch model
    alive in the process, including another framework's during a multi-framework run.
    Explicit state avoids that entirely.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.lif = build_lif(cfg)
        self.mem = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        spikes, self.mem = self.lif(x, self.mem)
        self._record(spikes)
        return spikes

    def reset(self) -> None:
        self.mem = None

    def describe(self) -> dict:
        """Read off the BUILT snn.Leaky -- these are the values training will use."""
        n = self.cfg.NEURON["snntorch"]
        stype, salpha = n["surrogate"]["type"], n["surrogate"]["alpha"]
        live_type, live_alpha = _live_surrogate(self.lif.spike_grad)
        return {
            "framework": "snntorch",
            "beta": reconcile(self.lif.beta, n["beta"]),
            "threshold": reconcile(self.lif.threshold, n["threshold"]),
            "reset_mechanism": reconcile(
                self.lif.reset_mechanism,
                n["reset_mechanism"],
            ),
            "reset_delay": reconcile(self.lif.reset_delay, n["reset_delay"]),
            "surrogate": reconcile(f"{live_type}(alpha={live_alpha})",
                                   f"{stype}(alpha={salpha})"),
        }
