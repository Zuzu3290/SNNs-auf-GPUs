"""SpikingJelly LIF, behind BaseLIF."""
from __future__ import annotations

from typing import Any

import torch
from spikingjelly.activation_based import neuron, surrogate

from frameworks.adapters.base import BaseLIF
from skeleton.neuron_spec import (
    neuron_cfg, require_bool, require_choice, require_float, require_surrogate,
)

SURROGATES = {"atan": surrogate.ATan, "sigmoid": surrogate.Sigmoid}


def build_lif(cfg) -> neuron.LIFNode:
    n = neuron_cfg(cfg, "spikingjelly")
    stype, salpha = require_surrogate(n)
    if stype not in SURROGATES:
        raise ValueError(
            f"neuron.spikingjelly.surrogate.type = {stype!r}; options are {sorted(SURROGATES)}"
        )

    step_mode = require_choice(n, "step_mode", ["s", "m"])
    if step_mode != "s":
        # The shared network hands every layer ONE timestep, so multi-step cannot be
        # wired in here. Refused outright rather than silently ignored: 'm' is also the
        # only mode that supports the fused cupy backend, so accepting it here would
        # imply an optimisation that is not actually running.
        raise ValueError(
            "neuron.spikingjelly.step_mode must be 's' for the shared per-timestep "
            "network. 'm' (and the cupy backend it enables) is a separate experiment -- "
            "verified equivalent in spikes, different only in speed."
        )

    return neuron.LIFNode(
        tau=require_float(n, "tau"),
        # SpikingJelly defaults decay_input=True, which divides the input by tau and
        # drops the input gain to 1/tau. False keeps gain at 1.0 like the others.
        decay_input=require_bool(n, "decay_input"),
        v_threshold=require_float(n, "v_threshold"),
        v_reset=require_float(n, "v_reset"),
        surrogate_function=SURROGATES[stype](alpha=salpha),
        # MEASURED: flipping this changed d(spikes)/d(weight) from 2.64 to 4.34 on a
        # 6-step test. False matches snnTorch, whose reset path also carries gradient.
        detach_reset=require_bool(n, "detach_reset"),
        step_mode=step_mode,
        backend=require_choice(n, "backend", ["torch", "cupy"]),
    )


class SpikingJellyLIF(BaseLIF):
    """SpikingJelly keeps state on the module and returns spikes alone."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.lif = build_lif(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes = self.lif(x)
        self._record(spikes)
        return spikes

    def reset(self) -> None:
        # Instance method, not the module-level functional.reset_net(), so this resets
        # THIS layer only.
        self.lif.reset()

    def has_state(self) -> bool:
        # v is the scalar 0.0 before the first forward, a tensor afterwards.
        return isinstance(getattr(self.lif, "v", 0.0), torch.Tensor)

    def membrane(self) -> torch.Tensor | None:
        v = getattr(self.lif, "v", None)
        return v if isinstance(v, torch.Tensor) else None

    def describe(self) -> dict[str, Any]:
        n = neuron_cfg(self.cfg, "spikingjelly")
        stype, salpha = require_surrogate(n)
        return {
            "framework": "spikingjelly",
            "tau": require_float(n, "tau"),
            "decay_input": require_bool(n, "decay_input"),
            "v_threshold": require_float(n, "v_threshold"),
            "v_reset": require_float(n, "v_reset"),
            "detach_reset": require_bool(n, "detach_reset"),
            "step_mode": require_choice(n, "step_mode", ["s", "m"]),
            "backend": require_choice(n, "backend", ["torch", "cupy"]),
            "surrogate": f"{stype}(alpha={salpha})",
        }
