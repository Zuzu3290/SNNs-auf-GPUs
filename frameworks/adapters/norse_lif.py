"""Norse LIF, behind BaseLIF.

Uses LIFBoxCell, NOT LIFCell. LIFCell carries synaptic current as a second state (its
state fields are v AND i), making it a SECOND-ORDER neuron, while the other three
frameworks here are first-order. LIFBoxCell's state is (v,) only. Verified against
norse 1.1.0.
"""
from __future__ import annotations

import logging
from typing import Any

import norse.torch as norse
import torch
from norse.torch.functional.reset import reset_subtract, reset_value

from frameworks.adapters.base import BaseLIF
from skeleton.neuron_spec import (
    neuron_cfg, require_choice, require_float, require_surrogate,
)

logger = logging.getLogger(__name__)

RESETS = {"value": reset_value, "subtract": reset_subtract}
SURROGATES = ("circ", "super", "heaviside", "tanh", "triangle", "heavi_erfc")


def build_lif(cfg) -> norse.LIFBoxCell:
    n = neuron_cfg(cfg, "norse")
    require_choice(n, "cell", ["lif_box"])

    stype, salpha = require_surrogate(n)
    if stype not in SURROGATES:
        raise ValueError(f"neuron.norse.surrogate.type = {stype!r}; options {sorted(SURROGATES)}")
    if stype == "super":
        # MEASURED against norse 1.1.0: SuperSpike's backward never references
        # ctx.alpha, so every alpha gives byte-identical gradients -- it behaves as
        # alpha=1, whose fat tails measured 6.03x the gradient norm of the other
        # frameworks. Allowed, but never silently.
        logger.warning(
            "[NORSE] surrogate 'super' IGNORES alpha in norse 1.1.0. It behaves as "
            "alpha=1 and measured 6.03x the gradient norm of the other frameworks. "
            "'circ' with alpha=0.5 tracks ATan(2) far more closely (1.19x)."
        )

    params = norse.LIFBoxParameters(
        tau_mem_inv=torch.as_tensor(require_float(n, "tau_mem_inv"), dtype=torch.float32),
        v_leak=torch.as_tensor(require_float(n, "v_leak"), dtype=torch.float32),
        v_th=torch.as_tensor(require_float(n, "v_th"), dtype=torch.float32),
        v_reset=torch.as_tensor(require_float(n, "v_reset"), dtype=torch.float32),
        method=stype,
        alpha=torch.as_tensor(salpha, dtype=torch.float32),
        reset_method=RESETS[require_choice(n, "reset_method", sorted(RESETS))],
    )
    return norse.LIFBoxCell(p=params, dt=require_float(n, "dt"))


class NorseLIF(BaseLIF):
    """Norse hands state back to the caller; we hold it between timesteps.

    Also applies a fixed input gain. LIFBoxCell's update is

        v = (1 - dt*tau_mem_inv) * v + (dt*tau_mem_inv) * input

    so its decay and its input gain are the SAME quantity and cannot be set
    independently -- choosing decay 0.9 forces gain 0.1. Sinabs has the identical lock
    but exposes `norm_input` to break it; norse does not, so the gain is restored on
    the way in. `input_scale` in the neuron spec is that factor (0.1 * 10.0 = 1.0).
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.lif = build_lif(cfg)
        self.input_scale = require_float(neuron_cfg(cfg, "norse"), "input_scale")
        self.state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes, self.state = self.lif(x * self.input_scale, self.state)
        self._record(spikes)
        return spikes

    def reset(self) -> None:
        self.state = None

    def has_state(self) -> bool:
        return self.state is not None

    def membrane(self) -> torch.Tensor | None:
        return None if self.state is None else self.state.v

    def describe(self) -> dict[str, Any]:
        n = neuron_cfg(self.cfg, "norse")
        stype, salpha = require_surrogate(n)
        return {
            "framework": "norse",
            "cell": "lif_box",
            "dt": require_float(n, "dt"),
            "tau_mem_inv": require_float(n, "tau_mem_inv"),
            "v_th": require_float(n, "v_th"),
            "v_reset": require_float(n, "v_reset"),
            "v_leak": require_float(n, "v_leak"),
            "reset_method": require_choice(n, "reset_method", sorted(RESETS)),
            "input_scale": self.input_scale,
            "surrogate": f"{stype}(alpha={salpha})",
        }
