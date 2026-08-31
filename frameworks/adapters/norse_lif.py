"""Norse LIF, behind BaseLIF.

Uses LIFBoxCell, NOT LIFCell. LIFCell carries synaptic current as a second state (its
state fields are v AND i), making it a SECOND-ORDER neuron, while the other three
frameworks here are first-order. LIFBoxCell's state is (v,) only. Verified against
norse 1.1.0.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

with warnings.catch_warnings():
    # Importing norse re-registers LIFParameters/LIFBoxParameters -- both namedtuple
    # subclasses -- with torch's pytree registry, which torch >= 2.4 warns about twice
    # on every import. It says nothing about this pipeline and nothing the reader can
    # act on, and it buries the one norse warning that DOES matter (see build_lif).
    warnings.filterwarnings("ignore", message=r".*is a subclass of `collections\.namedtuple`.*")
    import norse.torch as norse

import torch
from norse.torch.functional.reset import reset_subtract, reset_value

from frameworks.adapters.base import BaseLIF, reconcile, scalar
from skeleton.neuron_spec import (
    neuron_cfg, require_choice, require_float, require_surrogate,
)

logger = logging.getLogger(__name__)

RESETS = {"value": reset_value, "subtract": reset_subtract}
SURROGATES = ("circ", "super", "heaviside", "tanh", "triangle", "heavi_erfc")

# Guards the SuperSpike alpha warning below -- see build_lif for why once per process.
_ALPHA_WARNING_SHOWN = False


def reset_alpha_warning() -> None:
    """Re-arm the once-per-process SuperSpike warning. For tests only: a suppression
    that cannot be cleared is a suppression nothing can prove still fires."""
    global _ALPHA_WARNING_SHOWN
    _ALPHA_WARNING_SHOWN = False


def build_lif(cfg) -> norse.LIFBoxCell:
    global _ALPHA_WARNING_SHOWN
    n = neuron_cfg(cfg, "norse")
    require_choice(n, "cell", ["lif_box"])

    stype, salpha = require_surrogate(n)
    if stype not in SURROGATES:
        raise ValueError(f"neuron.norse.surrogate.type = {stype!r}; options {sorted(SURROGATES)}")
    if stype == "super" and not _ALPHA_WARNING_SHOWN:
        # MEASURED against norse 1.1.0: SuperSpike's backward never references
        # ctx.alpha, so every alpha gives byte-identical gradients -- it behaves as
        # alpha=1, whose fat tails measured 6.03x the gradient norm of the other
        # frameworks. Allowed, but never silently.
        #
        # Once per process, not once per layer. The network builds one of these per
        # LIF layer and check_network builds the whole network four times over, so an
        # unguarded warning printed the same three lines repeatedly and read as three
        # separate problems. It is one fact about the config.
        _ALPHA_WARNING_SHOWN = True
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
        """Read off the BUILT LIFBoxCell -- these are the values training will use.

        Everything except dt lives in `p`, the LIFBoxParameters namedtuple the cell was
        constructed with, so the values below are the ones its forward step reads.
        """
        n = neuron_cfg(self.cfg, "norse")
        stype, salpha = require_surrogate(n)
        p = self.lif.p
        reset_key = require_choice(n, "reset_method", sorted(RESETS))
        return {
            "framework": "norse",
            # Not reconciled: this adapter builds exactly one cell type, and build_lif
            # already rejects any other value for `cell`.
            "cell": type(self.lif).__name__,
            "dt": reconcile(self.lif.dt, require_float(n, "dt")),
            "tau_mem_inv": reconcile(p.tau_mem_inv, require_float(n, "tau_mem_inv")),
            "v_th": reconcile(p.v_th, require_float(n, "v_th")),
            "v_reset": reconcile(p.v_reset, require_float(n, "v_reset")),
            "v_leak": reconcile(p.v_leak, require_float(n, "v_leak")),
            # norse stores the reset as a FUNCTION, so the check is identity against the
            # one this adapter's table maps the config key to, not a name comparison.
            "reset_method": reconcile(getattr(p.reset_method, "__name__", p.reset_method),
                                      reset_key,
                                      agrees=p.reset_method is RESETS[reset_key]),
            # Applied by this adapter on the way in, not by norse -- see the class
            # docstring. Live by construction: forward() multiplies by this attribute.
            "input_scale": self.input_scale,
            # p.alpha is stored faithfully; norse 1.1.0 simply never reads it back for
            # 'super'. That is the warning build_lif raises, not a wiring mismatch.
            "surrogate": reconcile(f"{p.method}(alpha={scalar(p.alpha)})",
                                   f"{stype}(alpha={salpha})"),
        }
