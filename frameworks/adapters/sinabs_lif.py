"""Sinabs LIF, behind BaseLIF.

Three sinabs defaults would each silently break the comparison, and all three are set
explicitly from the neuron spec:

    norm_input=True   multiplies input by (1 - alpha), dropping input gain to 0.049 at
                      tau_mem=20. The same decay/gain lock norse has structurally, but
                      sinabs lets one boolean break it -- which is why sinabs needs no
                      input_scale where norse does.
    spike_fn          defaults to MultiSpike: ONE neuron may emit 2, 3 or more spikes
                      in a SINGLE timestep. The other three emit a binary spike.
    reset_fn          defaults to MembraneSubtract (soft reset).

None of these is an upstream bug: no-leak + multi-spike + subtract-reset is exactly
what makes a spiking neuron's firing rate equal ReLU, which is the ANN-to-SNN
conversion guarantee sinabs exists to provide. It is simply a different goal.
"""
from __future__ import annotations

from typing import Any

import sinabs.layers as sl
import torch
from sinabs.activation import (
    Gaussian, Heaviside, MembraneReset, MembraneSubtract, MultiGaussian,
    MultiSpike, PeriodicExponential, SingleExponential, SingleSpike,
)

from frameworks.adapters.base import BaseLIF, reconcile, scalar
from skeleton.neuron_spec import (
    NeuronSpecError, neuron_cfg, optional_float, require_bool, require_choice,
    require_float,
)

SPIKE_FNS = {"single": SingleSpike, "multi": MultiSpike}
RESETS = {"zero": MembraneReset, "subtract": MembraneSubtract}

# Surrogate gradients, with the config keys each one needs.
#
# sinabs does NOT use the name `alpha` for any of them, so this project's usual
# `surrogate.alpha` key does not apply here -- each type declares its own real
# parameter names instead. Pretending they were all `alpha` would misreport what was
# actually configured, which is why this adapter reads the surrogate itself rather than
# going through neuron_spec.require_surrogate() like the other three.
SURROGATE_PARAMS = {
    "single_exponential": ("grad_width", "grad_scale"),
    "periodic_exponential": ("grad_width", "grad_scale"),
    "gaussian": ("mu", "sigma", "grad_scale"),
    "multi_gaussian": ("mu", "sigma", "h", "s", "grad_scale"),
    "heaviside": ("window",),
}


def build_surrogate(n: dict) -> Any:
    """Construct the surrogate gradient object from the sinabs neuron block.

    Only the keys belonging to the CHOSEN type are read, so a config carries the
    parameters of one surrogate rather than the union of all five.

    Wired explicitly rather than left to sinabs' own default: ex2 selects
    `periodic_exponential`, and a default-only adapter would accept that config and
    silently run single_exponential instead.
    """
    # neuron_spec's helpers key off the LAST path segment, so the surrogate sub-block
    # has to be fetched first and read from directly; the dotted path is only there to
    # make the error message name the full location.
    if "surrogate" not in n:
        raise NeuronSpecError(
            "neuron.sinabs.surrogate is not set. State it explicitly -- sinabs' own "
            f"default is single_exponential. Keys present: {sorted(n)}"
        )
    surrogate_block = n["surrogate"]
    if not isinstance(surrogate_block, dict):
        raise NeuronSpecError(
            f"neuron.sinabs.surrogate must be a mapping with a 'type', got "
            f"{surrogate_block!r}"
        )

    kind = require_choice(surrogate_block, "sinabs.surrogate.type", sorted(SURROGATE_PARAMS))

    def param(name: str) -> float:
        return require_float(surrogate_block, f"sinabs.surrogate.{name}")

    if kind == "single_exponential":
        return SingleExponential(
            grad_width=param("grad_width"), grad_scale=param("grad_scale")
        )
    if kind == "periodic_exponential":
        return PeriodicExponential(
            grad_width=param("grad_width"), grad_scale=param("grad_scale")
        )
    if kind == "gaussian":
        return Gaussian(
            mu=param("mu"), sigma=param("sigma"), grad_scale=param("grad_scale")
        )
    if kind == "multi_gaussian":
        return MultiGaussian(
            mu=param("mu"), sigma=param("sigma"), h=param("h"), s=param("s"),
            grad_scale=param("grad_scale"),
        )
    return Heaviside(window=param("window"))


SURROGATE_CLASSES = {
    "single_exponential": SingleExponential, "periodic_exponential": PeriodicExponential,
    "gaussian": Gaussian, "multi_gaussian": MultiGaussian, "heaviside": Heaviside,
}


def _live_surrogate_description(fn: Any) -> str:
    """The same string _surrogate_description produces, built from the OBJECT the module
    holds so the two can be compared directly. Falls back to the class name for a
    surrogate this adapter does not build."""
    for kind, cls in SURROGATE_CLASSES.items():
        if type(fn) is cls:
            shown = ", ".join("{}={}".format(name, getattr(fn, name, "?"))
                              for name in SURROGATE_PARAMS[kind])
            return "{}({})".format(kind, shown)
    return type(fn).__name__


def _surrogate_description(n: dict) -> str:
    """e.g. "periodic_exponential(grad_width=0.5, grad_scale=1.0)" for the run record."""
    block = n.get("surrogate")
    if not isinstance(block, dict) or "type" not in block:
        return "(not set)"
    kind = str(block["type"])
    names = SURROGATE_PARAMS.get(kind, ())
    shown = ", ".join(f"{name}={block.get(name)}" for name in names)
    return f"{kind}({shown})"


def build_lif(cfg) -> sl.LIF:
    n = neuron_cfg(cfg, "sinabs")
    reset_key = require_choice(n, "reset_mechanism", sorted(RESETS))
    if reset_key == "zero":
        reset_fn = MembraneReset(reset_value=require_float(n, "v_reset"))
    else:
        # MembraneSubtract's subtract_value=None means "subtract the threshold".
        reset_fn = MembraneSubtract()

    return sl.LIF(
        # In timesteps. sinabs has no dt, so tau_mem IS the time constant -- no unit
        # conversion, unlike norse's dt * tau_mem_inv.
        tau_mem=require_float(n, "tau_mem"),
        # null = first-order (membrane only). A number adds a synaptic state.
        tau_syn=optional_float(n, "tau_syn"),
        spike_threshold=torch.as_tensor(require_float(n, "spike_threshold")),
        spike_fn=SPIKE_FNS[require_choice(n, "spike_fn", sorted(SPIKE_FNS))],
        reset_fn=reset_fn,
        min_v_mem=optional_float(n, "min_v_mem"),
        train_alphas=require_bool(n, "train_alphas"),
        norm_input=require_bool(n, "norm_input"),
        # BACKWARD PASS ONLY -- forward membrane/spike dynamics are identical whatever
        # this is. Set explicitly so a config that names a surrogate actually gets it.
        surrogate_grad_fn=build_surrogate(n),
    )


class SinabsLIF(BaseLIF):
    """Sinabs keeps state in a buffer, and is written for WHOLE SEQUENCES."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.lif = build_lif(cfg)

        # Freeze the neuron's own parameters -- in practice tau_mem, or alpha_mem if
        # train_alphas were true.
        #
        # ⚠ train_alphas does NOT decide WHETHER a time-constant parameter exists, only
        # WHICH quantity it is. Either way sinabs leaves an nn.Parameter in the layer,
        # so the optimiser learns it: MEASURED as 18,257 trainable params against
        # 18,254 for the other three, one extra per spiking layer.
        #
        # Unconditional rather than configurable, because there is nothing here to
        # decide: snnTorch, SpikingJelly and Norse all hold their time constants as
        # plain constants and none of them can learn one in this pipeline. A flag would
        # only offer the choice of making sinabs incomparable. "Learn the time
        # constants" is a different experiment that all four would have to join.
        for parameter in self.lif.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Charge, fire, reset for ONE timestep.

        The unsqueeze/squeeze is not cosmetic. Sinabs unpacks its input as

            batch_size, time_steps, *trailing = input_data.shape

        so a single timestep must arrive as (batch, 1, C, H, W). Passing
        (batch, C, H, W) would be accepted without complaint and would treat the
        CHANNEL axis as time -- wrong answers, no error.

        Because the membrane persists in a buffer between calls, feeding one timestep
        per call is equivalent to feeding the whole sequence at once, the same way
        SpikingJelly's step_mode 's' relates to 'm'. That equivalence is what lets
        sinabs live inside the shared per-timestep network at all.
        """
        spikes = self.lif(x.unsqueeze(1)).squeeze(1)
        self._record(spikes)
        return spikes

    def reset(self) -> None:
        """Drop the state by restoring its buffers to zero-size.

        Sinabs' own call is `reset_states()`, and we deliberately do NOT use it. Its
        body is `buffer.zero_()`: it zeroes the VALUES but keeps the buffer's SHAPE,
        which leaves the layer "initialised" for one specific batch size. Two problems
        follow, both avoided here:

          1. A short final batch then takes sinabs'
             `handle_state_batch_size_mismatch` path, which resamples the state with
             `torch.randint` -- it fills each neuron's membrane by copying a RANDOM
             other sample's. Harmless only while the buffer happens to be all zeros.
          2. `is_state_initialised()` would keep returning True after a reset, so
             has_state() could not honestly report whether the reset worked.

        Restoring the zero-size tensor that register_buffer originally held makes
        is_state_initialised() False again, so the next forward re-infers the shape
        from its actual input -- the same "None means fresh neuron" semantics the
        snnTorch and Norse adapters have.
        """
        for name, buffer in list(self.lif.named_buffers()):
            self.lif.register_buffer(name, torch.zeros((0), device=buffer.device))

    def has_state(self) -> bool:
        """Sinabs' own test: reads shapes only, so it forces no host/device sync."""
        return bool(self.lif.is_state_initialised())

    def membrane(self) -> torch.Tensor | None:
        if not self.lif.is_state_initialised():
            return None
        # Shaped (batch, 1, ...) because forward() feeds one timestep at a time.
        return self.lif.v_mem.squeeze(1) if self.lif.v_mem.dim() > 1 else self.lif.v_mem

    def describe(self) -> dict[str, Any]:
        """Read off the BUILT sl.LIF -- these are the values training will use."""
        n = neuron_cfg(self.cfg, "sinabs")
        lif = self.lif
        reset_key = require_choice(n, "reset_mechanism", sorted(RESETS))
        spike_key = require_choice(n, "spike_fn", sorted(SPIKE_FNS))
        # From the module, so the decay/gain pair below describes the neuron that will
        # actually run rather than the one the file asked for.
        tau_mem = float(scalar(lif.tau_mem))
        norm_input = bool(lif.norm_input)
        decay = float(torch.exp(torch.tensor(-1.0 / tau_mem)))
        gain = (1.0 - decay) if norm_input else 1.0
        return {
            "framework": "sinabs",
            "tau_mem": reconcile(lif.tau_mem, require_float(n, "tau_mem")),
            "tau_syn": reconcile(lif.tau_syn, optional_float(n, "tau_syn")),
            "spike_threshold": reconcile(lif.spike_threshold,
                                         require_float(n, "spike_threshold")),
            # sinabs holds CLASSES and OBJECTS where the config holds keys, so these
            # three compare identity against this adapter's own tables.
            "spike_fn": reconcile(lif.spike_fn.__name__, spike_key,
                                  agrees=lif.spike_fn is SPIKE_FNS[spike_key]),
            "reset_mechanism": reconcile(type(lif.reset_fn).__name__, reset_key,
                                         agrees=type(lif.reset_fn) is RESETS[reset_key]),
            # Only MembraneReset carries a reset LEVEL. Under 'subtract' sinabs builds
            # MembraneSubtract(subtract_value=None), which has no such field, so the
            # neuron genuinely does not contain this number -- and this report claims
            # to show the neuron that was BUILT. Reporting v_reset unconditionally read
            # as "the built neuron resets to 0.0", which under subtract it does not.
            "v_reset": (reconcile(getattr(lif.reset_fn, "reset_value", None),
                                  require_float(n, "v_reset")) if reset_key == "zero"
                        else "n/a -- subtract reset has no reset level"),
            "min_v_mem": reconcile(lif.min_v_mem, optional_float(n, "min_v_mem")),
            "norm_input": reconcile(lif.norm_input, require_bool(n, "norm_input")),
            "train_alphas": reconcile(lif.train_alphas, require_bool(n, "train_alphas")),
            # Always False here, by the freeze in __init__. Recorded rather than
            # assumed: sinabs makes the time constant trainable by default, so "we
            # switched it off" is a fact about the run, not about the config.
            "tau_mem_trainable": any(p.requires_grad for p in self.lif.parameters()),
            # Named with its real parameters rather than a fictional `alpha`, since
            # sinabs uses none of the other frameworks' surrogate naming.
            "surrogate": reconcile(_live_surrogate_description(lif.surrogate_grad_fn),
                                   _surrogate_description(n)),
            # Derived, for reading only -- on the same decay/gain scale as the other
            # three adapters so all four can be compared at a glance.
            "effective_decay_gain": f"{decay:.4f}/{gain:.4f}",
        }
