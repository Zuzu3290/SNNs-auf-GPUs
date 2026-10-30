"""SpikingJelly LIF, behind BaseLIF."""
from __future__ import annotations


import numpy as np
import torch
from spikingjelly.activation_based import neuron, surrogate

from frameworks.adapters.base import BaseLIF, reconcile

SURROGATES = {"atan": surrogate.ATan, "sigmoid": surrogate.Sigmoid}


def restore_np_int_alias() -> None:
    """Make the cupy backend runnable on NumPy 2, without changing the pinned SpikingJelly.

    spikingjelly 0.0.0.0.14 -- the latest PyPI release, and the version every recorded
    run used -- validates kernel argument dtypes in auto_cuda/base.py:249 with
    `value.dtype == np.int`. NumPy removed that alias in 2.0, so merely READING it
    raises AttributeError and every cupy forward dies inside a dtype assertion, before
    any kernel runs. The fix exists only on SpikingJelly's master branch.

    Upgrading to master would mean running the cupy arm on a different SpikingJelly than
    the four runs it is meant to be compared against, which is exactly the variable the
    arm is supposed to isolate. So the alias is restored instead, to the value it always
    had: `np.int` was documented as a plain alias for the builtin `int`, and
    `dtype == int` is still valid in NumPy 2. The assertion therefore behaves as its
    author wrote it, and nothing else about either library moves.

    Called only when backend='cupy' is requested -- a torch-backend run never touches it.
    """
    if not hasattr(np, "int"):
        np.int = int


def build_lif(cfg) -> neuron.LIFNode:
    n = cfg.NEURON["spikingjelly"]
    stype, salpha = n["surrogate"]["type"], n["surrogate"]["alpha"]

    step_mode = n["step_mode"]
    backend = n["backend"]
    if backend == "cupy":
        restore_np_int_alias()
    if backend == "cupy" and step_mode != "m":
        # SpikingJelly ignores backend='cupy' under step_mode='s' rather than refusing
        # it, so this configuration would run the plain torch path while the run record
        # claimed the fused one.
        raise ValueError(
            "neuron.spikingjelly.backend = 'cupy' requires step_mode = 'm'. "
            "SpikingJelly silently falls back to the torch path otherwise, which would "
            "record a fused-kernel run that never happened."
        )

    return neuron.LIFNode(
        tau=n["tau"],
        # SpikingJelly defaults decay_input=True, which divides the input by tau and
        # drops the input gain to 1/tau. False keeps gain at 1.0 like the others.
        decay_input=n["decay_input"],
        v_threshold=n["v_threshold"],
        v_reset=n["v_reset"],
        surrogate_function=SURROGATES[stype](alpha=salpha),
        # MEASURED: flipping this changed d(spikes)/d(weight) from 2.64 to 4.34 on a
        # 6-step test. False matches snnTorch, whose reset path also carries gradient.
        detach_reset=n["detach_reset"],
        step_mode=step_mode,
        backend=backend,
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

    def consumes_sequence(self) -> bool:
        """step_mode='m' is exactly the multi-step contract: the whole [T, batch, ...] stack in one call, which is what lets the cupy backend fuse the timestep loop into a single kernel."""
        return self.lif.step_mode == "m"

    def describe(self) -> dict:
        """Read off the BUILT LIFNode -- these are the values training will use."""
        n = self.cfg.NEURON["spikingjelly"]
        stype, salpha = n["surrogate"]["type"], n["surrogate"]["alpha"]
        node, fn = self.lif, self.lif.surrogate_function
        # SpikingJelly names its surrogate classes ATan / Sigmoid while the config keys
        # are lower case, so the comparison is on the class the table maps `stype` to
        # rather than on the two spellings.
        wanted_class = SURROGATES.get(stype)
        return {
            "framework": "spikingjelly",
            "tau": reconcile(node.tau, n["tau"]),
            "decay_input": reconcile(node.decay_input, n["decay_input"]),
            "v_threshold": reconcile(node.v_threshold, n["v_threshold"]),
            "v_reset": reconcile(node.v_reset, n["v_reset"]),
            "detach_reset": reconcile(node.detach_reset, n["detach_reset"]),
            "step_mode": reconcile(node.step_mode, n["step_mode"]),
            "backend": reconcile(node.backend,
                                 n["backend"]),
            "surrogate": reconcile(
                f"{type(fn).__name__}(alpha={getattr(fn, 'alpha', '?')})",
                f"{stype}(alpha={salpha})",
                agrees=(wanted_class is not None and isinstance(fn, wanted_class)
                        and _agree_alpha(getattr(fn, "alpha", None), salpha)),
            ),
        }


def _agree_alpha(live, wanted) -> bool:
    return live is not None and abs(float(live) - float(wanted)) < 1e-9
