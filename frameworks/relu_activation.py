"""The non-spiking activation, built to be measured exactly like a spiking one.

WHY IT SUBCLASSES BaseLIF. The control model's whole job is to be identical to the
SNN except for the neuron, and that has to hold for the MEASUREMENT as much as for the
architecture. BaseLIF is where spike counting, per-layer neuron counts, state reset and
the slot-name contract live, so inheriting it means build_network(), SpikingNet,
ActivityMonitor, layers.csv, the SynOps map and CV(ISI) all work on the ReLU model with
no branching anywhere. A separate parallel class would have meant a second copy of every
metric, free to drift from the one it is supposed to be compared against.

WHAT "spike rate" MEANS HERE. ReLU emits a continuous value, so the recorded quantity is
the fraction of units whose output is NON-ZERO -- activation density. That is the honest
common ground with a spiking layer: both answer "what fraction of this layer contributed
anything to the next one", and 1 - that number is the activation sparsity the SNN's
efficiency claim rests on. It is NOT a spike count, and nothing here pretends it is; the
energy model reads it through a different constant (a ReLU unit's contribution is a
multiply-accumulate, a spike's is an accumulate).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from frameworks.adapters.base import BaseLIF


class ReLUActivation(BaseLIF):
    """One layer of ReLU units, stateless, measured like a LIF layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(x)
        # Density, not magnitude: a unit counts once when it fires at all, which is the
        # only quantity a spiking layer can also report.
        self._record((out > 0).to(out.dtype))
        return out

    def reset(self) -> None:
        """No state to clear -- ReLU has no membrane. Defined because BaseLIF requires it, and because it is the difference being measured: the SNN carries state across timesteps and this does not."""

    def describe(self) -> dict:
        return {"neuron": "relu", "stateful": False, "surrogate": None}


def relu_factory(cfg):
    """Mirror of frameworks.adapters.lif_factory: hands build_layers() a slot-named activation."""
    def make_relu(slot_name: str) -> ReLUActivation:
        return ReLUActivation()
    return make_relu
