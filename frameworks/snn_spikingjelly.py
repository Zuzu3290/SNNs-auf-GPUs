"""SpikingJelly binding for the one shared model.

The network is no longer written out per framework: frameworks/spiking_net.py holds the
single SpikingNet and frameworks/snn_model.py holds the single ModelInterface, with only
the NEURON injected per framework. This file exists so that
learning/main.py's FRAMEWORK_MODULES keeps resolving "sj" to a class, unchanged.

The neuron itself comes from network_architecture.yaml -- the `neuron:` block for its
parameters, and `neuron_types.<framework>` for which neuron fills each layer slot.
"""
from __future__ import annotations

from frameworks.snn_model import SNNModel
from skeleton.snn_config import Settings

FRAMEWORK = "sj"


class SNN_SJ(SNNModel):
    """SpikingJelly. See frameworks/snn_model.py for everything it does."""

    def __init__(self, cfg: Settings):
        super().__init__(cfg, framework=FRAMEWORK)
