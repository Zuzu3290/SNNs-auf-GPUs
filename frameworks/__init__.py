"""The model side of the pipeline: one shared network in which only the LIF neuron changes per framework."""
from .snn_model import SNNModel

__all__ = ["SNNModel"]
