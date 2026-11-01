"""Norse LIF behind BaseNeuron; uses LIFBoxCell (membrane only), since LIFCell adds a synaptic current the other three lack."""
from __future__ import annotations
import sys
import types
import torch

# Colab only: an empty tensorboard, registered before norse loads, stops norse pulling in TensorFlow/jax, which crash on NumPy 2.
if "torch.utils.tensorboard" not in sys.modules:
    def refuse_tensorboard(*args, **kwargs):
        raise RuntimeError("tensorboard is stubbed out by frameworks/norse_lif.py; remove the stub to use it.")
    stub = types.ModuleType("torch.utils.tensorboard")
    stub.SummaryWriter = stub.FileWriter = refuse_tensorboard
    stub.__all__ = ["SummaryWriter", "FileWriter"]
    stub.__spec__ = types.SimpleNamespace(name="torch.utils.tensorboard", loader=None)
    sys.modules["torch.utils.tensorboard"] = stub

import norse.torch as norse
from norse.torch.functional.reset import reset_subtract, reset_value
from frameworks.snn_model import BaseNeuron

RESETS = {"value": reset_value, "subtract": reset_subtract}


class Neuron(BaseNeuron):
    """Norse hands state back to the caller; input_scale restores the input gain that LIFBoxCell ties to its decay (gain = dt*tau_mem_inv)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        n = cfg.NEURON["norse"]
        self.neuron = norse.LIFBoxCell(
            p=norse.LIFBoxParameters(
                tau_mem_inv=torch.as_tensor(n["tau_mem_inv"], dtype=torch.float32),
                v_leak=torch.as_tensor(n["v_leak"], dtype=torch.float32),
                v_th=torch.as_tensor(n["v_th"], dtype=torch.float32),
                v_reset=torch.as_tensor(n["v_reset"], dtype=torch.float32),
                method=n["surrogate"]["type"],
                alpha=torch.as_tensor(n["surrogate"]["alpha"], dtype=torch.float32),
                reset_method=RESETS[n["reset_method"]],
            ),
            dt=n["dt"],
        )
        self.input_scale = n["input_scale"]
        self.state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spikes, self.state = self.neuron(x * self.input_scale, self.state)
        self.record(spikes)
        return spikes

    def reset(self) -> None:
        self.state = None
