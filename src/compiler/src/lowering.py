"""
Lowering pass: converts an nn.Module into a ComputeGraph.

Iterates over the model's sequential network, creating IR nodes for
each layer. LIF-type neurons expand into four nodes:
    MembraneUpdate → Threshold → SpikeGen → Reset

The original nn.Module reference is stored in IRNode.attrs["module"] so
the runtime can call it directly for correct, weighted execution.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from skeleton import Settings
from compiler.src.ir import ComputeGraph, DeviceTag, IRNode, OpType

logger = logging.getLogger(__name__)

try:
    from spikingjelly.activation_based.neuron import BaseNode as SJBaseNode
except Exception:
    SJBaseNode = None

try:
    import snntorch as _snt
    STBaseNode = getattr(_snt, "SpikingNeuron", None)
except Exception:
    STBaseNode = None

LIF_NAME_FRAGMENTS = (
    "lif", "leaky", "izhikevich", "spiking", "integrate",
    "alpha", "synaptic", "adex", "rlif",   # SNNTorch neuron types
)


def is_spiking_neuron(module: nn.Module) -> bool:
    if SJBaseNode is not None and isinstance(module, SJBaseNode):
        return True
    if STBaseNode is not None and isinstance(module, STBaseNode):
        return True
    name = type(module).__name__.lower()
    return any(frag in name for frag in LIF_NAME_FRAGMENTS)


def device_tag(cfg: Settings) -> DeviceTag:
    return DeviceTag.CUDA if cfg.COMPILER_BACKEND == "cuda" else DeviceTag.CPU


def lower_to_ir(model: nn.Module, cfg: Settings) -> ComputeGraph:
    """
    Structurally lower model to IR.

    Resolves model.net (SNN_SJ style) or falls back to model itself.
    Each nn.Sequential child becomes one or more IR nodes.
    """
    graph   = ComputeGraph()
    dev     = device_tag(cfg)
    prev    = "input"
    counter = 0

    graph.add(IRNode(op=OpType.INPUT, name="input", device=dev))

    net    = getattr(model, "net", model)
    layers = list(net) if isinstance(net, nn.Sequential) else [net]

    for module in layers:
        type_name = type(module).__name__
        base_name = f"{type_name.lower()}_{counter}"

        if isinstance(module, nn.Conv2d):
            graph.add(IRNode(
                op=OpType.CONV2D, name=base_name, inputs=[prev], device=dev,
                attrs={
                    "module":       module,
                    "in_channels":  module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size":  module.kernel_size,
                },
            ))
            prev = base_name
            counter += 1

        elif isinstance(module, nn.Linear):
            graph.add(IRNode(
                op=OpType.DENSE, name=base_name, inputs=[prev], device=dev,
                attrs={
                    "module":       module,
                    "in_features":  module.in_features,
                    "out_features": module.out_features,
                },
            ))
            prev = base_name
            counter += 1

        elif isinstance(module, nn.MaxPool2d):
            graph.add(IRNode(
                op=OpType.POOL2D, name=base_name, inputs=[prev], device=dev,
                attrs={"module": module},
            ))
            prev = base_name
            counter += 1

        elif isinstance(module, nn.Flatten):
            graph.add(IRNode(
                op=OpType.FLATTEN, name=base_name, inputs=[prev], device=dev,
                attrs={"module": module},
            ))
            prev = base_name
            counter += 1

        elif is_spiking_neuron(module):
            tau       = float(getattr(module, "tau",         2.0))
            beta      = (tau - 1.0) / tau if tau > 1.0 else 0.5
            threshold = float(getattr(module, "v_threshold", 1.0))

            mem_name = f"membrane_update_{counter}"
            thr_name = f"threshold_{counter}"
            spk_name = f"spike_gen_{counter}"
            rst_name = f"reset_{counter}"

            graph.add(IRNode(
                op=OpType.MEMBRANE_UPDATE, name=mem_name, inputs=[prev], device=dev,
                attrs={"module": module, "beta": beta, "tau": tau},
            ))
            graph.add(IRNode(
                op=OpType.THRESHOLD, name=thr_name, inputs=[mem_name], device=dev,
                attrs={"threshold": threshold},
            ))
            graph.add(IRNode(
                op=OpType.SPIKE_GEN, name=spk_name, inputs=[thr_name], device=dev,
            ))
            graph.add(IRNode(
                op=OpType.RESET, name=rst_name, inputs=[spk_name, mem_name], device=dev,
                attrs={"beta": beta},
            ))
            prev = rst_name
            counter += 1

        else:
            logger.debug("[LOWERING] Skipping unrecognised layer: %s", type_name)

    graph.add(IRNode(op=OpType.AGGREGATE, name="aggregate", inputs=[prev], device=dev))
    graph.add(IRNode(op=OpType.OUTPUT,    name="output",    inputs=["aggregate"], device=dev))

    if cfg.COMPILER_LOG_IR:
        logger.info("[LOWERING] IR graph:\n%s", graph)

    logger.info("[LOWERING] Lowered %d modules → %d IR nodes", counter, len(graph.nodes))
    return graph
