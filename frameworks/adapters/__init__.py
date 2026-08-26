"""One LIF adapter per framework, all behind BaseLIF, selected per layer by config.

Imports are LAZY, deliberately. Each framework brings its own CUDA extension or JIT
backend, and on Windows the spawn-based DataLoader re-imports the parent module in every
worker process -- workers only ever touch the dataset transform, never a model, so
loading all four frameworks in each of them is pure cost. learning/main.py already
avoids this for the model classes; keeping the same discipline here means importing the
adapter package does not drag in snntorch AND norse AND spikingjelly AND sinabs.

WHICH NEURON GETS BUILT
-----------------------
The per-layer, per-framework picker in network_architecture.yaml decides:

    neuron_types:
      snntorch:
        lif1: lif
        lif2: lif
        lif_out: lif

Only `lif` is implemented. The other names below are recognised -- so a typo is told
apart from a deliberate choice -- but every one of them raises. See NOT_IMPLEMENTED.

There is deliberately NO default. Previously `build_lif_layer` defaulted to `alpha` for
snnTorch and `build_sj_layer` defaulted to `izhikevich` for SpikingJelly, so a missing
or misspelled key silently produced a two-compartment or Izhikevich neuron inside a run
labelled a LIF comparison. A missing key now raises.
"""
from __future__ import annotations

import importlib
from typing import Callable

from frameworks.adapters.base import BaseLIF

# cfg.FRAMEWORK selector -> (module, class). Same keys as FW_TO_CFG_KEY in snn_config.
ADAPTERS = {
    "torch":  ("frameworks.adapters.snntorch_lif",     "SnnTorchLIF"),
    "norse":  ("frameworks.adapters.norse_lif",        "NorseLIF"),
    "sj":     ("frameworks.adapters.spikingjelly_lif", "SpikingJellyLIF"),
    "sinabs": ("frameworks.adapters.sinabs_lif",       "SinabsLIF"),
}

# cfg.FRAMEWORK selector -> the neuron: / neuron_types: key for that framework.
# Duplicated from skeleton.snn_config.FW_TO_CFG_KEY rather than imported, so that
# importing an adapter never pulls in Settings and its YAML reads.
FRAMEWORK_KEYS = {
    "torch": "snntorch",
    "norse": "norse",
    "sj": "spikingjelly",
    "sinabs": "sinabs",
}

# The one neuron every framework here can build, and the one the comparison is about.
IMPLEMENTED = "lif"

# Recognised names that are NOT built yet, and why. Listed rather than lumped into one
# error so the message can say what the neuron actually is -- the reason `izhikevich`
# is absent is different in kind from the reason `alpha` is.
NOT_IMPLEMENTED = {
    "alpha": (
        "snnTorch's snn.Alpha is a TWO-COMPARTMENT neuron: it carries a synaptic "
        "current alongside the membrane and returns (spk, syn, mem). BaseLIF holds one "
        "state, so supporting it means extending the adapter's state handling, "
        "has_state() and membrane(). Not done yet -- no experiment needs it."
    ),
    "leaky": (
        "'leaky' was snnTorch's own name for this neuron in the previous config. It IS "
        "the implemented neuron -- write 'lif' instead, so all four frameworks name the "
        "same neuron the same way."
    ),
    "lif_cell": (
        "'lif_cell' was norse's name for this neuron in the previous config, and it "
        "selected norse.LIFCell -- a SECOND-ORDER neuron carrying synaptic current as a "
        "second state. The adapter builds norse.LIFBoxCell instead, which is "
        "first-order like the other three. Write 'lif'."
    ),
    "izhikevich": (
        "SpikingJelly's IzhikevichNode has two state variables and a different "
        "spike-and-reset mechanism. It is not a leaky integrate-and-fire neuron at all, "
        "so wrapping it in BaseLIF would make membrane() and the equivalence check "
        "report on something they do not describe. Deliberately not implemented."
    ),
    "iaf": (
        "Integrate-and-fire is reachable through the LIF adapter already: set "
        "neuron.sinabs.tau_mem to .inf, which makes alpha = exp(-1/tau) = 1 and removes "
        "the leak. Use 'lif' with that tau rather than a separate neuron class."
    ),
    "alif": (
        "Adaptive LIF (a spike-driven adaptive threshold) is a different neuron and is "
        "not implemented."
    ),
}


class NeuronNotImplemented(Exception):
    """A recognised neuron type that this pipeline does not build yet."""


def resolve_neuron_type(cfg, framework: str, layer_name: str) -> str:
    """The neuron type configured for one layer slot of one framework.

    Raises rather than defaulting, on every failure path: unknown framework, missing
    neuron_types block, missing layer key, recognised-but-unimplemented neuron, and
    unrecognised name.
    """
    if framework not in FRAMEWORK_KEYS:
        raise ValueError(
            f"unknown framework {framework!r}. Options: {sorted(FRAMEWORK_KEYS)}"
        )
    framework_key = FRAMEWORK_KEYS[framework]

    all_types = getattr(cfg, "NEURON_TYPES", None) or {}
    if framework_key not in all_types:
        raise ValueError(
            f"network_architecture.yaml has no neuron_types.{framework_key} block. "
            f"Present: {sorted(all_types) or '(nothing)'}"
        )

    layer_types = all_types[framework_key]
    if layer_name not in layer_types:
        raise ValueError(
            f"neuron_types.{framework_key}.{layer_name} is not set. State it "
            f"explicitly -- this pipeline does not fall back to a framework default, "
            f"because every framework's default is a different neuron. "
            f"Keys present: {sorted(layer_types) or '(nothing)'}"
        )

    neuron_type = layer_types[layer_name]
    if neuron_type == IMPLEMENTED:
        return neuron_type

    if neuron_type in NOT_IMPLEMENTED:
        raise NeuronNotImplemented(
            f"neuron_types.{framework_key}.{layer_name} = {neuron_type!r} is not "
            f"implemented.\n  {NOT_IMPLEMENTED[neuron_type]}\n"
            f"  Implemented: {IMPLEMENTED!r}."
        )

    raise ValueError(
        f"neuron_types.{framework_key}.{layer_name} = {neuron_type!r} is not a "
        f"recognised neuron name. Implemented: {IMPLEMENTED!r}. "
        f"Recognised but not implemented: {sorted(NOT_IMPLEMENTED)}."
    )


def lif_factory(framework: str, cfg) -> Callable[[str], BaseLIF]:
    """Return a callable that builds one fresh LIF layer for a named layer slot.

    A FACTORY rather than a layer, because the network needs three independent
    instances (two hidden, one output) and each must own its own state.

    It takes the layer NAME because the picker is per-layer: the network calls
    `make_lif("lif1")`, `make_lif("lif2")`, `make_lif("lif_out")`, and each call
    re-reads that slot's configured neuron type. Today all three must be 'lif', so
    every layer gets the same neuron -- but the check happens per slot, which is what
    keeps the config's per-layer shape honest instead of decorative.
    """
    if framework not in ADAPTERS:
        raise ValueError(
            f"unknown framework {framework!r}. Options: {sorted(ADAPTERS)}"
        )

    def make_lif(layer_name: str) -> BaseLIF:
        resolve_neuron_type(cfg, framework, layer_name)  # raises unless 'lif'
        module_name, class_name = ADAPTERS[framework]
        adapter_cls = getattr(importlib.import_module(module_name), class_name)
        return adapter_cls(cfg)

    return make_lif


__all__ = [
    "BaseLIF",
    "ADAPTERS",
    "FRAMEWORK_KEYS",
    "IMPLEMENTED",
    "NOT_IMPLEMENTED",
    "NeuronNotImplemented",
    "resolve_neuron_type",
    "lif_factory",
]
