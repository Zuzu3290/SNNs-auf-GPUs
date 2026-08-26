"""
Typed, fail-loud access to the unified `neuron:` block in network_architecture.yaml.

Ported from the SNNs_2 comparison pipeline, including its central rule:

    NO SILENT DEFAULTS.

A missing neuron key raises. It does not fall back to the framework's own default,
because every framework's default differs from every other's -- that is precisely how
the previous per-framework config ended up with a 40x spread in the input needed to
fire, and with norse and sinabs emitting nothing at all. A loud failure costs a minute;
a silent default costs a whole experiment.

These helpers deliberately do NOT validate whether a value is sensible. The config
owner decides the science; this module only guarantees the value was stated explicitly
and has the right type.
"""
from __future__ import annotations

from typing import Any, Sequence


class NeuronSpecError(Exception):
    """A neuron parameter is missing, mistyped, or not one of the allowed choices."""


def neuron_cfg(cfg, framework_key: str) -> dict:
    """The `neuron:` sub-block for one framework, by its FW_TO_CFG_KEY value
    ('snntorch' | 'norse' | 'spikingjelly' | 'sinabs')."""
    block = getattr(cfg, "NEURON", None)
    if not block:
        raise NeuronSpecError(
            "network_architecture.yaml has no 'neuron:' section. The unified neuron "
            "spec is required -- see that file's header for the target neuron."
        )
    if framework_key not in block:
        raise NeuronSpecError(
            f"neuron.{framework_key} missing from network_architecture.yaml. "
            f"Present: {sorted(block)}"
        )
    return block[framework_key]


def _get(d: dict, path: str) -> Any:
    key = path.split(".")[-1]
    if key not in d:
        raise NeuronSpecError(
            f"neuron.{path} is not set. State it explicitly -- this project does not "
            f"fall back to a framework default. Keys present: {sorted(d)}"
        )
    return d[key]


# A quoted YAML scalar arrives as a string: `beta: "0.9"` gives "0.9", not 0.9. float()
# would accept that silently, so these check the TYPE rather than attempting a
# conversion -- the same strictness require_bool() below already applies, and the same
# rule as src/config.py in the SNNs_2 pipeline these were ported from. A bool is
# rejected explicitly because bool is a subclass of int in Python, so `True` would
# otherwise pass as the number 1.


def require_float(d: dict, path: str) -> float:
    value = _get(d, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NeuronSpecError(
            f"neuron.{path} must be a number, got {value!r} ({type(value).__name__}). "
            "A quoted YAML value is a string -- unquote it."
        )
    return float(value)


def require_int(d: dict, path: str) -> int:
    value = _get(d, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NeuronSpecError(
            f"neuron.{path} must be a whole number, got {value!r} "
            f"({type(value).__name__}). A quoted YAML value is a string -- unquote it."
        )
    return value


def require_bool(d: dict, path: str) -> bool:
    value = _get(d, path)
    if not isinstance(value, bool):
        raise NeuronSpecError(
            f"neuron.{path} must be true or false, got {value!r}. YAML reads bare "
            "yes/no/on/off as booleans but quoted 'true' as a string -- unquote it."
        )
    return value


def require_choice(d: dict, path: str, allowed: Sequence[str]) -> str:
    value = _get(d, path)
    if value not in allowed:
        raise NeuronSpecError(
            f"neuron.{path} must be one of {sorted(allowed)}, got {value!r}"
        )
    return str(value)


def optional_float(d: dict, path: str) -> float | None:
    """For keys whose documented 'off' value is null (sinabs tau_syn, min_v_mem).
    The key must still be PRESENT -- only its value may be null."""
    value = _get(d, path)
    return None if value is None else float(value)


def require_surrogate(d: dict, path: str = "surrogate") -> tuple[str, float]:
    """(type, alpha) for the frameworks that expose a surrogate gradient.

    Named `alpha` throughout this project for consistency, but be aware the three
    frameworks do not agree on what the sharpness parameter is called or how it
    scales -- so the same number does NOT mean the same curve across frameworks.
    That is a documented limitation, not something this function papers over.
    """
    block = _get(d, path)
    if not isinstance(block, dict):
        raise NeuronSpecError(f"neuron.{path} must be a mapping with 'type' and 'alpha'")
    if "type" not in block:
        raise NeuronSpecError(f"neuron.{path}.type is not set")
    if "alpha" not in block:
        raise NeuronSpecError(f"neuron.{path}.alpha is not set")
    return str(block["type"]), float(block["alpha"])


def describe(cfg) -> dict:
    """Flat, loggable view of the whole neuron block, for the run record. Lets a
    results file answer 'what neuron was this?' without reopening the YAML."""
    block = getattr(cfg, "NEURON", None) or {}
    flat: dict[str, Any] = {}
    for fw, params in block.items():
        for key, value in params.items():
            if isinstance(value, dict):
                for sub, subvalue in value.items():
                    flat[f"{fw}.{key}.{sub}"] = subvalue
            else:
                flat[f"{fw}.{key}"] = value
    return flat
