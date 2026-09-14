"""Typed, fail-loud access to any config block.

    NO SILENT DEFAULTS.

`skeleton/neuron_spec.py` has enforced this for the `neuron:` block since the merge
began, and this module is the same rule applied to everything else -- `convolution:`,
`training:`, `framing:`, `temporal_slicing:`, `augmentation:`, `cache:`,
`resource_policy:`.

WHY IT HAD TO SPREAD. The pattern being replaced was `conv.get("conv1_out", 12)`. A
misspelled key never reaches it, so a typo silently produced the literal instead:

    conv1_ou: 64      (typo for conv1_out)   ->  CONV1_OUT = 12
    n_time_bin: 40    (typo for n_time_bins) ->  N_TIME_BINS = 16

No error, no warning, and an experiment that quietly did not happen. There were 83 of
these across the two config classes. A misspelled SECTION already raised
(config_loader.check_no_section_collisions); a misspelled KEY inside a valid section
did not, and keys are what decide T, the filter counts and the epoch length.

WHAT THIS MODULE DOES NOT DO. It never judges whether a value is sensible -- the config
owner decides the science. It only guarantees a value was stated explicitly and has the
type the code will treat it as.
"""
from __future__ import annotations

from typing import Any, Sequence


class ConfigKeyError(Exception):
    """A config key is missing, mistyped, or not one of the allowed choices."""


# ---------------------------------------------------------------------------
# Type checks, shared with neuron_spec so both report a mistyped value the same way.
#
# A quoted YAML scalar arrives as a string: `beta: "0.9"` gives "0.9", not 0.9. float()
# would swallow that, so these check the TYPE rather than attempting a conversion. bool
# is rejected from the numeric checks explicitly, because bool subclasses int in Python
# and `true` would otherwise pass as the number 1.
# ---------------------------------------------------------------------------
def check_float(value: Any, label: str, exc: type[Exception]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise exc(
            f"{label} must be a number, got {value!r} ({type(value).__name__}). "
            "A quoted YAML value is a string -- unquote it."
        )
    return float(value)


def check_int(value: Any, label: str, exc: type[Exception]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise exc(
            f"{label} must be a whole number, got {value!r} "
            f"({type(value).__name__}). A quoted YAML value is a string -- unquote it."
        )
    return value


def check_bool(value: Any, label: str, exc: type[Exception]) -> bool:
    if not isinstance(value, bool):
        raise exc(
            f"{label} must be true or false, got {value!r}. YAML reads bare "
            "yes/no/on/off as booleans but quoted 'true' as a string -- unquote it."
        )
    return value


def check_str(value: Any, label: str, exc: type[Exception]) -> str:
    if not isinstance(value, str):
        raise exc(f"{label} must be text, got {value!r} ({type(value).__name__}).")
    return value


def check_choice(value: Any, label: str, allowed: Sequence[str],
                 exc: type[Exception]) -> str:
    if value not in allowed:
        raise exc(f"{label} must be one of {sorted(allowed)}, got {value!r}")
    return str(value)


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------
class Section:
    """One config block, read strictly.

    `label` is the block's dotted path in the YAML, so an error names the key the way
    it is written in the file:

        training.optimizer.lr is not set. State it explicitly -- this pipeline does not
        fall back to a built-in default. Keys present: ['momentum', 'type']

    Nothing here caches: each call reads the underlying dict, so a Section stays valid
    if the dict is mutated (which the dataset registry does to `convolution`).
    """

    def __init__(self, data: Any, label: str, exc: type[Exception] = ConfigKeyError):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise exc(f"{label} must be a mapping, got {type(data).__name__}")
        self._data = data
        self._label = label
        self._exc = exc

    # ---- lookup ------------------------------------------------------------
    def _fetch(self, key: str) -> Any:
        if key not in self._data:
            raise self._exc(
                f"{self._label}.{key} is not set. State it explicitly -- this pipeline "
                f"does not fall back to a built-in default. "
                f"Keys present: {sorted(self._data)}"
            )
        return self._data[key]

    def _tag(self, key: str) -> str:
        return f"{self._label}.{key}"

    def has(self, key: str) -> bool:
        return key in self._data

    def sub(self, key: str) -> "Section":
        """A nested block. Must be present, and must be a mapping."""
        return Section(self._fetch(key), self._tag(key), self._exc)

    # ---- required ----------------------------------------------------------
    def require_float(self, key: str) -> float:
        return check_float(self._fetch(key), self._tag(key), self._exc)

    def require_int(self, key: str) -> int:
        return check_int(self._fetch(key), self._tag(key), self._exc)

    def require_bool(self, key: str) -> bool:
        return check_bool(self._fetch(key), self._tag(key), self._exc)

    def require_str(self, key: str) -> str:
        return check_str(self._fetch(key), self._tag(key), self._exc)

    def require_choice(self, key: str, allowed: Sequence[str]) -> str:
        return check_choice(self._fetch(key), self._tag(key), allowed, self._exc)

    # ---- nullable ----------------------------------------------------------
    #
    # The key must still be PRESENT. `null` is a stated choice with a documented
    # meaning ("no denoising", "duration unknown", "no cap"); an absent key is a
    # mistake. That distinction is the whole point of this module, so it survives here.
    def optional_float(self, key: str) -> float | None:
        value = self._fetch(key)
        return None if value is None else check_float(value, self._tag(key), self._exc)

    def optional_int(self, key: str) -> int | None:
        value = self._fetch(key)
        return None if value is None else check_int(value, self._tag(key), self._exc)

    def optional_str(self, key: str) -> str | None:
        value = self._fetch(key)
        return None if value is None else check_str(value, self._tag(key), self._exc)


def section(config: dict, path: str, exc: type[Exception] = ConfigKeyError) -> Section:
    """A top-level (or dotted) block of the merged config, read strictly.

    The block itself must exist. A run started with an overlay that dropped a whole
    section should fail here rather than silently taking every value from the code.
    """
    node: Any = config
    walked: list[str] = []
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            where = ".".join(walked) or "the merged config"
            present = sorted(node) if isinstance(node, dict) else "(not a mapping)"
            raise exc(
                f"config section '{path}' is missing -- '{part}' not found in {where}. "
                f"Present: {present}"
            )
        node = node[part]
        walked.append(part)
    return Section(node, path, exc)


__all__ = [
    "ConfigKeyError", "Section", "section",
    "check_bool", "check_choice", "check_float", "check_int", "check_str",
]
