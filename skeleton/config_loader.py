"""Loads the three base config files, and optionally one experiment overlay on top.

TWO WAYS TO RUN, BOTH SUPPORTED
-------------------------------
1. No overlay. The three files in `configuration/` are the whole config, exactly as
   this pipeline has always worked:

       python learning/main.py

2. One overlay file per experiment, stating ONLY what differs. Everything it does not
   mention is inherited from the three base files:

       python learning/main.py --config experiments/ex2/config.yaml --experiment ex2

WHY A FLAT OVERLAY WORKS
------------------------
The three base files share no top-level section name:

    SNN_module.yaml            architecture, training, output
    network_architecture.yaml  convolution, neuron_types, neuron
    data_workflow.yaml         framing, temporal_slicing, augmentation, cache,
                               resource_policy

so they can be merged into one dict without ambiguity, and one flat overlay can reach
any key in any of them. `check_no_section_collisions()` asserts that property still
holds rather than trusting it -- if someone later adds `training:` to
data_workflow.yaml, the loader says so instead of silently letting one file win.

An overlay may itself start with `extends: other.yaml`, resolved relative to its own
directory, so a variant experiment inherits from the experiment it varies rather than
copying it. Duplicated config is a real hazard in a study whose whole claim is
"everything held equal": change a value in one copy, forget the other, and the
comparison is silently invalid.

PRECEDENCE
----------
    CLI argument  >  --config overlay  >  the three base files  >  code default

The code-default step is deliberately thin: the neuron spec has none at all and raises
instead, because every framework's own default is a different neuron.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configuration"

# The three files that make up the base config, and what each one owns.
BASE_FILES = {
    "SNN_module.yaml": "how the network is TRAINED",
    "network_architecture.yaml": "what the network IS",
    "data_workflow.yaml": "how the DATA arrives",
}


class ConfigError(Exception):
    """A config file is missing, malformed, or contradicts another one."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay `override` on `base`. Lists are REPLACED, not concatenated.

    Replacing lists is the safer rule for this project: a config that says
    `inputs: [a]` means exactly one input, not "append a to whatever the base had".
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def check_no_section_collisions(per_file: dict[str, dict[str, Any]]) -> None:
    """The three base files must not both define the same top-level section.

    If they did, merging them would silently let whichever loaded last win, and an
    overlay key would land in an unpredictable place.
    """
    seen: dict[str, str] = {}
    collisions: list[str] = []
    for filename, content in per_file.items():
        for section in content:
            if section in seen:
                collisions.append(f"'{section}' in both {seen[section]} and {filename}")
            else:
                seen[section] = filename
    if collisions:
        raise ConfigError(
            "the base config files define overlapping top-level sections, so merging "
            "them is ambiguous:\n  " + "\n  ".join(collisions)
            + "\nGive each section a single home."
        )


def load_base(config_dir: Path | str = CONFIG_DIR) -> dict[str, Any]:
    """The three shipped files, merged into one dict."""
    config_dir = Path(config_dir)
    per_file = {name: read_yaml(config_dir / name) for name in BASE_FILES}
    check_no_section_collisions(per_file)
    merged: dict[str, Any] = {}
    for content in per_file.values():
        merged = deep_merge(merged, content)
    return merged


def load_overlay(path: Path | str, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """One overlay file, with its `extends:` chain already resolved into it.

    Does NOT include the base files -- see load_config() for that.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"--config file not found: {path}")

    resolved = path.resolve()
    if resolved in _chain:
        loop = " -> ".join(p.name for p in (*_chain, resolved))
        raise ConfigError(f"circular 'extends' between config files: {loop}")

    loaded = read_yaml(path)
    parent = loaded.pop("extends", None)
    if parent is None:
        return loaded
    if not isinstance(parent, str):
        raise ConfigError(f"'extends' in {path} must be a filename, got {parent!r}")

    inherited = load_overlay(path.parent / parent, _chain=(*_chain, resolved))
    return deep_merge(inherited, loaded)


def known_sections(config_dir: Path | str = CONFIG_DIR) -> set[str]:
    return set(load_base(config_dir))


def check_overlay_sections(overlay: dict[str, Any], config_dir: Path | str = CONFIG_DIR) -> None:
    """An overlay section that matches nothing in the base files is almost certainly a
    typo, and a typo'd section is invisible: the run proceeds with the base value and
    the experiment quietly does not happen. Caught here instead."""
    unknown = sorted(set(overlay) - known_sections(config_dir))
    if unknown:
        raise ConfigError(
            f"--config sets unknown top-level section(s): {unknown}. "
            f"Known sections: {sorted(known_sections(config_dir))}. "
            "An overlay may only override sections that exist in the base config."
        )


def load_config(
    overlay_path: Path | str | None = None, config_dir: Path | str = CONFIG_DIR
) -> dict[str, Any]:
    """The full config: the three base files, with an optional overlay merged on top."""
    base = load_base(config_dir)
    if overlay_path is None:
        return base
    overlay = load_overlay(overlay_path)
    check_overlay_sections(overlay, config_dir)
    return deep_merge(base, overlay)
