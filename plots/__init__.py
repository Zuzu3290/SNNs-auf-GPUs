"""Shared plotting schema for framework comparison.

Design, sources and conventions: `local_docs/plotting_schema.md`.

The rule that makes this reusable: nothing here knows which experiment it is
drawing. `Results.condition` names the column being compared -- normally
`framework`, but a later experiment comparing library defaults, neuron types or
datasets points it elsewhere and every figure still works.
"""

from __future__ import annotations

from .data import METRICS, Results, load, summary
from .style import FRAMEWORKS, apply_rcparams, identity, save

__all__ = [
    "METRICS",
    "Results",
    "load",
    "summary",
    "FRAMEWORKS",
    "apply_rcparams",
    "identity",
    "save",
]
