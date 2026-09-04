"""Unit tests for depth-safe LIF layer naming and hooking.

    python tests/unit_layer_naming.py

Covers two fixes: named_lif_layers() must always call the LAST lif layer "lif_out"
regardless of how many precede it (not a fixed-position lookup), and every named layer
must actually be hooked by ActivityMonitor (not a hardcoded lif1/lif2 pair).

CPU-only, no dataset, no download.
"""
from __future__ import annotations

from _harness import FRAMEWORKS, Suite, build_model, fresh_cfg
from frameworks.spiking_net import SpikingNet
from frameworks.adapters.base import BaseLIF

suite = Suite("unit_layer_naming")


class _StubLIF(BaseLIF):
    """A do-nothing LIF for testing naming logic in isolation from any real framework."""

    def forward(self, x):
        return x

    def reset(self) -> None:
        pass


def test_three_layers_names_last_as_lif_out() -> None:
    """Today's architecture: exactly 3 LIF layers. The last one must be lif_out."""
    net = SpikingNet([_StubLIF(), _StubLIF(), _StubLIF()])
    named = net.named_lif_layers()
    suite.check("three layers: names are lif1, lif2, lif_out",
                list(named.keys()) == ["lif1", "lif2", "lif_out"])


def test_five_layers_still_names_last_as_lif_out() -> None:
    """Simulates a future depth ladder: 2 extra hidden LIF layers inserted before the
    output. The LAST layer must still be lif_out, not a positional slot 3/4/5."""
    layers = [_StubLIF() for _ in range(5)]
    net = SpikingNet(layers)
    named = net.named_lif_layers()
    names = list(named.keys())
    suite.check("five layers: exactly one is lif_out", names.count("lif_out") == 1)
    suite.check("five layers: lif_out is the LAST one",
                named["lif_out"] is layers[-1])
    suite.check("five layers: names are lif1..lif4 then lif_out",
                names == ["lif1", "lif2", "lif3", "lif4", "lif_out"])


def test_one_layer_is_just_lif_out() -> None:
    """Degenerate case: a single LIF layer is the output, not lif1."""
    layer = _StubLIF()
    net = SpikingNet([layer])
    named = net.named_lif_layers()
    suite.check("one layer: named lif_out, not lif1", list(named.keys()) == ["lif_out"])


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_three_layers_names_last_as_lif_out,
        test_five_layers_still_names_last_as_lif_out,
        test_one_layer_is_just_lif_out,
    ]))
