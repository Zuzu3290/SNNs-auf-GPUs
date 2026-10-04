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


def test_activity_monitor_hooks_every_lif_layer() -> None:
    """SNNModel.activity must hook ALL named LIF layers, including lif_out -- not a
    hardcoded lif1/lif2 pair."""
    model, cfg = build_model("sj")
    named = model.net.named_lif_layers()
    hooked = set(model.activity.buffers.keys())
    suite.check("lif_out is hooked", "lif_out" in hooked,
                f"hooked={sorted(hooked)}")
    suite.check("every named layer is hooked", hooked == set(named.keys()),
                f"named={sorted(named.keys())} hooked={sorted(hooked)}")


def test_synops_layer_map_includes_every_layer_with_a_downstream_dense() -> None:
    """synops_layer_map() must not hardcode lif1/lif2 either -- lif_out has no
    downstream dense layer (it's the last layer), so it's correctly ABSENT from the
    map, but that must be because dense_after() returns None for it, not because the
    iteration never considered it."""
    model, cfg = build_model("sj")
    mapping = model.synops_layer_map()
    suite.check("lif1 and lif2 are in the synops map",
                {"lif1", "lif2"} <= set(mapping.keys()))
    suite.check("lif_out is correctly absent (no downstream dense layer)",
                "lif_out" not in mapping)


def test_dense_before_pairs_each_lif_with_its_upstream_weight_layer() -> None:
    """dense_before(lif_name) is the Conv2d/Linear that FEEDS that LIF -- the mirror
    image of dense_after(), which finds the downstream one."""
    model, cfg = build_model("sj")
    net = model.net
    conv1_out = net.dense_before("lif1")
    conv2_out = net.dense_before("lif2")
    classifier = net.dense_before("lif_out")
    suite.check("lif1's upstream layer is a Conv2d",
                conv1_out is not None and type(conv1_out).__name__ == "Conv2d")
    suite.check("lif2's upstream layer is a Conv2d",
                conv2_out is not None and type(conv2_out).__name__ == "Conv2d")
    suite.check("lif_out's upstream layer is the classifier Linear",
                classifier is not None and type(classifier).__name__ == "Linear")
    suite.check("lif1 and lif2 have DIFFERENT upstream conv layers",
                conv1_out is not conv2_out)


def test_dense_before_returns_none_with_no_preceding_dense_layer() -> None:
    """A LIF layer with nothing but non-dense modules before it (or nothing at all)
    has no upstream weight layer."""
    import torch.nn as nn
    net = SpikingNet([_StubLIF()])
    suite.check("no dense layer before a lone LIF", net.dense_before("lif_out") is None)


def _depth_cfg(layers: int, size: int = 16):
    cfg = fresh_cfg()
    cfg.FC_HIDDEN_LAYERS = layers
    cfg.FC_HIDDEN_SIZE = size
    return cfg


def test_fc_hidden_zero_is_the_original_architecture() -> None:
    """fc_hidden.layers = 0 (the base config) must build exactly the pre-ex8 network:
    one Linear, three LIF slots."""
    model, cfg = build_model("sj", _depth_cfg(0))
    linears = [m for m in model.net.layers if type(m).__name__ == "Linear"]
    suite.check("layers=0: exactly one Linear (the classifier)", len(linears) == 1)
    suite.check("layers=0: names are lif1, lif2, lif_out",
                list(model.net.named_lif_layers()) == ["lif1", "lif2", "lif_out"])


def test_depth_ladder_builds_hidden_layers() -> None:
    """ex8's rungs: N hidden Linear->LIF blocks, named lif3.., wired into every
    per-layer tool (ActivityMonitor, SynOps map, gradient-norm pairing)."""
    import torch
    from skeleton.results_collect import _architecture

    for n in (1, 2, 4):
        model, cfg = build_model("sj", _depth_cfg(n))
        net = model.net
        linears = [m for m in net.layers if isinstance(m, torch.nn.Linear)]
        names = list(net.named_lif_layers())
        expected = ["lif1", "lif2"] + [f"lif{i}" for i in range(3, 3 + n)] + ["lif_out"]
        suite.check(f"layers={n}: {n + 1} Linear modules", len(linears) == n + 1,
                    f"got {len(linears)}")
        suite.check(f"layers={n}: slot names", names == expected, f"got {names}")
        suite.check(f"layers={n}: first Linear reads the flatten width",
                    linears[0].in_features == cfg.FC_IN)
        suite.check(f"layers={n}: classifier reads the hidden size",
                    linears[-1].in_features == 16 and linears[-1].out_features == 10)
        suite.check(f"layers={n}: every slot hooked by ActivityMonitor",
                    set(model.activity.buffers) == set(names))
        suite.check(f"layers={n}: every hidden slot in the SynOps map",
                    set(expected[:-1]) == set(model.synops_layer_map()))
        suite.check(f"layers={n}: lif_out paired with the classifier for grad norms",
                    net.dense_before("lif_out") is linears[-1])

        fields = _architecture(model, cfg)
        suite.check(f"layers={n}: results row counts the hidden layers",
                    fields["fc_hidden_layers"] == n and fields["fc_hidden_size"] == 16,
                    str({k: fields[k] for k in ("fc_hidden_layers", "fc_hidden_size")}))


def test_depth_forward_runs_on_every_framework() -> None:
    """A 2-hidden-layer network must run forward AND backward on all four frameworks,
    since ex8 runs all four."""
    from _harness import spike_input

    for framework in FRAMEWORKS:
        try:
            model, cfg = build_model(framework, _depth_cfg(2))
            x = spike_input(cfg, time_steps=4, batch=2)
            out = model(x)
            out.sum().backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            ok = tuple(out.shape) == (4, 2, cfg.NUM_CLASSES) and all(
                g is not None for g in grads)
            suite.check(f"{framework}: depth-2 forward+backward", ok,
                        f"out {tuple(out.shape)}")
        except Exception as error:  # noqa: BLE001
            suite.check(f"{framework}: depth-2 forward+backward", False,
                        f"{type(error).__name__}: {error}")


def test_missing_lif_hidden_key_raises_only_when_needed() -> None:
    """No silent default: a config without neuron_types.<fw>.lif_hidden must still
    build at depth 0 (older configs), but raise at depth > 0."""
    from frameworks.adapters import lif_factory
    from frameworks.spiking_net import build_network

    cfg = _depth_cfg(0)
    del cfg.NEURON_TYPES["spikingjelly"]["lif_hidden"]
    build_network(lif_factory("sj", cfg), cfg)
    suite.check("depth 0 builds without lif_hidden", True)

    cfg.FC_HIDDEN_LAYERS = 1
    try:
        build_network(lif_factory("sj", cfg), cfg)
    except ValueError:
        suite.check("depth 1 without lif_hidden raises", True)
    else:
        suite.check("depth 1 without lif_hidden raises", False, "built anyway")


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_fc_hidden_zero_is_the_original_architecture,
        test_depth_ladder_builds_hidden_layers,
        test_depth_forward_runs_on_every_framework,
        test_missing_lif_hidden_key_raises_only_when_needed,
        test_three_layers_names_last_as_lif_out,
        test_five_layers_still_names_last_as_lif_out,
        test_one_layer_is_just_lif_out,
        test_activity_monitor_hooks_every_lif_layer,
        test_synops_layer_map_includes_every_layer_with_a_downstream_dense,
        test_dense_before_pairs_each_lif_with_its_upstream_weight_layer,
        test_dense_before_returns_none_with_no_preceding_dense_layer,
    ]))
