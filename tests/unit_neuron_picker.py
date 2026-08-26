"""Unit tests for the per-layer neuron picker.

    python tests/unit_neuron_picker.py

The picker is the part of the config that decides WHICH neuron fills each LIF slot.
Only `lif` is implemented; everything else must fail loudly rather than fall back.
That last word is the whole point of these tests: before the merge, the code defaulted
to `alpha` for snnTorch and `izhikevich` for SpikingJelly, so a missing or misspelled
key silently produced a two-compartment or Izhikevich neuron inside a run labelled a
LIF comparison. A silent default is exactly the failure these tests exist to prevent.

No dataset, no download, no GPU -- this builds neurons and reads config only.
"""
from __future__ import annotations

from _harness import FRAMEWORKS, SLOTS, Suite, fresh_cfg

from frameworks.adapters import (  # noqa: E402
    IMPLEMENTED, NOT_IMPLEMENTED, NeuronNotImplemented, lif_factory,
    resolve_neuron_type,
)
from frameworks.adapters.base import BaseLIF  # noqa: E402
suite = Suite("unit_neuron_picker")
check = suite.check
expect_raises = suite.expect_raises


# ---------------------------------------------------------------------------
# 1. the shipped config resolves to `lif` everywhere
# ---------------------------------------------------------------------------
def test_shipped_config_is_all_lif() -> None:
    cfg = fresh_cfg()
    for framework in FRAMEWORKS:
        for slot in SLOTS:
            resolved = resolve_neuron_type(cfg, framework, slot)
            check(f"shipped config: {framework}.{slot} resolves to 'lif'",
                  resolved == IMPLEMENTED, f"got {resolved!r}")


# ---------------------------------------------------------------------------
# 2. `lif` actually builds a BaseLIF, in every framework and every slot
# ---------------------------------------------------------------------------
def test_lif_builds_for_every_framework_and_slot() -> None:
    cfg = fresh_cfg()
    for framework in FRAMEWORKS:
        make_lif = lif_factory(framework, cfg)
        for slot in SLOTS:
            try:
                layer = make_lif(slot)
                check(f"lif builds: {framework}.{slot}", isinstance(layer, BaseLIF),
                      f"got {type(layer).__name__}")
            except Exception as error:  # noqa: BLE001
                check(f"lif builds: {framework}.{slot}", False,
                      f"{type(error).__name__}: {error}")


def test_each_slot_is_an_independent_instance() -> None:
    """Three slots must be three objects, or they would share neuron state."""
    cfg = fresh_cfg()
    for framework in FRAMEWORKS:
        make_lif = lif_factory(framework, cfg)
        layers = [make_lif(slot) for slot in SLOTS]
        distinct = len({id(layer) for layer in layers}) == len(SLOTS)
        check(f"independent instances: {framework}", distinct)


# ---------------------------------------------------------------------------
# 3. recognised-but-unimplemented neurons raise, and say why
# ---------------------------------------------------------------------------
def test_unimplemented_neurons_raise() -> None:
    for neuron in sorted(NOT_IMPLEMENTED):
        cfg = fresh_cfg()
        cfg.NEURON_TYPES["snntorch"]["lif1"] = neuron
        expect_raises(
            f"unimplemented neuron raises: {neuron!r}",
            NeuronNotImplemented,
            lambda c=cfg: resolve_neuron_type(c, "torch", "lif1"),
            must_mention=[neuron, "lif1"],
        )


def test_unimplemented_neuron_raises_through_the_factory() -> None:
    """The check must fire when the network asks for a layer, not only when the
    resolver is called directly -- the network only ever goes through the factory."""
    cfg = fresh_cfg()
    cfg.NEURON_TYPES["spikingjelly"]["lif2"] = "izhikevich"
    make_lif = lif_factory("sj", cfg)
    check("factory still builds an unaffected slot",
          isinstance(make_lif("lif1"), BaseLIF))
    expect_raises(
        "factory raises on the affected slot",
        NeuronNotImplemented,
        lambda: make_lif("lif2"),
        must_mention=["izhikevich", "lif2"],
    )


def test_old_names_point_at_the_new_one() -> None:
    """`leaky` and `lif_cell` were the previous per-framework names for this slot.
    They must not silently work, and must say what to write instead."""
    for framework_key, framework, old_name in [
        ("snntorch", "torch", "leaky"),
        ("norse", "norse", "lif_cell"),
    ]:
        cfg = fresh_cfg()
        cfg.NEURON_TYPES[framework_key]["lif1"] = old_name
        expect_raises(
            f"old name {old_name!r} rejected with a migration hint",
            NeuronNotImplemented,
            lambda c=cfg, f=framework: resolve_neuron_type(c, f, "lif1"),
            must_mention=["lif"],
        )


# ---------------------------------------------------------------------------
# 4. nothing falls back to a default -- the regression this whole unit is about
# ---------------------------------------------------------------------------
def test_missing_layer_key_raises() -> None:
    cfg = fresh_cfg()
    del cfg.NEURON_TYPES["snntorch"]["lif1"]
    expect_raises(
        "missing layer key raises instead of defaulting",
        ValueError,
        lambda: resolve_neuron_type(cfg, "torch", "lif1"),
        must_mention=["lif1", "snntorch"],
    )


def test_missing_framework_block_raises() -> None:
    cfg = fresh_cfg()
    del cfg.NEURON_TYPES["sinabs"]
    expect_raises(
        "missing framework block raises",
        ValueError,
        lambda: resolve_neuron_type(cfg, "sinabs", "lif1"),
        must_mention=["sinabs"],
    )


def test_empty_neuron_types_raises() -> None:
    cfg = fresh_cfg()
    cfg.NEURON_TYPES = {}
    expect_raises(
        "empty neuron_types raises",
        ValueError,
        lambda: resolve_neuron_type(cfg, "torch", "lif1"),
        must_mention=["snntorch"],
    )


def test_unrecognised_name_raises() -> None:
    cfg = fresh_cfg()
    cfg.NEURON_TYPES["snntorch"]["lif1"] = "leakyy"  # a plausible typo
    expect_raises(
        "typo'd neuron name raises",
        ValueError,
        lambda: resolve_neuron_type(cfg, "torch", "lif1"),
        must_mention=["leakyy"],
    )


def test_unknown_framework_raises() -> None:
    cfg = fresh_cfg()
    expect_raises(
        "unknown framework raises in resolve_neuron_type",
        ValueError,
        lambda: resolve_neuron_type(cfg, "brian2", "lif1"),
        must_mention=["brian2"],
    )
    expect_raises(
        "unknown framework raises in lif_factory",
        ValueError,
        lambda: lif_factory("brian2", cfg),
        must_mention=["brian2"],
    )


# ---------------------------------------------------------------------------
# 5. the picker is genuinely per-layer, not per-framework
# ---------------------------------------------------------------------------
def test_picker_is_per_layer() -> None:
    """One bad slot must not condemn the others, and one good slot must not excuse a
    bad one. If the check were per-framework, one of these two would fail."""
    cfg = fresh_cfg()
    cfg.NEURON_TYPES["norse"]["lif_out"] = "alpha"
    check("per-layer: lif1 still resolves",
          resolve_neuron_type(cfg, "norse", "lif1") == IMPLEMENTED)
    expect_raises(
        "per-layer: lif_out still rejected",
        NeuronNotImplemented,
        lambda: resolve_neuron_type(cfg, "norse", "lif_out"),
        must_mention=["lif_out"],
    )


def main() -> int:
    return suite.run([
        test_shipped_config_is_all_lif,
        test_lif_builds_for_every_framework_and_slot,
        test_each_slot_is_an_independent_instance,
        test_unimplemented_neurons_raise,
        test_unimplemented_neuron_raises_through_the_factory,
        test_old_names_point_at_the_new_one,
        test_missing_layer_key_raises,
        test_missing_framework_block_raises,
        test_empty_neuron_types_raises,
        test_unrecognised_name_raises,
        test_unknown_framework_raises,
        test_picker_is_per_layer,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
