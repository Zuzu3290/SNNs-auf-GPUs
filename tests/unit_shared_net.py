"""Unit tests for the one shared network, the FC_IN cross-check, and the shared
optimizer/loss.

    python tests/unit_shared_net.py

No dataset, no download, no GPU -- synthetic tensors of the right shape only.

What these are really guarding is the claim the whole comparison rests on: that the
four frameworks differ ONLY in their neuron. Four hand-written networks could agree
today and drift tomorrow with nothing to catch it; one shared network plus these tests
is what makes the claim checkable instead of hopeful.
"""
from __future__ import annotations

import importlib

import torch

from _harness import FRAMEWORKS, MODEL_CLASSES, Suite, fresh_cfg, spike_input

from frameworks.adapters import lif_factory  # noqa: E402
from frameworks.adapters.base import BaseLIF  # noqa: E402
from frameworks.spiking_net import (  # noqa: E402
    FlattenSizeMismatch, build_network, check_flat_features, measure_flat_features,
)
from learning.utilities import build_loss, build_optimizer  # noqa: E402
from skeleton.seeding import seed_model_init, shared_weight_fingerprint  # noqa: E402
from skeleton.snn_config import Settings  # noqa: E402

T, B = 8, 3
suite = Suite("unit_shared_net")
check = suite.check


# ---------------------------------------------------------------------------
# 1. FC_IN: formula and probe must agree, and a disagreement must be caught
# ---------------------------------------------------------------------------
def test_probe_matches_formula_on_real_sensor_sizes() -> None:
    """Every sensor in the dataset registry, plus the two that break the formula."""
    for label, height, width in [
        ("N-MNIST 34x34", 34, 34),
        ("DVS128 128x128", 128, 128),
        ("N-Caltech101 180x240", 180, 240),
    ]:
        cfg = fresh_cfg()
        cfg.SENSOR_H, cfg.SENSOR_W = height, width
        cfg.FC_IN = cfg.compute_fc_in(height, width)
        make_lif = lif_factory("torch", cfg)
        try:
            net = build_network(make_lif, cfg)
            linear = [m for m in net.layers if isinstance(m, torch.nn.Linear)][0]
            check(f"FC_IN formula == probe: {label}", linear.in_features == cfg.FC_IN,
                  f"formula {cfg.FC_IN}, probe {linear.in_features}")
        except Exception as error:  # noqa: BLE001
            check(f"FC_IN formula == probe: {label}", False,
                  f"{type(error).__name__}: {error}")


def test_impossible_sensor_is_caught_not_silently_wrong() -> None:
    """At 10x10 the formula floor-divides -1 // 2 to -1 and returns a positive 32.
    The probe cannot build those layers at all, so the build must fail rather than
    produce a network sized from an impossible number."""
    cfg = fresh_cfg()
    cfg.SENSOR_H = cfg.SENSOR_W = 10
    cfg.FC_IN = cfg.compute_fc_in(10, 10)
    check("formula returns a plausible-but-wrong value at 10x10", cfg.FC_IN == 32,
          f"got {cfg.FC_IN}")
    try:
        build_network(lif_factory("torch", cfg), cfg)
    except Exception as error:  # noqa: BLE001 - any failure is the correct outcome
        check("build fails at 10x10 rather than using it", True,
              f"{type(error).__name__}")
    else:
        check("build fails at 10x10 rather than using it", False,
              "built a network from an impossible flatten size")


def test_mismatch_raises() -> None:
    cfg = fresh_cfg()
    cfg.FC_IN = 12345  # deliberately wrong
    try:
        check_flat_features(800, cfg)
    except FlattenSizeMismatch as error:
        message = str(error)
        check("FC_IN mismatch raises FlattenSizeMismatch",
              "12345" in message and "800" in message,
              "message names both values" if "12345" in message else "message incomplete")
    else:
        check("FC_IN mismatch raises FlattenSizeMismatch", False, "did not raise")


def test_probe_leaves_no_state_behind() -> None:
    """The probe runs a real forward pass through real LIF layers. If it left their
    membranes charged, the first training batch would start from a polluted state."""
    for framework in FRAMEWORKS:
        cfg = fresh_cfg()
        net = build_network(lif_factory(framework, cfg), cfg)
        clean = not any(layer.has_state() for layer in net.lif_layers())
        check(f"probe leaves no state: {framework}", clean)


# ---------------------------------------------------------------------------
# 2. the four frameworks start from the same place
# ---------------------------------------------------------------------------
def test_shared_weight_fingerprint_identical() -> None:
    fingerprints, counts = {}, {}
    for framework in FRAMEWORKS:
        cfg = fresh_cfg()
        seed_model_init(cfg.SEED)
        net = build_network(lif_factory(framework, cfg), cfg)
        fingerprints[framework] = shared_weight_fingerprint(net)
        counts[framework] = sum(p.numel() for p in net.parameters() if p.requires_grad)

    check("all four share one weight fingerprint", len(set(fingerprints.values())) == 1,
          str(fingerprints))
    check("all four have the same trainable parameter count",
          len(set(counts.values())) == 1, str(counts))
    check("sinabs has no extra trainable tau_mem",
          counts.get("sinabs") == counts.get("torch"),
          f"sinabs {counts.get('sinabs')}, snntorch {counts.get('torch')}")


def test_different_seed_changes_the_fingerprint() -> None:
    """Guards against the fingerprint being constant for a reason other than the seed
    -- an all-zero or unseeded init would also make the test above pass."""
    cfg = fresh_cfg()
    seed_model_init(0)
    first = shared_weight_fingerprint(build_network(lif_factory("torch", cfg), cfg))
    seed_model_init(1)
    second = shared_weight_fingerprint(build_network(lif_factory("torch", cfg), cfg))
    check("a different seed gives a different fingerprint", first != second,
          f"{first} vs {second}")


# ---------------------------------------------------------------------------
# 3. one output shape, one spike-rate scale
# ---------------------------------------------------------------------------
def test_output_shape_is_time_first_for_every_framework() -> None:
    for framework in FRAMEWORKS:
        cfg = fresh_cfg()
        seed_model_init(cfg.SEED)
        net = build_network(lif_factory(framework, cfg), cfg)
        with torch.no_grad():
            out = net(spike_input(cfg))
        check(f"output is [T, B, C]: {framework}",
              tuple(out.shape) == (T, B, cfg.NUM_CLASSES), str(tuple(out.shape)))


def test_spike_rates_identical_across_frameworks() -> None:
    """The equivalence claim, measured. Same seed, same input, same neuron spec -- the
    four frameworks should fire identically, not merely similarly."""
    rates: dict[str, dict[str, float]] = {}
    for framework in FRAMEWORKS:
        cfg = fresh_cfg()
        seed_model_init(cfg.SEED)
        net = build_network(lif_factory(framework, cfg), cfg)
        net.set_spike_counting(True)
        with torch.no_grad():
            net(spike_input(cfg))
        rates[framework] = net.spike_rates()

    for slot in ["lif1", "lif2", "lif_out"]:
        values = [rates[f][slot] for f in FRAMEWORKS]
        spread = max(values) - min(values)
        check(f"spike rate agrees across frameworks: {slot}", spread < 1e-9,
              f"spread {spread:.3e}, values {[f'{v:.9f}' for v in values]}")

    check("the hidden layers actually fire",
          all(rates[f]["lif1"] > 0 for f in FRAMEWORKS),
          str({f: rates[f]["lif1"] for f in FRAMEWORKS}))
    check("the output layer actually fires",
          all(rates[f]["lif_out"] > 0 for f in FRAMEWORKS),
          str({f: rates[f]["lif_out"] for f in FRAMEWORKS}))


def test_forward_is_repeatable() -> None:
    """The network resets its own neuron state at the start of forward(). If it did
    not, a second pass over the same input would give a different answer."""
    for framework in FRAMEWORKS:
        cfg = fresh_cfg()
        seed_model_init(cfg.SEED)
        net = build_network(lif_factory(framework, cfg), cfg)
        data = spike_input(cfg)
        with torch.no_grad():
            first, second = net(data), net(data)
        check(f"forward is repeatable without an explicit reset: {framework}",
              torch.equal(first, second))


# ---------------------------------------------------------------------------
# 4. the model wrapper the rest of the pipeline sees
# ---------------------------------------------------------------------------
def test_model_interface_surface() -> None:
    for framework, (module_name, class_name) in MODEL_CLASSES.items():
        cfg = fresh_cfg()
        model_cls = getattr(importlib.import_module(module_name), class_name)
        seed_model_init(cfg.SEED)
        model = model_cls(cfg)

        check(f"tensor_format is TB: {framework}", model.tensor_format() == "TB",
              model.tensor_format())
        check(f"ActivityMonitor is hooked: {framework}",
              len(model.activity.buffers) == 2, str(len(model.activity.buffers)))
        check(f"synops_layer_map is populated: {framework}",
              len(model.synops_layer_map()) == 2, str(len(model.synops_layer_map())))
        check(f"cfg.FRAMEWORK is written back: {framework}",
              cfg.FRAMEWORK == framework, cfg.FRAMEWORK)
        check(f"describe_neuron reports the framework: {framework}",
              model.describe_neuron().get("framework") is not None)


def test_one_loss_path_gives_one_loss_value() -> None:
    """Before the merge, SpikingJelly returned [B, C] and took nn.CrossEntropyLoss
    while the others returned [T, B, C] and took sum-over-T. Same weights and same
    input therefore gave a different loss. One shape means one number."""
    losses = {}
    for framework, (module_name, class_name) in MODEL_CLASSES.items():
        cfg = fresh_cfg()
        model_cls = getattr(importlib.import_module(module_name), class_name)
        seed_model_init(cfg.SEED)
        model = model_cls(cfg)
        torch.manual_seed(11)
        targets = torch.randint(0, cfg.NUM_CLASSES, (B,))
        with torch.no_grad():
            losses[framework] = model.loss_fn(model(spike_input(cfg)), targets).item()
    spread = max(losses.values()) - min(losses.values())
    check("all four report the same loss at identical weights", spread < 1e-6,
          f"spread {spread:.3e}, {losses}")


def test_gradients_reach_every_weight() -> None:
    for framework, (module_name, class_name) in MODEL_CLASSES.items():
        cfg = fresh_cfg()
        model_cls = getattr(importlib.import_module(module_name), class_name)
        seed_model_init(cfg.SEED)
        model = model_cls(cfg)
        torch.manual_seed(11)
        targets = torch.randint(0, cfg.NUM_CLASSES, (B,))
        loss = model.loss_fn(model(spike_input(cfg)), targets)
        model.zero_grad()
        model.backward_pass(loss, scaler=None, do_step=False)
        missing = [
            name for name, param in model.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        check(f"every trainable weight gets a gradient: {framework}", not missing,
              f"missing: {missing}")


# ---------------------------------------------------------------------------
# 5. the shared optimizer and loss
# ---------------------------------------------------------------------------
def test_optimizer_defaults_and_failure() -> None:
    cfg = fresh_cfg()
    parameters = [torch.nn.Parameter(torch.zeros(2))]

    check("default optimizer is nadam", cfg.OPTIMIZER == "nadam", cfg.OPTIMIZER)
    check("default weight decay is 0.0", cfg.WEIGHT_DECAY == 0.0, str(cfg.WEIGHT_DECAY))
    optimizer = build_optimizer(parameters, cfg)
    check("build_optimizer returns NAdam",
          isinstance(optimizer, torch.optim.NAdam), type(optimizer).__name__)
    check("weight decay reaches the optimizer",
          optimizer.param_groups[0]["weight_decay"] == 0.0)

    for name, expected in [("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW),
                           ("sgd", torch.optim.SGD)]:
        cfg.OPTIMIZER = name
        check(f"optimizer {name} builds",
              isinstance(build_optimizer(parameters, cfg), expected))

    cfg.OPTIMIZER = "rmsprop"  # real torch optimizer, deliberately not on our list
    try:
        build_optimizer(parameters, cfg)
    except ValueError as error:
        check("unknown optimizer raises instead of falling back to Adam",
              "rmsprop" in str(error))
    else:
        check("unknown optimizer raises instead of falling back to Adam", False,
              "silently returned an optimizer")


def test_loss_selection() -> None:
    cfg = fresh_cfg()
    check("default loss is cross_entropy", cfg.LOSS_FN == "cross_entropy", cfg.LOSS_FN)
    check("build_loss returns a callable", callable(build_loss(cfg)))

    cfg.LOSS_FN = "hinge"
    try:
        build_loss(cfg)
    except ValueError as error:
        check("unknown loss raises", "hinge" in str(error))
    else:
        check("unknown loss raises", False, "did not raise")


def main() -> int:
    return suite.run([
        test_probe_matches_formula_on_real_sensor_sizes,
        test_impossible_sensor_is_caught_not_silently_wrong,
        test_mismatch_raises,
        test_probe_leaves_no_state_behind,
        test_shared_weight_fingerprint_identical,
        test_different_seed_changes_the_fingerprint,
        test_output_shape_is_time_first_for_every_framework,
        test_spike_rates_identical_across_frameworks,
        test_forward_is_repeatable,
        test_model_interface_surface,
        test_one_loss_path_gives_one_loss_value,
        test_gradients_reach_every_weight,
        test_optimizer_defaults_and_failure,
        test_loss_selection,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
