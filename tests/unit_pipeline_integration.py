"""Unit tests for the seam between the new shared model and THIS pipeline's own code.

    python tests/unit_pipeline_integration.py

The other suites test the pieces we added. This one tests the parts of the host
pipeline that the swap had to keep working: `ModelInterface`, `ActivityMonitor`,
`measure_dense_macs`, `compute_cv_isi`, `aggregate_spike_output`,
`sum_over_time_cross_entropy` and `SpikingNet`'s introspection helpers.

Three of these used to behave differently depending on which framework was running,
and every such difference is checked here as an equality across all four:

  * ActivityMonitor had no hooks on sinabs, so sinabs silently got no SynOps, no
    CV-ISI, and no activity-regularisation penalty while the other three got all three
  * measure_dense_macs returned {} for sinabs for the same reason
  * aggregate_spike_output received [B, C] from SpikingJelly and [T, B, C] from
    everyone else
"""
from __future__ import annotations

import torch

from _harness import FRAMEWORKS, Suite, build_model, fresh_cfg, spike_input
from frameworks.model_interface import ModelInterface
from learning.training import aggregate_spike_output
from learning.utilities import (
    ActivityMonitor, DenseTimestepBuffer, build_optimizer, compute_cv_isi,
    cv_isi_single_neuron, measure_dense_macs, sum_over_time_cross_entropy,
)

suite = Suite("unit_pipeline_integration")

T, B = 6, 2


# ---------------------------------------------------------------------------
# 1. the ModelInterface contract
# ---------------------------------------------------------------------------
def test_every_abstract_method_is_implemented() -> None:
    for framework in FRAMEWORKS:
        model, _ = build_model(framework)
        suite.check(f"is a ModelInterface: {framework}", isinstance(model, ModelInterface))
        for name in ModelInterface.__abstractmethods__:
            suite.check(f"implements {name}: {framework}",
                        callable(getattr(model, name, None)))


def test_interface_methods_behave() -> None:
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)

        model.train_mode()
        suite.check(f"train_mode sets training: {framework}", model.training)
        model.eval_mode()
        suite.check(f"eval_mode clears training: {framework}", not model.training)

        suite.check(f"get_lr reports the configured lr: {framework}",
                    abs(model.get_lr() - cfg.LEARNING_RATE) < 1e-12, str(model.get_lr()))
        suite.check(f"credit_assignment is BPTT+SG: {framework}",
                    model.credit_assignment() == "BPTT+SG", model.credit_assignment())

        state = model.get_state()
        for key in ["model_state_dict", "optimizer_state_dict", "framework", "neuron"]:
            suite.check(f"get_state has {key}: {framework}", key in state)
        suite.check(f"get_state names the framework: {framework}",
                    state["framework"] == framework, str(state["framework"]))
        suite.check(f"get_state records the neuron: {framework}",
                    isinstance(state["neuron"], dict) and bool(state["neuron"]))


def test_state_dict_round_trips() -> None:
    """Checkpointing must actually restore: same input, same output after a reload."""
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        data = spike_input(cfg, T, B)
        with torch.no_grad():
            before = model(data)
        saved = model.get_state()["model_state_dict"]

        reloaded, _ = build_model(framework)
        reloaded.load_state_dict(saved)
        with torch.no_grad():
            after = reloaded(data)
        suite.check(f"state_dict round-trips: {framework}", torch.equal(before, after))


def test_reset_state_is_exposed() -> None:
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        with torch.no_grad():
            model(spike_input(cfg, T, B))
        model.reset_state()
        clean = not any(layer.has_state() for layer in model.net.lif_layers())
        suite.check(f"reset_state clears every layer: {framework}", clean)


def test_backward_pass_honours_do_step() -> None:
    """The trainer accumulates gradients over several micro-batches with do_step=False
    and steps on the last one. Weights must not move until it does."""
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        targets = torch.randint(0, cfg.NUM_CLASSES, (B,))
        data = spike_input(cfg, T, B)

        reference = model.net.layers[0].weight.detach().clone()
        model.zero_grad()
        model.backward_pass(model.loss_fn(model(data), targets), scaler=None, do_step=False)
        suite.check(f"do_step=False leaves weights alone: {framework}",
                    torch.equal(model.net.layers[0].weight.detach(), reference))
        suite.check(f"do_step=False still fills .grad: {framework}",
                    model.net.layers[0].weight.grad is not None)

        model.zero_grad()
        model.backward_pass(model.loss_fn(model(data), targets), scaler=None, do_step=True)
        suite.check(f"do_step=True moves the weights: {framework}",
                    not torch.equal(model.net.layers[0].weight.detach(), reference))


def test_zero_grad_clears() -> None:
    model, cfg = build_model("torch")
    targets = torch.randint(0, cfg.NUM_CLASSES, (B,))
    model.backward_pass(model.loss_fn(model(spike_input(cfg, T, B)), targets),
                        scaler=None, do_step=False)
    model.zero_grad()
    suite.check("zero_grad sets grads to None",
                all(p.grad is None for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# 2. ActivityMonitor -- was silently a no-op for sinabs
# ---------------------------------------------------------------------------
def test_hooks_record_the_same_shapes_for_every_framework() -> None:
    shapes = {}
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        with torch.no_grad():
            model(spike_input(cfg, T, B))
        recordings = model.activity.recordings()
        shapes[framework] = {k: tuple(v.shape) for k, v in recordings.items() if v is not None}
        suite.check(f"both hidden layers recorded: {framework}",
                    set(recordings) == {"lif1", "lif2"}, str(sorted(recordings)))
        for name, tensor in recordings.items():
            suite.check(f"{name} recording is time-first with T={T}: {framework}",
                        tensor is not None and tensor.shape[0] == T,
                        str(None if tensor is None else tuple(tensor.shape)))
    suite.check("recorded shapes identical across all four frameworks",
                len({str(v) for v in shapes.values()}) == 1, str(shapes))


def test_clear_empties_the_buffers() -> None:
    model, cfg = build_model("norse")
    with torch.no_grad():
        model(spike_input(cfg, T, B))
    model.activity.clear()
    suite.check("clear empties every buffer",
                all(v is None for v in model.activity.recordings().values()))


def test_forward_clears_before_recording() -> None:
    """forward() clears first, so two passes must not stack up 2T timesteps."""
    model, cfg = build_model("sj")
    data = spike_input(cfg, T, B)
    with torch.no_grad():
        model(data)
        model(data)
    recorded = model.activity.recordings()["lif1"]
    suite.check("a second forward does not accumulate timesteps", recorded.shape[0] == T,
                str(tuple(recorded.shape)))


def test_pause_and_resume() -> None:
    """TRADES pauses recording during its adversarial inner loop so those extra
    forward passes do not pollute the activity statistics."""
    model, cfg = build_model("torch")
    data = spike_input(cfg, T, B)
    model.activity.clear()
    model.activity.pause()
    with torch.no_grad():
        model.net(data)
    suite.check("nothing is recorded while paused",
                all(v is None for v in model.activity.recordings().values()))
    model.activity.resume()
    with torch.no_grad():
        model.net(data)
    suite.check("recording resumes",
                all(v is not None for v in model.activity.recordings().values()))


def test_regularization_loss_is_equal_across_frameworks() -> None:
    """It adds a term to the loss. Unequal across frameworks means the loss itself
    differs -- which is exactly what happened when sinabs had no hooks."""
    penalties = {}
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        with torch.no_grad():
            model(spike_input(cfg, T, B))
        penalties[framework] = float(model.activity.regularization_loss(
            min_rate=cfg.ACTIVITY_REG_MIN_RATE, max_rate=cfg.ACTIVITY_REG_MAX_RATE,
            lambda_low=cfg.ACTIVITY_REG_LAMBDA_LOW, lambda_high=cfg.ACTIVITY_REG_LAMBDA_HIGH,
        ))
    spread = max(penalties.values()) - min(penalties.values())
    suite.check("activity penalty identical across frameworks", spread < 1e-9,
                f"spread {spread:.3e}, {penalties}")
    suite.check("the penalty is non-zero (so the check is not vacuous)",
                all(v > 0 for v in penalties.values()), str(penalties))


def test_monitor_with_no_layers_is_a_safe_no_op() -> None:
    monitor = ActivityMonitor()
    suite.check("empty monitor records nothing", monitor.recordings() == {})
    suite.check("empty monitor returns a zero penalty",
                float(monitor.regularization_loss()) == 0.0)


# ---------------------------------------------------------------------------
# 3. DenseTimestepBuffer
# ---------------------------------------------------------------------------
def test_dense_timestep_buffer() -> None:
    buffer = DenseTimestepBuffer()
    suite.check("empty buffer stacks to None", buffer.stack() is None)
    suite.check("empty buffer has no firing rate", buffer.firing_rate_tensor() is None)

    for _ in range(4):
        buffer.push(torch.ones(2, 3))
    stacked = buffer.stack()
    suite.check("stack rebuilds [T, ...]", tuple(stacked.shape) == (4, 2, 3),
                str(tuple(stacked.shape)))
    suite.check("all-ones buffer has firing rate 1.0",
                abs(float(buffer.firing_rate_tensor()) - 1.0) < 1e-9)
    suite.check("firing_rate_tensor stays a tensor (no host sync)",
                isinstance(buffer.firing_rate_tensor(), torch.Tensor))
    buffer.clear()
    suite.check("clear empties the buffer", buffer.stack() is None)


def test_buffer_detaches() -> None:
    """Recordings are diagnostics. Holding the autograd graph would leak memory across
    the whole epoch."""
    buffer = DenseTimestepBuffer()
    buffer.push(torch.ones(2, 2, requires_grad=True) * 2)
    suite.check("pushed tensors are detached", not buffer.stack().requires_grad)


# ---------------------------------------------------------------------------
# 4. measure_dense_macs -- returned {} for sinabs before
# ---------------------------------------------------------------------------
def test_dense_macs_identical_across_frameworks() -> None:
    macs = {}
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        macs[framework] = measure_dense_macs(model, spike_input(cfg, T, B))
        suite.check(f"both layers measured: {framework}",
                    set(macs[framework]) == {"lif1", "lif2"}, str(sorted(macs[framework])))
        suite.check(f"MAC counts are positive: {framework}",
                    all(v > 0 for v in macs[framework].values()), str(macs[framework]))
    suite.check("MAC counts identical across all four frameworks",
                len({str(sorted(v.items())) for v in macs.values()}) == 1, str(macs))


def test_synops_map_points_downstream() -> None:
    """SynOps gates the MACs of the module AFTER a spiking layer, so lif1 must map to
    conv2 and lif2 to the Linear -- derived from the layer list, not hand-written."""
    for framework in FRAMEWORKS:
        model, _ = build_model(framework)
        mapping = model.synops_layer_map()
        suite.check(f"lif1 maps to a Conv2d: {framework}",
                    isinstance(mapping.get("lif1"), torch.nn.Conv2d),
                    type(mapping.get("lif1")).__name__)
        suite.check(f"lif2 maps to a Linear: {framework}",
                    isinstance(mapping.get("lif2"), torch.nn.Linear),
                    type(mapping.get("lif2")).__name__)


# ---------------------------------------------------------------------------
# 5. aggregate_spike_output and the loss helper
# ---------------------------------------------------------------------------
def test_aggregate_spike_output() -> None:
    stack = torch.ones(T, B, 10)
    reduced = aggregate_spike_output(stack)
    suite.check("[T, B, C] sums over time to [B, C]", tuple(reduced.shape) == (B, 10),
                str(tuple(reduced.shape)))
    suite.check("the sum is over T", bool(torch.all(reduced == float(T))))

    already = torch.ones(B, 10)
    suite.check("[B, C] passes through unchanged",
                torch.equal(aggregate_spike_output(already), already))

    suite.expect_raises("a 4-D input raises", ValueError,
                        lambda: aggregate_spike_output(torch.ones(2, 2, 2, 2)))


def test_every_framework_feeds_aggregate_the_same_shape() -> None:
    """The regression this replaced: SpikingJelly used to hand it [B, C] while the
    others handed it [T, B, C]."""
    shapes = set()
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        with torch.no_grad():
            shapes.add(tuple(model(spike_input(cfg, T, B)).shape))
    suite.check("all four produce one identical output shape", len(shapes) == 1, str(shapes))


def test_sum_over_time_cross_entropy() -> None:
    """Must equal plain cross-entropy applied to the time-summed counts."""
    torch.manual_seed(5)
    stack = torch.rand(T, B, 10)
    targets = torch.randint(0, 10, (B,))
    expected = torch.nn.functional.cross_entropy(stack.sum(0), targets)
    actual = sum_over_time_cross_entropy(stack, targets)
    suite.check("sum-over-T CE matches CE on the summed counts",
                torch.allclose(actual, expected), f"{actual.item()} vs {expected.item()}")


# ---------------------------------------------------------------------------
# 6. CV-ISI
# ---------------------------------------------------------------------------
def test_cv_isi_single_neuron() -> None:
    import numpy as np

    suite.check("fewer than two spikes gives None (an ISI is undefined)",
                cv_isi_single_neuron(np.array([3])) is None)
    suite.check("no spikes gives None", cv_isi_single_neuron(np.array([])) is None)
    regular = cv_isi_single_neuron(np.array([0, 5, 10, 15]))
    suite.check("perfectly regular firing has CV 0.0", regular == 0.0, str(regular))
    irregular = cv_isi_single_neuron(np.array([0, 1, 9, 10]))
    suite.check("irregular firing has CV > 0", irregular > 0, str(irregular))


def test_cv_isi_across_frameworks() -> None:
    values = {}
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        with torch.no_grad():
            model(spike_input(cfg, T, B))
        result = compute_cv_isi(model.activity.recordings())
        values[framework] = result.get("network_wide")
        suite.check(f"CV-ISI reports both layers plus the network: {framework}",
                    set(result) == {"lif1", "lif2", "network_wide"}, str(sorted(result)))
    spread = max(values.values()) - min(values.values())
    suite.check("CV-ISI identical across frameworks", spread < 1e-9,
                f"spread {spread:.3e}, {values}")


def test_cv_isi_skips_silent_layers() -> None:
    """A layer with no multi-spike neuron is omitted rather than reported as 0.0,
    which would read as 'perfectly regular' instead of 'no data'."""
    suite.check("an all-silent layer is omitted",
                compute_cv_isi({"lif1": torch.zeros(T, B, 5)}) == {})
    suite.check("a None recording is skipped", compute_cv_isi({"lif1": None}) == {})


# ---------------------------------------------------------------------------
# 7. SpikingNet introspection
# ---------------------------------------------------------------------------
def test_network_introspection() -> None:
    for framework in FRAMEWORKS:
        model, _ = build_model(framework)
        net = model.net
        suite.check(f"three LIF layers: {framework}", len(net.lif_layers()) == 3,
                    str(len(net.lif_layers())))
        suite.check(f"named lif1/lif2/lif_out: {framework}",
                    list(net.named_lif_layers()) == ["lif1", "lif2", "lif_out"],
                    str(list(net.named_lif_layers())))
        suite.check(f"dense_after('lif_out') is None: {framework}",
                    net.dense_after("lif_out") is None)
        suite.check(f"dense_after on an unknown slot is None: {framework}",
                    net.dense_after("lif9") is None)


def test_forward_rejects_the_wrong_rank() -> None:
    model, cfg = build_model("torch")
    suite.expect_raises("a 4-D batch is refused", ValueError,
                        lambda: model.net(torch.zeros(B, cfg.IN_CHANNELS,
                                                      cfg.SENSOR_H, cfg.SENSOR_W)))


def test_spike_counting_toggle_on_the_network() -> None:
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        suite.check(f"counting starts off: {framework}",
                    all(not layer.count_spikes for layer in model.net.lif_layers()))
        model.set_spike_counting(True)
        with torch.no_grad():
            model(spike_input(cfg, T, B))
        rates = model.spike_rates()
        suite.check(f"rates reported for all three slots: {framework}",
                    set(rates) == {"lif1", "lif2", "lif_out"}, str(sorted(rates)))
        suite.check(f"every rate is a fraction in [0, 1]: {framework}",
                    all(0.0 <= v <= 1.0 for v in rates.values()), str(rates))
        model.set_spike_counting(False)
        suite.check(f"counting turns back off: {framework}",
                    all(not layer.count_spikes for layer in model.net.lif_layers()))


# ---------------------------------------------------------------------------
# 8. optimizer wiring not covered by unit_shared_net
# ---------------------------------------------------------------------------
def test_sgd_momentum_comes_from_config() -> None:
    """Momentum was hardcoded at 0.9. It is a config value now, so it must travel."""
    cfg = fresh_cfg()
    cfg.OPTIMIZER, cfg.SGD_MOMENTUM = "sgd", 0.42
    optimizer = build_optimizer([torch.nn.Parameter(torch.zeros(2))], cfg)
    suite.check("configured SGD momentum reaches the optimizer",
                optimizer.param_groups[0]["momentum"] == 0.42,
                str(optimizer.param_groups[0]["momentum"]))


def test_optimizer_covers_every_trainable_weight() -> None:
    for framework in FRAMEWORKS:
        model, _ = build_model(framework)
        owned = {id(p) for group in model.optimizer.param_groups for p in group["params"]}
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        suite.check(f"optimizer owns every trainable parameter: {framework}",
                    trainable <= owned, f"{len(trainable - owned)} unowned")


def main() -> int:
    return suite.run([
        test_every_abstract_method_is_implemented,
        test_interface_methods_behave,
        test_state_dict_round_trips,
        test_reset_state_is_exposed,
        test_backward_pass_honours_do_step,
        test_zero_grad_clears,
        test_hooks_record_the_same_shapes_for_every_framework,
        test_clear_empties_the_buffers,
        test_forward_clears_before_recording,
        test_pause_and_resume,
        test_regularization_loss_is_equal_across_frameworks,
        test_monitor_with_no_layers_is_a_safe_no_op,
        test_dense_timestep_buffer,
        test_buffer_detaches,
        test_dense_macs_identical_across_frameworks,
        test_synops_map_points_downstream,
        test_aggregate_spike_output,
        test_every_framework_feeds_aggregate_the_same_shape,
        test_sum_over_time_cross_entropy,
        test_cv_isi_single_neuron,
        test_cv_isi_across_frameworks,
        test_cv_isi_skips_silent_layers,
        test_network_introspection,
        test_forward_rejects_the_wrong_rank,
        test_spike_counting_toggle_on_the_network,
        test_sgd_momentum_comes_from_config,
        test_optimizer_covers_every_trainable_weight,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
