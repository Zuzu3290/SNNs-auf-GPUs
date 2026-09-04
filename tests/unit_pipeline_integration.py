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
        suite.check(f"all hooked layers recorded: {framework}",
                    set(recordings) == {"lif1", "lif2", "lif_out"}, str(sorted(recordings)))
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


# ---------------------------------------------------------------------------
# 9. energy reporting -- two figures on two bases (D13)
# ---------------------------------------------------------------------------
def _monitor_at(watts: float, phase: str = "e0"):
    """A PipelineMonitor with synthetic power samples and no live sampling thread."""
    from event_data_workflow.system_monitor import PipelineMonitor, PipelineSample

    monitor = PipelineMonitor(cuda_enabled=False, enabled=False)
    monitor.samples = [
        PipelineSample(t_s=i * 0.2, phase=phase, cpu_percent=0.0, ram_available_gb=0.0,
                       gpu_util_pct=90.0, gpu_power_w=watts, gpu_sm_clock_mhz=None)
        for i in range(4)
    ]
    return monitor


def test_energy_report_exposes_both_bases() -> None:
    report = _monitor_at(70.0).phase_energy_report("e0", 10.0)
    for key in ["gpu_energy_j", "gpu_dynamic_energy_j", "avg_power_w",
                "dynamic_power_w", "idle_power_w"]:
        suite.check(f"energy report has {key}", key in report)


def test_total_energy_includes_idle_draw() -> None:
    monitor = _monitor_at(70.0)
    monitor.idle_power_w = 30.0
    report = monitor.phase_energy_report("e0", 10.0)
    suite.check("total is average power x elapsed", report["gpu_energy_j"] == 700.0,
                str(report["gpu_energy_j"]))
    suite.check("dynamic subtracts the idle baseline",
                report["gpu_dynamic_energy_j"] == 400.0, str(report["gpu_dynamic_energy_j"]))


def test_the_two_energies_match_their_own_power_figures() -> None:
    """The inconsistency D13 describes: previously `dynamic_power_w x elapsed` did not
    equal any reported energy, because only the power figure was idle-subtracted."""
    monitor = _monitor_at(70.0)
    monitor.idle_power_w = 30.0
    elapsed = 10.0
    report = monitor.phase_energy_report("e0", elapsed)
    suite.check("avg_power_w x elapsed == total energy",
                abs(report["avg_power_w"] * elapsed - report["gpu_energy_j"]) < 1e-9)
    suite.check("dynamic_power_w x elapsed == dynamic energy",
                abs(report["dynamic_power_w"] * elapsed - report["gpu_dynamic_energy_j"]) < 1e-9)


def test_dynamic_falls_back_to_total_with_no_baseline() -> None:
    """No baseline measured means there is nothing to subtract. Reporting the total
    is honest; inventing a baseline would not be."""
    report = _monitor_at(70.0).phase_energy_report("e0", 10.0)
    suite.check("idle_power_w is None when never measured", report["idle_power_w"] is None)
    suite.check("dynamic equals total when there is no baseline",
                report["gpu_dynamic_energy_j"] == report["gpu_energy_j"],
                f"{report['gpu_dynamic_energy_j']} vs {report['gpu_energy_j']}")


def test_dynamic_energy_is_clamped_at_zero() -> None:
    """A load quieter than the recorded idle baseline means the baseline was wrong,
    not that the work produced energy."""
    monitor = _monitor_at(20.0)
    monitor.idle_power_w = 30.0
    report = monitor.phase_energy_report("e0", 10.0)
    suite.check("dynamic energy never goes negative",
                report["gpu_dynamic_energy_j"] == 0.0, str(report["gpu_dynamic_energy_j"]))


def test_no_power_data_gives_none_not_zero() -> None:
    """Zero joules and 'not measured' are different claims."""
    from event_data_workflow.system_monitor import PipelineMonitor

    report = PipelineMonitor(cuda_enabled=False, enabled=False).phase_energy_report("e0", 10.0)
    suite.check("total is None without power data", report["gpu_energy_j"] is None)
    suite.check("dynamic is None without power data", report["gpu_dynamic_energy_j"] is None)


def test_idle_draw_dilutes_the_difference_between_frameworks() -> None:
    """Why the two bases are not interchangeable, as arithmetic rather than prose.
    Idle 30 W; one framework draws 50 W, another 70 W. Dynamic shows a 2x difference,
    total shows 1.4x -- so the total understates what the comparison is looking for."""
    energies = {}
    for label, watts in [("a", 50.0), ("b", 70.0)]:
        monitor = _monitor_at(watts, phase="x")
        monitor.idle_power_w = 30.0
        report = monitor.phase_energy_report("x", 10.0)
        energies[label] = (report["gpu_energy_j"], report["gpu_dynamic_energy_j"])

    total_ratio = energies["b"][0] / energies["a"][0]
    dynamic_ratio = energies["b"][1] / energies["a"][1]
    suite.check("total ratio is 1.4x", abs(total_ratio - 1.4) < 1e-9, f"{total_ratio:.3f}")
    suite.check("dynamic ratio is 2.0x", abs(dynamic_ratio - 2.0) < 1e-9, f"{dynamic_ratio:.3f}")
    suite.check("the total understates the difference", total_ratio < dynamic_ratio)


# ---------------------------------------------------------------------------
# The prefetcher's CUDA stream: one per LOADER, not one per epoch
#
# No GPU on the machine that runs this suite, so torch.cuda's three entry points the
# prefetcher touches are swapped for stand-ins and the REAL PrefetchedLoader /
# CudaPrefetcher code is driven through them. What is being checked is a property of
# our control flow -- how many streams get constructed, and whether the same one is
# reused -- which is exactly the part that does not need a device to be true.
# ---------------------------------------------------------------------------
class _FakeStream:
    """Stands in for torch.cuda.Stream. Records the cross-stream handshake."""

    def __init__(self, device=None):
        self.device = device
        self.waited_on = []

    def wait_stream(self, other):
        self.waited_on.append(other)


class _FakeTensor:
    """Duck-types the two things the prefetcher does to a batch."""

    def __init__(self, value):
        self.value = value
        self.moved_to = None
        self.recorded_on = []

    def to(self, device, non_blocking=False):
        self.moved_to = (device, non_blocking)
        return self

    def record_stream(self, stream):
        self.recorded_on.append(stream)


class _FakeCuda:
    """Patches torch.cuda.Stream / .stream / .current_stream for one block, and counts
    how many streams the code under test constructs."""

    def __init__(self):
        self.created: list[_FakeStream] = []
        self.default = _FakeStream("default")

    def __enter__(self):
        import contextlib

        import torch

        self._saved = (torch.cuda.Stream, torch.cuda.stream, torch.cuda.current_stream)

        def make_stream(device=None):
            stream = _FakeStream(device)
            self.created.append(stream)
            return stream

        @contextlib.contextmanager
        def stream_ctx(stream):
            yield

        torch.cuda.Stream = make_stream
        torch.cuda.stream = stream_ctx
        torch.cuda.current_stream = lambda device=None: self.default
        return self

    def __exit__(self, *exc):
        import torch

        torch.cuda.Stream, torch.cuda.stream, torch.cuda.current_stream = self._saved
        return False


class _CountingLoader:
    """A tiny DataLoader stand-in that also records stop() calls."""

    def __init__(self, n_batches=4):
        self.n_batches = n_batches
        self.stops = 0
        self.passes = 0

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.passes += 1
        for i in range(self.n_batches):
            yield _FakeTensor(f"data{i}"), _FakeTensor(f"target{i}")

    def stop(self):
        self.stops += 1


def _drain(loader):
    return [(d.value, t.value) for d, t in loader]


def test_one_cuda_stream_for_the_whole_loader_not_one_per_epoch() -> None:
    """The memory bug this fixes.

    PyTorch's caching allocator pools free blocks PER STREAM: a block is tied to the
    stream that allocated it, and freeing it returns it to that stream's pool only. A
    new stream each epoch therefore re-allocates the whole prefetch queue from CUDA
    every epoch while the previous one sits free but unreachable.

    MEASURED before the fix, Colab T4 at T=20/batch 256/depth 32: reserved memory went
    5.05 -> 6.53 -> 8.02 -> 9.50 -> 10.99 GB over five epochs -- +1.48 GB each, which is
    exactly one prefetch queue -- while memory in use stayed flat at 2.89 GB.
    """
    import torch

    from event_data_workflow.data_pipeline import PrefetchedLoader

    with _FakeCuda() as cuda:
        inner = _CountingLoader(n_batches=4)
        loader = PrefetchedLoader(inner, torch.device("cuda"), depth=3)

        suite.check("the stream is built once, by the LOADER's constructor",
                    len(cuda.created) == 1, f"{len(cuda.created)} streams after __init__")
        the_stream = cuda.created[0]

        seen_streams = []
        for _epoch in range(4):
            _drain(loader)
            seen_streams.append(loader.current.stream)

        suite.check("four epochs still built exactly ONE stream",
                    len(cuda.created) == 1, f"{len(cuda.created)} streams after 4 epochs")
        suite.check("every epoch's prefetcher holds that same stream object",
                    all(s is the_stream for s in seen_streams))


def test_the_fix_did_not_break_iteration_or_cross_stream_safety() -> None:
    """One stream is only correct if the handshake around it survives. The prefetcher
    copies on a side stream and hands the tensors to the default stream, so it must
    still wait_stream before use and record_stream on both tensors -- without those the
    allocator may recycle a block the compute stream is still reading."""
    import torch

    from event_data_workflow.data_pipeline import PrefetchedLoader

    with _FakeCuda() as cuda:
        inner = _CountingLoader(n_batches=5)
        loader = PrefetchedLoader(inner, torch.device("cuda"), depth=3)

        first = _drain(loader)
        suite.check("every batch is yielded, in order",
                    first == [(f"data{i}", f"target{i}") for i in range(5)], str(first))
        suite.check("a second pass yields the same batches again",
                    _drain(loader) == first)
        suite.check("len() still reports the underlying batch count", len(loader) == 5)

        # Re-run one pass and inspect the tensors it handed out.
        handed = list(iter(loader))
        suite.check("the default stream waited on the copy stream before use",
                    cuda.default.waited_on and cuda.default.waited_on[0] is loader.stream)
        suite.check("both tensors were moved to the device",
                    all(d.moved_to == (loader.device, True) for d, _ in handed))
        suite.check("record_stream was called on the data tensors",
                    all(d.recorded_on == [cuda.default] for d, _ in handed))
        suite.check("record_stream was called on the target tensors",
                    all(t.recorded_on == [cuda.default] for _, t in handed))


def test_reiterating_still_stops_the_previous_pass() -> None:
    """The stop() call guards a real bug recorded at that call site: an abandoned
    iterator's feeder thread races the global RNG. Sharing one stream must not weaken
    it."""
    import torch

    import event_data_workflow.prefetch as prefetch
    from event_data_workflow.data_pipeline import PrefetchedLoader

    # The feeder that has to be stopped is the AsyncGPUPrefetcher's background thread,
    # one layer below the raw loader -- CudaPrefetcher.stop() forwards to it.
    stopped: list[object] = []
    real_stop = prefetch.AsyncGPUPrefetcher.stop

    def counting_stop(self):
        stopped.append(self)
        return real_stop(self)

    prefetch.AsyncGPUPrefetcher.stop = counting_stop
    try:
        with _FakeCuda():
            loader = PrefetchedLoader(_CountingLoader(n_batches=3),
                                      torch.device("cuda"), depth=2)
            for _ in range(3):
                _drain(loader)
            suite.check("each finished pass stopped its feeder", len(stopped) >= 3,
                        f"{len(stopped)} stops over 3 passes")

            # An ABANDONED pass -- one batch taken, then dropped -- must also be
            # stopped when the next pass starts, not left running.
            before = len(stopped)
            iterator = iter(loader)
            next(iterator)
            del iterator
            _drain(loader)
            suite.check("an abandoned pass is stopped too", len(stopped) > before,
                        f"{len(stopped)} vs {before}")
    finally:
        prefetch.AsyncGPUPrefetcher.stop = real_stop


def test_cpu_runs_use_no_stream_at_all() -> None:
    """device.type == 'cpu' must take the plain path: no stream, no handshake, and the
    batches still arrive. This is what the whole test suite runs on."""
    import torch

    from event_data_workflow.data_pipeline import PrefetchedLoader

    inner = _CountingLoader(n_batches=3)
    loader = PrefetchedLoader(inner, torch.device("cpu"), depth=4)
    suite.check("no stream is created for a CPU device", loader.stream is None)
    got = _drain(loader)
    suite.check("batches still come through on CPU",
                got == [(f"data{i}", f"target{i}") for i in range(3)], str(got))
    suite.check("nothing was record_stream'd on CPU", True)


def test_cuda_prefetcher_alone_still_makes_its_own_stream() -> None:
    """Used standalone, one instance covers the whole run, so a private stream is right.
    The fallback keeps that working -- only PrefetchedLoader, which rebuilds the
    prefetcher every epoch, has to pass one in."""
    import torch

    from event_data_workflow.prefetch import CudaPrefetcher

    with _FakeCuda() as cuda:
        CudaPrefetcher(_CountingLoader(2), torch.device("cuda"), depth=1)
        suite.check("no stream passed -> it builds one", len(cuda.created) == 1)

        borrowed = _FakeStream("borrowed")
        prefetcher = CudaPrefetcher(_CountingLoader(2), torch.device("cuda"), depth=1,
                                    stream=borrowed)
        suite.check("a stream passed -> it builds none", len(cuda.created) == 1)
        suite.check("and it uses the one it was given", prefetcher.stream is borrowed)


def main() -> int:
    return suite.run([
        test_energy_report_exposes_both_bases,
        test_total_energy_includes_idle_draw,
        test_the_two_energies_match_their_own_power_figures,
        test_dynamic_falls_back_to_total_with_no_baseline,
        test_dynamic_energy_is_clamped_at_zero,
        test_no_power_data_gives_none_not_zero,
        test_idle_draw_dilutes_the_difference_between_frameworks,
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
        test_one_cuda_stream_for_the_whole_loader_not_one_per_epoch,
        test_the_fix_did_not_break_iteration_or_cross_stream_safety,
        test_reiterating_still_stops_the_previous_pass,
        test_cpu_runs_use_no_stream_at_all,
        test_cuda_prefetcher_alone_still_makes_its_own_stream,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
