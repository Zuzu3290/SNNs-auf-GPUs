"""Unit tests for the four LIF adapters: the neuron each one actually builds.

    python tests/unit_adapters.py

The network tests (unit_shared_net.py) show the four frameworks AGREE. These show
WHAT they agree on -- that each adapter really produces the neuron the config asked
for: decay 0.9, input gain 1.0, threshold 1.0, hard reset to 0, one binary spike.

That distinction matters. Four adapters could agree on the wrong neuron. Each check
here measures a property directly from a single layer, with no network and no dataset,
so a failure names the framework and the property rather than just "the frameworks
disagree".

There is also one test per framework-specific TRAP: the default that, left alone,
silently makes that framework incomparable with the other three.
"""
from __future__ import annotations

import torch

from _harness import FRAMEWORKS, Suite, fresh_cfg, single_neuron

suite = Suite("unit_adapters")

# What every framework's neuron must be, per network_architecture.yaml's `neuron:` block.
TARGET_DECAY = 0.9
TARGET_GAIN = 1.0
TARGET_THRESHOLD = 1.0
TOLERANCE = 1e-6


def membrane_value(layer) -> float:
    membrane = layer.membrane()
    return float("nan") if membrane is None else float(membrane.detach().flatten()[0])


# ---------------------------------------------------------------------------
# 1. the target neuron, measured one property at a time
# ---------------------------------------------------------------------------
def test_input_gain_is_one() -> None:
    """A single sub-threshold impulse must land on the membrane undiminished.

    This is the one Norse and sinabs both get wrong by default: each locks input gain
    to its decay, so decay 0.9 would give gain 0.1 and the neuron would need 10x the
    input to fire. Norse compensates with input_scale, sinabs with norm_input=false.
    """
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.reset()
        layer(torch.tensor([[0.3]]))
        gain = membrane_value(layer) / 0.3
        suite.check(f"input gain is 1.0: {framework}", abs(gain - TARGET_GAIN) < TOLERANCE,
                    f"measured {gain:.9f}")


def test_decay_is_zero_point_nine() -> None:
    """Charge once, then feed zero: the membrane must retain exactly 90%."""
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.reset()
        layer(torch.tensor([[0.3]]))
        charged = membrane_value(layer)
        layer(torch.zeros(1, 1))
        decay = membrane_value(layer) / charged
        suite.check(f"decay is 0.9: {framework}", abs(decay - TARGET_DECAY) < TOLERANCE,
                    f"measured {decay:.9f}")


def test_threshold_is_one() -> None:
    """Just over threshold fires, just under stays silent. Brackets the value rather
    than reading it from the config, so a framework that ignores the setting is caught."""
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.reset()
        fires = float(layer(torch.tensor([[TARGET_THRESHOLD + 0.01]])).flatten()[0])
        layer.reset()
        silent = float(layer(torch.tensor([[TARGET_THRESHOLD - 0.01]])).flatten()[0])
        suite.check(f"threshold 1.0 brackets correctly: {framework}",
                    fires == 1.0 and silent == 0.0, f"above={fires}, below={silent}")


def test_reset_is_hard_to_zero() -> None:
    """After a spike the membrane must be 0, not (membrane - threshold).

    snnTorch and sinabs both default to a SOFT reset, which leaves the remainder
    behind; the config pins both to hard. Driven well above threshold so soft and hard
    are far apart and cannot be confused by float noise.
    """
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.reset()
        spike = float(layer(torch.tensor([[1.6]])).flatten()[0])
        after = membrane_value(layer)
        suite.check(f"hard reset to 0 after a spike: {framework}",
                    spike == 1.0 and abs(after) < TOLERANCE,
                    f"spike={spike}, membrane after={after:.9f} (soft reset would be ~0.6)")


def test_reset_is_immediate_not_delayed() -> None:
    """snnTorch's reset_delay=True applies the reset on the FOLLOWING timestep, so its
    membrane reads differently from the other three for one step after every spike.
    Fire, then feed zero: a delayed reset shows up as a non-zero membrane here."""
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.reset()
        layer(torch.tensor([[1.6]]))
        layer(torch.zeros(1, 1))
        after = membrane_value(layer)
        suite.check(f"reset is immediate, not delayed: {framework}", abs(after) < TOLERANCE,
                    f"membrane one step after the spike = {after:.9f}")


def test_spike_is_binary() -> None:
    """sinabs defaults to MultiSpike: one neuron may emit 2, 3 or more spikes in a
    single timestep, which would make its spike-rate FRACTION exceed 1.0 and stop
    being comparable. Driven to 5x threshold, where MultiSpike would return 5."""
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.reset()
        spike = float(layer(torch.tensor([[5.0]])).flatten()[0])
        suite.check(f"one spike maximum per timestep: {framework}", spike == 1.0,
                    f"emitted {spike}")


def test_subthreshold_integration_over_time() -> None:
    """The whole neuron at once: repeated sub-threshold input must integrate toward
    0.15/(1-0.9) = 1.5, cross the threshold, reset, and climb again. All four must
    fire on exactly the same timestep."""
    first_spike_step = {}
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.reset()
        for step in range(40):
            if float(layer(torch.tensor([[0.15]])).flatten()[0]) == 1.0:
                first_spike_step[framework] = step
                break
    suite.check("every framework eventually fires on a 0.15 ramp",
                len(first_spike_step) == len(FRAMEWORKS), str(first_spike_step))
    suite.check("all four fire on the same timestep",
                len(set(first_spike_step.values())) == 1, str(first_spike_step))


# ---------------------------------------------------------------------------
# 2. the BaseLIF state contract
# ---------------------------------------------------------------------------
def test_state_lifecycle() -> None:
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        suite.check(f"no state before the first forward: {framework}", not layer.has_state())
        suite.check(f"membrane is None before the first forward: {framework}",
                    layer.membrane() is None)
        layer(torch.tensor([[0.3]]))
        suite.check(f"state exists after a forward: {framework}", layer.has_state())
        suite.check(f"membrane is readable after a forward: {framework}",
                    layer.membrane() is not None)
        layer.reset()
        suite.check(f"reset clears the state: {framework}", not layer.has_state())
        suite.check(f"membrane is None again after reset: {framework}",
                    layer.membrane() is None)


def test_reset_actually_restarts_the_trajectory() -> None:
    """has_state() going False is not proof the neuron forgot. Charge it, reset, then
    repeat the identical input and require the identical response."""
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.reset()
        first = [float(layer(torch.tensor([[0.4]])).flatten()[0]) for _ in range(12)]
        layer.reset()
        second = [float(layer(torch.tensor([[0.4]])).flatten()[0]) for _ in range(12)]
        suite.check(f"reset restarts the trajectory: {framework}", first == second,
                    f"{first} vs {second}")


def test_shape_is_preserved() -> None:
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.reset()
        data = torch.rand(2, 5, 7, 7)
        suite.check(f"output shape matches input shape: {framework}",
                    tuple(layer(data).shape) == tuple(data.shape))


# ---------------------------------------------------------------------------
# 3. spike counting (BaseLIF._record) -- the spike-rate metric's foundation
# ---------------------------------------------------------------------------
def test_spike_counting_is_off_by_default() -> None:
    """Counting costs a GPU reduction per layer per timestep, and this project measures
    wall-clock time. It must stay off unless a measurement pass asks for it."""
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        suite.check(f"count_spikes defaults to False: {framework}", layer.count_spikes is False)
        layer(torch.ones(1, 4))
        suite.check(f"nothing accumulates while off: {framework}", layer.spike_slots == 0,
                    f"slots={layer.spike_slots}")


def test_spike_rate_is_a_true_fraction() -> None:
    """A neuron driven far above threshold every step fires every step, so its rate
    must be exactly 1.0 -- not T, and not a count."""
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.count_spikes = True
        layer.reset_spike_stats()
        layer.reset()
        for _ in range(10):
            layer(torch.full((3, 4), 5.0))
        suite.check(f"always-firing neuron has rate 1.0: {framework}",
                    abs(layer.spike_rate() - 1.0) < TOLERANCE, f"{layer.spike_rate():.9f}")
        suite.check(f"neurons() reports per-sample width: {framework}", layer.neurons() == 4,
                    str(layer.neurons()))
        suite.check(f"slots counts neurons x batch x steps: {framework}",
                    layer.spike_slots == 3 * 4 * 10, str(layer.spike_slots))


def test_silent_neuron_has_rate_zero() -> None:
    for framework in FRAMEWORKS:
        layer = single_neuron(framework)
        layer.count_spikes = True
        layer.reset_spike_stats()
        layer.reset()
        for _ in range(10):
            layer(torch.zeros(3, 4))
        suite.check(f"silent neuron has rate 0.0: {framework}", layer.spike_rate() == 0.0,
                    f"{layer.spike_rate()}")


def test_reset_spike_stats_clears_everything() -> None:
    layer = single_neuron("torch")
    layer.count_spikes = True
    layer.reset()
    layer(torch.full((2, 3), 5.0))
    layer.reset_spike_stats()
    suite.check("reset_spike_stats zeroes the total", layer.spike_total == 0.0)
    suite.check("reset_spike_stats zeroes the slots", layer.spike_slots == 0)
    suite.check("reset_spike_stats forgets the shape", layer.spike_shape is None)
    suite.check("rate is 0.0 with no slots", layer.spike_rate() == 0.0)


def test_counting_does_not_sync_to_host() -> None:
    """The running total must stay a device tensor. Calling .item() per layer per
    timestep would force a host sync and wreck the latency measurements this pipeline
    exists to take."""
    layer = single_neuron("torch")
    layer.count_spikes = True
    layer.reset()
    layer(torch.full((2, 3), 5.0))
    suite.check("accumulated spike total stays a tensor",
                isinstance(layer.spike_total, torch.Tensor), type(layer.spike_total).__name__)


# ---------------------------------------------------------------------------
# 4. describe() -- what lands in the run record
# ---------------------------------------------------------------------------
def test_describe_names_the_framework_and_its_real_parameters() -> None:
    expected_keys = {
        "torch": {"beta", "threshold", "reset_mechanism", "reset_delay", "surrogate"},
        "norse": {"cell", "dt", "tau_mem_inv", "v_th", "v_reset", "v_leak",
                  "reset_method", "input_scale", "surrogate"},
        "sj": {"tau", "decay_input", "v_threshold", "v_reset", "detach_reset",
               "step_mode", "backend", "surrogate"},
        "sinabs": {"tau_mem", "tau_syn", "spike_threshold", "spike_fn", "reset_mechanism",
                   "v_reset", "min_v_mem", "norm_input", "train_alphas",
                   "tau_mem_trainable", "surrogate"},
    }
    for framework in FRAMEWORKS:
        described = single_neuron(framework).describe()
        suite.check(f"describe names the framework: {framework}",
                    described.get("framework") is not None, str(described.get("framework")))
        missing = expected_keys[framework] - set(described)
        suite.check(f"describe reports every real parameter: {framework}", not missing,
                    f"missing {sorted(missing)}")


def test_sinabs_reports_its_frozen_time_constant() -> None:
    """sinabs makes tau_mem trainable by default and the adapter freezes it. That the
    freeze happened is a fact about the run, so it has to be recorded, not assumed."""
    described = single_neuron("sinabs").describe()
    suite.check("sinabs records tau_mem_trainable as False",
                described.get("tau_mem_trainable") is False,
                str(described.get("tau_mem_trainable")))
    suite.check("sinabs reports its effective decay/gain",
                described.get("effective_decay_gain", "").startswith("0.9"),
                str(described.get("effective_decay_gain")))


def test_sinabs_neuron_parameters_are_frozen() -> None:
    layer = single_neuron("sinabs")
    trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
    suite.check("no trainable parameter survives on the sinabs neuron", not trainable,
                f"trainable: {trainable}")


# ---------------------------------------------------------------------------
# 5. refusals -- settings the adapter must not silently accept
# ---------------------------------------------------------------------------
def test_spikingjelly_refuses_multi_step() -> None:
    """The shared network hands every layer ONE timestep, so 'm' cannot run here.
    Accepting it would also imply the fused cupy backend was in play when it is not."""
    cfg = fresh_cfg()
    cfg.NEURON["spikingjelly"]["step_mode"] = "m"
    suite.expect_raises("SpikingJelly refuses step_mode 'm'", ValueError,
                        lambda: single_neuron("sj", cfg), must_mention=["step_mode"])


def test_unknown_surrogate_is_refused() -> None:
    cfg = fresh_cfg()
    cfg.NEURON["snntorch"]["surrogate"]["type"] = "not_a_surrogate"
    suite.expect_raises("snnTorch refuses an unknown surrogate", ValueError,
                        lambda: single_neuron("torch", cfg),
                        must_mention=["not_a_surrogate"])


def test_sinabs_surrogate_is_actually_read() -> None:
    """The archived adapter never read the surrogate at all, so a config naming one was
    silently ignored. An unknown name must now be refused, which proves it is read."""
    cfg = fresh_cfg()
    cfg.NEURON["sinabs"]["surrogate"]["type"] = "not_a_surrogate"
    suite.expect_raises("sinabs refuses an unknown surrogate", Exception,
                        lambda: single_neuron("sinabs", cfg),
                        must_mention=["not_a_surrogate"])


def test_sinabs_honours_the_selected_surrogate() -> None:
    """ex2 selects periodic_exponential. Check the object actually reaches the layer."""
    cfg = fresh_cfg()
    cfg.NEURON["sinabs"]["surrogate"] = {
        "type": "periodic_exponential", "grad_width": 0.5, "grad_scale": 1.0,
    }
    layer = single_neuron("sinabs", cfg)
    name = type(layer.lif.surrogate_grad_fn).__name__
    suite.check("sinabs builds the configured surrogate", name == "PeriodicExponential", name)
    suite.check("sinabs describe reports it",
                "periodic_exponential" in layer.describe().get("surrogate", ""),
                layer.describe().get("surrogate", ""))


def test_missing_neuron_key_raises() -> None:
    """The no-silent-defaults rule, at the adapter level rather than the picker level."""
    cfg = fresh_cfg()
    del cfg.NEURON["snntorch"]["beta"]
    suite.expect_raises("a missing neuron key raises", Exception,
                        lambda: single_neuron("torch", cfg), must_mention=["beta"])


def test_missing_neuron_block_raises() -> None:
    cfg = fresh_cfg()
    cfg.NEURON = {}
    suite.expect_raises("an empty neuron block raises", Exception,
                        lambda: single_neuron("norse", cfg))


# ---------------------------------------------------------------------------
# 6. the config genuinely drives the neuron
# ---------------------------------------------------------------------------
def test_changing_the_config_changes_the_neuron() -> None:
    """Guards against an adapter that ignores the config and hardcodes the right
    answer -- every measurement above would still pass in that case."""
    overrides = {
        "torch": ("snntorch", "beta", 0.5),
        "norse": ("norse", "tau_mem_inv", 500.0),   # decay 1 - 0.001*500 = 0.5
        "sj": ("spikingjelly", "tau", 2.0),         # decay 1 - 1/2 = 0.5
        "sinabs": ("sinabs", "tau_mem", 1.4426950408889634),  # -1/ln(0.5)
    }
    for framework, (block, key, value) in overrides.items():
        cfg = fresh_cfg()
        cfg.NEURON[block][key] = value
        if framework == "norse":
            cfg.NEURON["norse"]["input_scale"] = 2.0  # keep gain at 1.0: 0.5 * 2.0
        layer = single_neuron(framework, cfg)
        layer.reset()
        layer(torch.tensor([[0.3]]))
        charged = membrane_value(layer)
        layer(torch.zeros(1, 1))
        decay = membrane_value(layer) / charged
        suite.check(f"config change moves decay to 0.5: {framework}",
                    abs(decay - 0.5) < 1e-5, f"measured {decay:.9f}")


def test_sinabs_v_reset_is_only_reported_when_the_reset_uses_it() -> None:
    """This report claims to show the neuron that was BUILT.

    sinabs takes a reset LEVEL only for MembraneReset. Under 'subtract' it builds
    MembraneSubtract(subtract_value=None), which has no such field -- so printing
    `v_reset 0.0` there named a number the built object does not contain, and read as
    "resets to 0.0" when it resets by subtracting the threshold.
    """
    cfg = fresh_cfg()
    cfg.NEURON["sinabs"]["reset_mechanism"] = "subtract"
    described = single_neuron("sinabs", cfg).describe()
    suite.check("under subtract, v_reset is not reported as a number",
                not isinstance(described["v_reset"], (int, float)),
                f"got {described['v_reset']!r}")
    suite.check("and it says why", "subtract" in str(described["v_reset"]))

    cfg = fresh_cfg()
    cfg.NEURON["sinabs"]["reset_mechanism"] = "zero"
    cfg.NEURON["sinabs"]["v_reset"] = 0.0
    described = single_neuron("sinabs", cfg).describe()
    suite.check("under zero, the real reset level IS reported",
                described["v_reset"] == 0.0, f"got {described['v_reset']!r}")

    # The live object is the authority on both branches.
    layer = single_neuron("sinabs", cfg)
    suite.check("sinabs really has no v_reset attribute of its own",
                not hasattr(layer.lif, "v_reset"))
    suite.check("the reset object is what actually carries it",
                type(layer.lif.reset_fn).__name__ == "MembraneReset")


def test_norse_alpha_warning_fires_once_per_process() -> None:
    """The finding is real and must not be silenced -- but it is ONE fact about the
    config, not one per layer. The network builds a LIF per layer and check_network
    builds the whole network four times over, so an unguarded warning printed the same
    three lines repeatedly and read as three separate problems."""
    import logging

    from frameworks.adapters import norse_lif

    class Collect(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    cfg = fresh_cfg()
    cfg.NEURON["norse"]["surrogate"]["type"] = "super"
    logger = logging.getLogger("frameworks.adapters.norse_lif")
    handler = Collect()
    logger.addHandler(handler)
    try:
        norse_lif.reset_alpha_warning()
        for _ in range(4):
            single_neuron("norse", cfg)
        alpha_warnings = [m for m in handler.messages if "IGNORES alpha" in m]
        suite.check("four norse layers warn exactly once", len(alpha_warnings) == 1,
                    f"{len(alpha_warnings)} warnings")
        suite.check("and the warning still names the measured ratio",
                    alpha_warnings and "6.03x" in alpha_warnings[0])

        # A suppression nothing can clear is a suppression nothing can prove.
        norse_lif.reset_alpha_warning()
        single_neuron("norse", cfg)
        suite.check("reset_alpha_warning re-arms it",
                    len([m for m in handler.messages if "IGNORES alpha" in m]) == 2)

        # circ is the recommended alternative and must stay quiet.
        quiet = fresh_cfg()
        quiet.NEURON["norse"]["surrogate"]["type"] = "circ"
        quiet.NEURON["norse"]["surrogate"]["alpha"] = 0.5
        norse_lif.reset_alpha_warning()
        before = len([m for m in handler.messages if "IGNORES alpha" in m])
        single_neuron("norse", quiet)
        suite.check("circ does not warn",
                    len([m for m in handler.messages if "IGNORES alpha" in m]) == before)
    finally:
        logger.removeHandler(handler)
        norse_lif.reset_alpha_warning()


def test_describe_reads_the_built_module_not_the_config() -> None:
    """A config value is a REQUEST. A framework can rename a constructor argument
    between versions, accept one and ignore it, or clamp it on the way in -- and a
    report that echoed the request would look correct in all three cases.

    So the module is corrupted AFTER construction, exactly the way a silently-ignored
    argument would leave it, and describe() must report the corrupted value rather than
    the config's.
    """
    import torch

    layer = single_neuron("torch")
    wanted = layer.describe()["beta"]
    layer.lif.beta = torch.tensor(0.123)
    got = layer.describe()["beta"]
    suite.check("snntorch beta comes from the module", str(got).startswith("0.123"),
                f"reported {got!r}, config said {wanted!r}")

    layer = single_neuron("sj")
    layer.lif.tau = 7.5
    suite.check("spikingjelly tau comes from the module",
                str(layer.describe()["tau"]).startswith("7.5"))

    layer = single_neuron("norse")
    layer.lif.p = layer.lif.p._replace(v_th=torch.tensor(3.25))
    suite.check("norse v_th comes from the LIFBoxParameters the cell holds",
                str(layer.describe()["v_th"]).startswith("3.25"))

    layer = single_neuron("sinabs")
    layer.lif.min_v_mem = torch.nn.Parameter(torch.tensor(-4.0))
    suite.check("sinabs min_v_mem comes from the module",
                str(layer.describe()["min_v_mem"]).startswith("-4.0"))


def test_a_module_that_drifts_from_its_config_is_flagged() -> None:
    """Reading the module is only half of it: the disagreement has to be visible."""
    import torch

    from frameworks.adapters.base import MISMATCH

    layer = single_neuron("torch")
    layer.lif.beta = torch.tensor(0.123)
    reported = layer.describe()["beta"]
    suite.check("a drifted float is marked MISMATCH", MISMATCH in str(reported))
    suite.check("and the marker names what the config wanted", "0.9" in str(reported),
                f"got {reported!r}")

    # Identity-based checks: the config key and the live object are spelled
    # differently, so only the adapter's own table can tell they agree.
    from sinabs.activation import MembraneReset, SingleSpike

    # Stated, not inherited: the swap has to land on the OTHER option than the config
    # names, and fresh_cfg() is the base config rather than ex2's.
    cfg = fresh_cfg()
    cfg.NEURON["sinabs"]["spike_fn"] = "multi"
    cfg.NEURON["sinabs"]["reset_mechanism"] = "subtract"
    layer = single_neuron("sinabs", cfg)
    layer.lif.spike_fn = SingleSpike
    layer.lif.reset_fn = MembraneReset(reset_value=0.0)
    described = layer.describe()
    suite.check("a swapped spike function is caught", MISMATCH in str(described["spike_fn"]))
    suite.check("a swapped reset object is caught",
                MISMATCH in str(described["reset_mechanism"]))

    from norse.torch.functional.reset import reset_subtract

    layer = single_neuron("norse")
    layer.lif.p = layer.lif.p._replace(reset_method=reset_subtract)
    suite.check("a swapped norse reset function is caught",
                MISMATCH in str(layer.describe()["reset_method"]))


def test_an_untouched_module_is_never_flagged() -> None:
    """The other half: no false positives. A float32 round-trip changes the last bits of
    a config float, and a report that called that a mismatch would train the reader to
    ignore the marker."""
    from frameworks.adapters.base import MISMATCH

    for framework in FRAMEWORKS:
        described = single_neuron(framework).describe()
        flagged = [key for key, value in described.items()
                   if isinstance(value, str) and MISMATCH in value]
        suite.check(f"{framework} reports no spurious mismatch", not flagged,
                    f"flagged {flagged}")


def test_sinabs_decay_gain_is_derived_from_the_module() -> None:
    """The decay/gain pair is what puts all four frameworks on one scale, so it has to
    describe the neuron that will run -- not the one the file asked for."""
    import torch

    cfg = fresh_cfg()
    cfg.NEURON["sinabs"]["tau_mem"] = float("inf")
    cfg.NEURON["sinabs"]["norm_input"] = False
    layer = single_neuron("sinabs", cfg)
    suite.check("tau_mem inf means no leak, gain 1",
                layer.describe()["effective_decay_gain"] == "1.0000/1.0000",
                layer.describe()["effective_decay_gain"])

    layer.lif.tau_mem = torch.nn.Parameter(torch.tensor(20.0))
    decay_gain = layer.describe()["effective_decay_gain"]
    suite.check("a changed module tau_mem changes the reported decay",
                decay_gain.startswith("0.95"), f"got {decay_gain}")


def main() -> int:
    return suite.run([
        test_input_gain_is_one,
        test_decay_is_zero_point_nine,
        test_threshold_is_one,
        test_reset_is_hard_to_zero,
        test_reset_is_immediate_not_delayed,
        test_spike_is_binary,
        test_subthreshold_integration_over_time,
        test_state_lifecycle,
        test_reset_actually_restarts_the_trajectory,
        test_shape_is_preserved,
        test_spike_counting_is_off_by_default,
        test_spike_rate_is_a_true_fraction,
        test_silent_neuron_has_rate_zero,
        test_reset_spike_stats_clears_everything,
        test_counting_does_not_sync_to_host,
        test_describe_names_the_framework_and_its_real_parameters,
        test_sinabs_reports_its_frozen_time_constant,
        test_sinabs_neuron_parameters_are_frozen,
        test_spikingjelly_refuses_multi_step,
        test_unknown_surrogate_is_refused,
        test_sinabs_surrogate_is_actually_read,
        test_sinabs_honours_the_selected_surrogate,
        test_missing_neuron_key_raises,
        test_missing_neuron_block_raises,
        test_changing_the_config_changes_the_neuron,
        test_sinabs_v_reset_is_only_reported_when_the_reset_uses_it,
        test_norse_alpha_warning_fires_once_per_process,
        test_describe_reads_the_built_module_not_the_config,
        test_a_module_that_drifts_from_its_config_is_flagged,
        test_an_untouched_module_is_never_flagged,
        test_sinabs_decay_gain_is_derived_from_the_module,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
