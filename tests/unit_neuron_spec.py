"""Unit tests for skeleton/neuron_spec.py and the Settings wiring behind it.

    python tests/unit_neuron_spec.py

One rule is being enforced here, and these tests exist because breaking it is silent:

    NO SILENT DEFAULTS.

A missing neuron key must raise, not fall back to the framework's own default. Every
framework's default differs from every other's, so a fallback does not produce a
slightly-off run -- it produces a run comparing four different neurons while reporting
that it compared one.

Also covers the Settings-level changes this unit made: the neuron block loading, the
seed, and the shared optimizer/loss replacing the old per-framework `frameworks:` block.
"""
from __future__ import annotations

from _harness import Suite, fresh_cfg
from skeleton import neuron_spec
from skeleton.neuron_spec import (
    NeuronSpecError, describe, neuron_cfg, optional_float, require_bool,
    require_choice, require_float, require_int, require_surrogate,
)
from skeleton.snn_config import FW_TO_CFG_KEY, Settings

suite = Suite("unit_neuron_spec")

FRAMEWORK_KEYS = ["snntorch", "norse", "spikingjelly", "sinabs"]


# ---------------------------------------------------------------------------
# 1. neuron_cfg: locating a framework's block
# ---------------------------------------------------------------------------
def test_every_framework_has_a_block() -> None:
    cfg = fresh_cfg()
    for key in FRAMEWORK_KEYS:
        block = neuron_cfg(cfg, key)
        suite.check(f"neuron.{key} block exists and is a mapping", isinstance(block, dict),
                    type(block).__name__)
        suite.check(f"neuron.{key} block is not empty", bool(block))


def test_missing_block_raises() -> None:
    cfg = fresh_cfg()
    del cfg.NEURON["norse"]
    suite.expect_raises("a missing framework block raises", NeuronSpecError,
                        lambda: neuron_cfg(cfg, "norse"), must_mention=["norse"])


def test_absent_neuron_section_raises() -> None:
    cfg = fresh_cfg()
    cfg.NEURON = {}
    suite.expect_raises("an absent neuron: section raises", NeuronSpecError,
                        lambda: neuron_cfg(cfg, "snntorch"), must_mention=["neuron"])


# ---------------------------------------------------------------------------
# 2. the typed accessors
# ---------------------------------------------------------------------------
def test_require_float_reads_and_validates() -> None:
    block = {"beta": 0.9, "count": 3, "name": "lif", "flag": True}
    suite.check("require_float reads a float", require_float(block, "beta") == 0.9)
    suite.check("require_float accepts an int as a float", require_float(block, "count") == 3.0)
    suite.expect_raises("require_float rejects text", NeuronSpecError,
                        lambda: require_float(block, "name"), must_mention=["name"])
    suite.expect_raises("require_float names a missing key", NeuronSpecError,
                        lambda: require_float(block, "absent"), must_mention=["absent"])


def test_require_bool_is_strict() -> None:
    """YAML turns a bare `no` into a boolean but a quoted "no" into a string, and a
    truthy string would silently invert a setting like decay_input."""
    block = {"flag": True, "text": "true", "number": 1}
    suite.check("require_bool reads a bool", require_bool(block, "flag") is True)
    suite.expect_raises("require_bool rejects the string 'true'", NeuronSpecError,
                        lambda: require_bool(block, "text"), must_mention=["text"])
    suite.expect_raises("require_bool rejects 1", NeuronSpecError,
                        lambda: require_bool(block, "number"), must_mention=["number"])


def test_require_choice_restricts() -> None:
    block = {"reset": "zero"}
    suite.check("require_choice accepts a listed value",
                require_choice(block, "reset", ["zero", "subtract"]) == "zero")
    suite.expect_raises("require_choice rejects an unlisted value", NeuronSpecError,
                        lambda: require_choice(block, "reset", ["subtract"]),
                        must_mention=["reset"])


def test_optional_float_distinguishes_null_from_missing() -> None:
    """An explicit `null` is a decision ("first-order neuron"); a missing key is an
    oversight. They must not look the same."""
    block = {"tau_syn": None, "tau_mem": 9.5}
    suite.check("optional_float returns None for an explicit null",
                optional_float(block, "tau_syn") is None)
    suite.check("optional_float returns a number when set",
                optional_float(block, "tau_mem") == 9.5)
    suite.expect_raises("optional_float still requires the key to be present",
                        NeuronSpecError, lambda: optional_float(block, "absent"),
                        must_mention=["absent"])


def test_require_int() -> None:
    block = {"steps": 4, "text": "4"}
    suite.check("require_int reads an int", require_int(block, "steps") == 4)
    suite.expect_raises("require_int rejects text", NeuronSpecError,
                        lambda: require_int(block, "text"), must_mention=["text"])


def test_require_surrogate() -> None:
    block = {"surrogate": {"type": "atan", "alpha": 2.0}}
    kind, alpha = require_surrogate(block)
    suite.check("require_surrogate returns the type", kind == "atan", kind)
    suite.check("require_surrogate returns alpha", alpha == 2.0, str(alpha))

    suite.expect_raises("require_surrogate rejects a missing type", NeuronSpecError,
                        lambda: require_surrogate({"surrogate": {"alpha": 1.0}}),
                        must_mention=["type"])
    suite.expect_raises("require_surrogate rejects a missing alpha", NeuronSpecError,
                        lambda: require_surrogate({"surrogate": {"type": "atan"}}),
                        must_mention=["alpha"])
    suite.expect_raises("require_surrogate rejects a non-mapping", NeuronSpecError,
                        lambda: require_surrogate({"surrogate": "atan"}))


def test_describe_flattens_the_whole_block() -> None:
    """The run record needs a flat view so a results file can answer 'what neuron was
    this?' without reopening the YAML."""
    flat = describe(fresh_cfg())
    suite.check("describe flattens nested surrogate keys",
                "snntorch.surrogate.type" in flat, str(sorted(flat)[:3]))
    suite.check("describe includes plain keys", flat.get("snntorch.beta") == 0.9,
                str(flat.get("snntorch.beta")))
    suite.check("describe covers all four frameworks",
                all(any(k.startswith(f"{fw}.") for k in flat) for fw in FRAMEWORK_KEYS))


# ---------------------------------------------------------------------------
# 3. the shipped neuron block really describes ONE neuron
# ---------------------------------------------------------------------------
def test_shipped_spec_targets_one_neuron() -> None:
    """The four blocks are written in four different unit systems. These are the
    conversions from network_architecture.yaml's own comments, checked as arithmetic
    rather than trusted as prose."""
    cfg = fresh_cfg()
    import math

    snntorch = neuron_cfg(cfg, "snntorch")
    suite.check("snntorch beta IS the decay", require_float(snntorch, "beta") == 0.9)
    suite.check("snntorch threshold is 1.0", require_float(snntorch, "threshold") == 1.0)

    sj = neuron_cfg(cfg, "spikingjelly")
    sj_decay = 1.0 - 1.0 / require_float(sj, "tau")
    suite.check("spikingjelly 1 - 1/tau == 0.9", abs(sj_decay - 0.9) < 1e-12,
                f"{sj_decay:.12f}")
    suite.check("spikingjelly decay_input is false (gain stays 1.0)",
                require_bool(sj, "decay_input") is False)
    suite.check("spikingjelly threshold is 1.0", require_float(sj, "v_threshold") == 1.0)

    norse = neuron_cfg(cfg, "norse")
    step = require_float(norse, "dt") * require_float(norse, "tau_mem_inv")
    suite.check("norse 1 - dt*tau_mem_inv == 0.9", abs((1.0 - step) - 0.9) < 1e-12,
                f"{1.0 - step:.12f}")
    suite.check("norse input_scale cancels the locked gain",
                abs(step * require_float(norse, "input_scale") - 1.0) < 1e-12,
                f"{step * require_float(norse, 'input_scale'):.12f}")
    suite.check("norse threshold is 1.0", require_float(norse, "v_th") == 1.0)

    sinabs = neuron_cfg(cfg, "sinabs")
    sinabs_decay = math.exp(-1.0 / require_float(sinabs, "tau_mem"))
    suite.check("sinabs exp(-1/tau_mem) == 0.9", abs(sinabs_decay - 0.9) < 1e-12,
                f"{sinabs_decay:.15f}")
    suite.check("sinabs norm_input is false (gain stays 1.0)",
                require_bool(sinabs, "norm_input") is False)
    suite.check("sinabs threshold is 1.0", require_float(sinabs, "spike_threshold") == 1.0)


def test_sinabs_tau_is_written_to_full_precision() -> None:
    """At 4 dp the derived alpha is 2.16e-07 low, which over 90 timesteps drifts the
    membrane past a 1e-6 tolerance. Nothing is wrong at 4 dp, but the margin is spent
    for no reason -- so the full-precision value is the one that must be in the file."""
    tau = require_float(neuron_cfg(fresh_cfg(), "sinabs"), "tau_mem")
    suite.check("sinabs tau_mem carries more than 4 decimal places",
                len(str(tau).split(".")[-1]) > 6, str(tau))


def test_every_framework_pins_a_hard_reset() -> None:
    cfg = fresh_cfg()
    suite.check("snntorch hard reset",
                require_choice(neuron_cfg(cfg, "snntorch"), "reset_mechanism",
                               ["zero", "subtract"]) == "zero")
    suite.check("snntorch reset is not delayed",
                require_bool(neuron_cfg(cfg, "snntorch"), "reset_delay") is False)
    suite.check("norse hard reset",
                require_choice(neuron_cfg(cfg, "norse"), "reset_method",
                               ["value", "subtract"]) == "value")
    suite.check("sinabs hard reset",
                require_choice(neuron_cfg(cfg, "sinabs"), "reset_mechanism",
                               ["zero", "subtract"]) == "zero")
    suite.check("sinabs emits a single spike",
                require_choice(neuron_cfg(cfg, "sinabs"), "spike_fn",
                               ["single", "multi"]) == "single")
    suite.check("sinabs is first-order (tau_syn null)",
                optional_float(neuron_cfg(cfg, "sinabs"), "tau_syn") is None)
    suite.check("norse uses the first-order box cell",
                require_choice(neuron_cfg(cfg, "norse"), "cell", ["lif_box"]) == "lif_box")


# ---------------------------------------------------------------------------
# 4. the Settings changes this unit made
# ---------------------------------------------------------------------------
def test_settings_exposes_the_new_fields() -> None:
    cfg = Settings()
    for field in ["NEURON", "NEURON_TYPES", "SEED", "OPTIMIZER", "LEARNING_RATE",
                  "WEIGHT_DECAY", "SGD_MOMENTUM", "LOSS_FN"]:
        suite.check(f"Settings exposes {field}", hasattr(cfg, field))


def test_old_per_framework_block_is_gone() -> None:
    """Neuron params moved to network_architecture.yaml and optimizer/loss became one
    shared setting, so the per-framework block should no longer exist at all."""
    cfg = Settings()
    suite.check("Settings no longer has FRAMEWORK_CFG", not hasattr(cfg, "FRAMEWORK_CFG"))
    suite.check("SNN_module.yaml no longer has a frameworks: block",
                "frameworks" not in cfg.config, str(sorted(cfg.config)))


def test_active_fw_cfg_points_at_the_neuron_block() -> None:
    cfg = Settings()
    for selector, key in FW_TO_CFG_KEY.items():
        cfg.FRAMEWORK = selector
        suite.check(f"active_fw_cfg resolves to neuron.{key}",
                    cfg.active_fw_cfg == cfg.NEURON[key])
    cfg.FRAMEWORK = "brian2"
    suite.expect_raises("active_fw_cfg raises on an unknown framework", ValueError,
                        lambda: cfg.active_fw_cfg, must_mention=["brian2"])


def test_comparison_relevant_defaults() -> None:
    """Settings values that would quietly invalidate a comparison run if they drifted."""
    cfg = Settings()
    suite.check("AMP is off and stated in the YAML", cfg.USE_AMP is False, str(cfg.USE_AMP))
    suite.check("use_amp is present in the YAML, not defaulted",
                "use_amp" in cfg.config.get("training", {}))
    suite.check("activity regularisation is off", cfg.ACTIVITY_REG_ENABLED is False)
    suite.check("seed is present in the YAML, not defaulted",
                "seed" in cfg.config.get("training", {}))
    suite.check("optimizer defaults to nadam", cfg.OPTIMIZER == "nadam", cfg.OPTIMIZER)
    suite.check("weight decay defaults to 0.0", cfg.WEIGHT_DECAY == 0.0, str(cfg.WEIGHT_DECAY))
    suite.check("loss defaults to cross_entropy", cfg.LOSS_FN == "cross_entropy", cfg.LOSS_FN)


def test_display_runs_without_the_old_block() -> None:
    """display() read FRAMEWORK_CFG before this unit. It must still work now that the
    block is gone, or every run's header crashes."""
    import contextlib
    import io

    cfg = Settings()
    for selector in FW_TO_CFG_KEY:
        cfg.FRAMEWORK = selector
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                cfg.display()
        except Exception as error:  # noqa: BLE001
            suite.check(f"display() works for {selector}", False,
                        f"{type(error).__name__}: {error}")
            continue
        text = buffer.getvalue()
        suite.check(f"display() works for {selector}", "NEURON SPEC" in text)
        suite.check(f"display() shows the seed for {selector}", "Seed" in text)
        suite.check(f"display() shows the optimizer for {selector}", "Optimizer" in text)


def main() -> int:
    return suite.run([
        test_every_framework_has_a_block,
        test_missing_block_raises,
        test_absent_neuron_section_raises,
        test_require_float_reads_and_validates,
        test_require_bool_is_strict,
        test_require_choice_restricts,
        test_optional_float_distinguishes_null_from_missing,
        test_require_int,
        test_require_surrogate,
        test_describe_flattens_the_whole_block,
        test_shipped_spec_targets_one_neuron,
        test_sinabs_tau_is_written_to_full_precision,
        test_every_framework_pins_a_hard_reset,
        test_settings_exposes_the_new_fields,
        test_old_per_framework_block_is_gone,
        test_active_fw_cfg_points_at_the_neuron_block,
        test_comparison_relevant_defaults,
        test_display_runs_without_the_old_block,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
