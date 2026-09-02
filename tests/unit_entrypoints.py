"""Unit tests for the three entry points themselves.

    python tests/unit_entrypoints.py

unit_cli_config.py covers the config loader and the shared CLI. This covers the
scripts that sit on top of them: check_network.py, equivalence_check.py, and
learning/main.py's argument parsing.

Each is exercised through its real `main()` with a patched argv, so what is tested is
what someone actually types -- not an internal function that happens to be importable.
Still CPU-only, still no dataset and no download.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import pathlib
import sys
import tempfile
from pathlib import Path

from _harness import Suite, fresh_cfg

REPO_ROOT = Path(__file__).resolve().parent.parent
suite = Suite("unit_entrypoints")


def run_script(module_name: str, argv: list[str]) -> tuple[int, str]:
    """Call a script's main() with argv patched, capturing stdout.

    Returns (exit_code, output). A SystemExit is caught so an argparse failure is a
    result to assert on rather than something that kills the suite.
    """
    module = importlib.import_module(module_name)
    saved = sys.argv
    buffer = io.StringIO()
    sys.argv = [f"{module_name}.py", *argv]
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = module.main()
    except SystemExit as exit_signal:      # argparse --help / bad flag
        code = int(exit_signal.code or 0)
    finally:
        sys.argv = saved
    return code, buffer.getvalue()


def write_overlay(tmpdir: Path, name: str, text: str) -> Path:
    path = tmpdir / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. check_network.py
# ---------------------------------------------------------------------------
def test_shape_without_download_from_the_registry() -> None:
    """The registry stores sensor size and class count as plain data, so a shape check
    must need no download -- that is what lets this run on a laptop with no data."""
    import check_network

    cfg = fresh_cfg()
    cfg.DATASET_NAME = "DVS128 Gesture"
    note = check_network.apply_shape_without_download(cfg)
    suite.check("sensor size taken from the registry",
                (cfg.SENSOR_H, cfg.SENSOR_W) == (128, 128), f"{cfg.SENSOR_H}x{cfg.SENSOR_W}")
    suite.check("class count taken from the registry", cfg.NUM_CLASSES == 11,
                str(cfg.NUM_CLASSES))
    suite.check("FC_IN recomputed for the new sensor", cfg.FC_IN == 26912, str(cfg.FC_IN))
    suite.check("the note says nothing was downloaded", "nothing downloaded" in note, note)


def test_shape_falls_back_to_the_convolution_block() -> None:
    """With no dataset named it must NOT prompt -- a shape check should never block."""
    import check_network

    cfg = fresh_cfg()
    cfg.DATASET_NAME = None
    note = check_network.apply_shape_without_download(cfg)
    suite.check("falls back to the convolution: block", "no dataset.name set" in note, note)
    suite.check("sensor left as configured", (cfg.SENSOR_H, cfg.SENSOR_W) == (34, 34))


def test_shape_raises_on_a_typo() -> None:
    from event_data_workflow.dataset_registry import UnknownDataset
    import check_network

    cfg = fresh_cfg()
    cfg.DATASET_NAME = "DVS128 Gestrue"
    suite.expect_raises("a typo'd dataset raises before any work", UnknownDataset,
                        lambda: check_network.apply_shape_without_download(cfg),
                        must_mention=["DVS128 Gesture"])


def test_check_network_all_passes_and_reports() -> None:
    code, out = run_script("check_network", ["--all"])
    suite.check("check_network --all exits 0", code == 0, f"exit {code}")
    suite.check("it reports OVERALL PASS", "OVERALL: PASS" in out)
    suite.check("it prints one row per framework",
                all(f"\n{fw:<12}" in out for fw in ["torch", "norse", "sj", "sinabs"]))
    suite.check("it reports the shared parameter count", "18,254" in out)
    suite.check("it names the neuron actually built", "NEURON ACTUALLY BUILT" in out)


def test_check_network_single_framework_shows_shapes() -> None:
    code, out = run_script("check_network", ["--framework", "sinabs"])
    suite.check("single-framework run exits 0", code == 0, f"exit {code}")
    suite.check("layer table printed", "layer by layer" in out)
    suite.check("the measured flatten width is shown", "flatten width measured" in out)
    suite.check("it states the formula agrees", "agree" in out and "True" in out)
    suite.check("the forward pass output shape is shown", "[T, batch, num_classes]" in out)


def test_check_network_honours_the_overlay() -> None:
    code, out = run_script("check_network",
                           ["--config", "experiments/ex2/config.yaml", "--framework", "sinabs"])
    suite.check("overlay run exits 0", code == 0, f"exit {code}")
    suite.check("the banner names the overlay", "experiments/ex2/config.yaml" in out)
    suite.check("the dataset came from the overlay, not a prompt", "N-MNIST" in out)
    suite.check("ex2's leak-free sinabs is reported", "inf" in out)


# ---------------------------------------------------------------------------
# 2. equivalence_check.py
# ---------------------------------------------------------------------------
def test_equivalence_pins_cpu() -> None:
    """Hardcoded, not a flag: one neuron for 90 steps gains nothing from a GPU, and
    float non-determinism would undermine a test built on exact comparison."""
    import equivalence_check

    cfg = fresh_cfg()
    cfg.DEVICE = "cuda"
    equivalence_check.make_cfg(cfg)
    suite.check("make_cfg forces cpu", cfg.DEVICE == "cpu", cfg.DEVICE)

    parser_args = equivalence_check.parse_args
    suite.check("parse_args exists", callable(parser_args))


def test_equivalence_takes_no_framework_or_seed_flag() -> None:
    """It builds all four and the poisson pattern carries its own seed, so neither
    flag would mean anything."""
    code, out = run_script("equivalence_check", ["--framework", "norse"])
    suite.check("--framework is rejected", code != 0, f"exit {code}")
    code, out = run_script("equivalence_check", ["--seed", "3"])
    suite.check("--seed is rejected", code != 0, f"exit {code}")


def test_input_patterns_stay_below_threshold() -> None:
    """Not a style point. An amplitude at or above threshold fires every step, leaving
    the membrane flat at 0 -- four broken neurons would agree on a flat line too."""
    import equivalence_check as eq

    for name, xs in eq.PATTERNS.items():
        suite.check(f"{name}: every amplitude is below threshold 1.0", max(xs) < 1.0,
                    f"max {max(xs)}")
        suite.check(f"{name}: the pattern is not all zeros", any(x > 0 for x in xs))
        suite.check(f"{name}: 90 timesteps", len(xs) == 90, str(len(xs)))


def test_constant_step_shape() -> None:
    import equivalence_check as eq

    xs = eq.constant_step(T=20, amplitude=0.15, onset=5)
    suite.check("silent before onset", xs[:5] == [0.0] * 5)
    suite.check("constant after onset", xs[5:] == [0.15] * 15)


def test_poisson_is_reproducible_and_sparse() -> None:
    import equivalence_check as eq

    first = eq.poisson(T=200, rate=0.15, amplitude=0.6, seed=1234)
    second = eq.poisson(T=200, rate=0.15, amplitude=0.6, seed=1234)
    suite.check("poisson is reproducible from its seed", first == second)
    suite.check("a different seed gives a different train",
                first != eq.poisson(T=200, rate=0.15, amplitude=0.6, seed=99))
    active = sum(1 for x in first if x > 0)
    suite.check("roughly the requested rate of events", 0.08 < active / 200 < 0.25,
                f"{active}/200")
    suite.check("every event has the requested amplitude",
                {x for x in first if x > 0} == {0.6})


def test_drive_returns_one_reading_per_timestep() -> None:
    import equivalence_check as eq

    cfg = eq.make_cfg(fresh_cfg())
    xs = [0.0] * 3 + [0.15] * 20
    for framework in ["torch", "norse", "sj", "sinabs"]:
        spikes, mems = eq.drive(framework, cfg, xs)
        suite.check(f"one spike reading per timestep: {framework}", len(spikes) == len(xs))
        suite.check(f"one membrane reading per timestep: {framework}", len(mems) == len(xs))
        suite.check(f"spikes are binary: {framework}", set(spikes) <= {0.0, 1.0},
                    str(sorted(set(spikes))))
        suite.check(f"the neuron actually fires on this ramp: {framework}",
                    sum(spikes) > 0, str(sum(spikes)))


def test_equivalence_reports_agreement_on_the_base_config() -> None:
    code, out = run_script("equivalence_check", [])
    suite.check("base config exits 0", code == 0, f"exit {code}")
    suite.check("all framework-patterns agree", "8/8 framework-patterns" in out, out[-400:])
    suite.check("no framework is listed as differing", "differing" not in out)


def test_equivalence_measures_but_does_not_judge() -> None:
    """The flaw found while wiring this up: a hard PASS/FAIL called ex2's own result an
    error. ex2 deliberately gives sinabs no leak, multi-spike and a subtract reset, so
    a large deviation IS the experiment's finding. It must be reported, and must not
    fail the run."""
    code, out = run_script("equivalence_check", ["--config", "experiments/ex2/config.yaml"])
    suite.check("a deliberately divergent config still exits 0", code == 0, f"exit {code}")
    suite.check("sinabs is reported as differing", "sinabs" in out and "differing" in out)
    # ex2 moves ALL FOUR neurons to their own defaults, so only the reference agrees
    # with itself -- 2 of 8 framework-patterns (torch, in both patterns). A wholesale
    # divergence IS the result here, which is exactly why this script does not gate.
    suite.check("every non-reference framework is reported as differing",
                "2/8 framework-patterns" in out, out[-400:])
    for framework in ("norse", "sj", "sinabs"):
        suite.check(f"{framework} differs under ex2", f"{framework} (" in out)
    suite.check("it says the deviation may be the intended result",
                "this IS the" in out or "deliberately varies" in out)
    suite.check("no PASS/FAIL verdict is issued", "GATE A3" not in out)


def test_equivalence_states_the_yardstick_is_not_a_gate() -> None:
    _, out = run_script("equivalence_check", [])
    suite.check("the yardstick is labelled as not enforced", "not enforced" in out)
    suite.check("the configured neuron is printed for the record",
                "configured neuron" in out)


# ---------------------------------------------------------------------------
# 3. learning/main.py argument parsing
# ---------------------------------------------------------------------------
def test_main_parses_no_arguments() -> None:
    """The colleague's path. Every flag must be optional."""
    from learning.main import parse_args

    saved = sys.argv
    sys.argv = ["main.py"]
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    suite.check("no --config by default", args.config is None)
    suite.check("no --framework by default", args.framework is None)
    suite.check("no --seed by default", args.seed is None)
    suite.check("no --experiment by default", args.experiment is None)


def test_main_parses_the_full_flag_set() -> None:
    from learning.main import parse_args

    saved = sys.argv
    sys.argv = ["main.py", "--config", "experiments/ex2/config.yaml", "--experiment", "ex2",
                "--framework", "sinabs", "--seed", "3",
                "--results-root", "/drive/runs", "--cache-root", "/content/cache"]
    try:
        args = parse_args()
    finally:
        sys.argv = saved
    suite.check("--config parsed", args.config == "experiments/ex2/config.yaml")
    suite.check("--experiment parsed", args.experiment == "ex2")
    suite.check("--framework parsed", args.framework == "sinabs")
    suite.check("--seed parsed as an int", args.seed == 3 and isinstance(args.seed, int))
    suite.check("--results-root parsed", args.results_root == "/drive/runs")
    suite.check("--cache-root parsed", args.cache_root == "/content/cache")


def test_main_has_no_device_flag() -> None:
    """Dropped deliberately: device stays training.device in the config."""
    from learning.main import parse_args

    saved = sys.argv
    sys.argv = ["main.py", "--device", "cpu"]
    try:
        parse_args()
    except SystemExit:
        suite.check("--device is rejected", True)
    else:
        suite.check("--device is rejected", False, "it was accepted")
    finally:
        sys.argv = saved


# ---------------------------------------------------------------------------
# 4. the scripts survive an overlay that changes the architecture
# ---------------------------------------------------------------------------
def test_entrypoints_follow_an_architecture_change() -> None:
    """An overlay that widens conv1 must flow all the way through: FC_IN recomputed,
    the probe agreeing with it, and the parameter count changing."""
    with tempfile.TemporaryDirectory() as tmp:
        overlay = write_overlay(Path(tmp), "wide.yaml", "convolution:\n  conv1_out: 20\n")
        code, out = run_script("check_network", ["--config", str(overlay), "--all"])
        suite.check("check_network handles a wider conv1", code == 0, f"exit {code}")
        suite.check("the parameter count is no longer the default 18,254",
                    "18,254" not in out)
        suite.check("all four still start from identical weights", "OVERALL: PASS" in out)


# ---------------------------------------------------------------------------
# 5. the advertised flag set -- HOW_TO_RUN.md documents these tables
# ---------------------------------------------------------------------------
def test_each_script_advertises_exactly_the_flags_it_can_act_on() -> None:
    """A flag that appears in --help but does nothing is the failure this whole merge
    has been removing. These sets are the ones documented in HOW_TO_RUN.md; if a script
    gains or loses a flag, this fails and the doc gets updated with it."""
    expected = {
        "learning.main": {"config", "framework", "seed", "experiment",
                          "results_root", "cache_root", "inference"},
        "check_network": {"config", "framework", "seed", "experiment",
                          "all", "batch", "timesteps"},
        # Writes FIGURES, so it takes results_root (they must be able to land on
        # mounted Drive on Colab) and formats. No cache_root -- it touches no dataset.
        # Builds all four, so no framework; the poisson pattern carries its own seed.
        "equivalence_check": {"config", "experiment", "results_root", "formats"},
        # Reads the environment and requirements.txt only. No config, no experiment,
        # no roots -- it writes nothing and knows nothing about a run.
        "check_env": {"strict"},
    }
    for module_name, flags in expected.items():
        module = importlib.import_module(module_name)
        saved = sys.argv
        sys.argv = [f"{module_name}.py"]
        try:
            actual = set(vars(module.parse_args()))
        finally:
            sys.argv = saved
        suite.check(f"{module_name} advertises exactly its documented flags",
                    actual == flags,
                    f"extra {sorted(actual - flags)}, missing {sorted(flags - actual)}")


def test_equivalence_check_roots() -> None:
    """It writes figures, so --results-root must work (Colab Drive). It reads no
    dataset, so --cache-root must NOT exist."""
    code, _ = run_script("equivalence_check", ["--cache-root", "/tmp/x"])
    suite.check("equivalence_check rejects --cache-root", code != 0, f"exit {code}")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        code, out = run_script("equivalence_check",
                               ["--experiment", "extest", "--results-root", tmp])
        suite.check("equivalence_check accepts --results-root with --experiment",
                    code == 0, f"exit {code}")
        written = list(pathlib.Path(tmp).rglob("EQ_*.png"))
        suite.check("figures land under the given results root", len(written) == 2,
                    f"{len(written)} figures: {[p.name for p in written]}")


def test_equivalence_check_formats() -> None:
    """PDF is vector, which a written report needs."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        code, _ = run_script("equivalence_check",
                             ["--experiment", "extest", "--results-root", tmp,
                              "--formats", "png,pdf"])
        suite.check("a multi-format run succeeds", code == 0, f"exit {code}")
        root = pathlib.Path(tmp)
        suite.check("PNG written", len(list(root.rglob("EQ_*.png"))) == 2)
        suite.check("PDF written", len(list(root.rglob("EQ_*.pdf"))) == 2)


def test_check_network_has_no_output_roots() -> None:
    """It writes nothing at all, so neither root should exist."""
    for flag in ["--results-root", "--cache-root"]:
        code, _ = run_script("check_network", [flag, "/tmp/x"])
        suite.check(f"check_network rejects {flag}", code != 0, f"exit {code}")


def _banner_rows(output: str) -> dict[str, str]:
    """The `  label   value` rows of the LEADING banner only, as a dict.

    Scoped to the region between the banner's own two `===` rules. The neuron blocks
    further down print `      framework   snntorch` in the same shape, so a parser that
    just scanned indented lines would report a framework row the banner never printed.
    """
    lines = output.splitlines()
    rules = [i for i, line in enumerate(lines) if set(line.strip()) == {"="} and line.strip()]
    if len(rules) < 2:
        return {}
    rows = {}
    for line in lines[rules[0] + 1:rules[1]]:
        if not line.startswith("  "):
            continue
        parts = line[2:].split(None, 1)
        if len(parts) == 2:
            rows.setdefault(parts[0], parts[1].strip())
    return rows


def test_check_network_banner_claims_only_what_it_uses() -> None:
    """A banner row is a claim about the run, and a wrong claim is worse than silence.

    check_network builds on the CPU whatever `training.device` says (see the module
    docstring: "no download, no GPU"), and under --all it builds EVERY framework rather
    than cfg.FRAMEWORK. The `shape` row already names the dataset, with its size, class
    count and provenance.
    """
    code, out = run_script("check_network", ["--config", "experiments/ex2/config.yaml", "--all"])
    suite.check("--all still passes", code == 0, f"exit {code}")
    rows = _banner_rows(out)
    for absent in ("device", "dataset", "framework"):
        suite.check(f"--all banner does not claim a {absent}", absent not in rows,
                    f"found {absent}={rows.get(absent)!r}")
    for present in ("config", "seed", "shape"):
        suite.check(f"--all banner still states the {present}", present in rows)
    suite.check("the shape row carries the dataset name", "N-MNIST" in rows.get("shape", ""))

    # Without --all exactly one framework IS inspected, so naming it is correct.
    code, out = run_script("check_network",
                           ["--config", "experiments/ex2/config.yaml", "--framework", "sinabs"])
    suite.check("single-framework run passes", code == 0, f"exit {code}")
    rows = _banner_rows(out)
    suite.check("without --all the framework IS named", rows.get("framework") == "sinabs",
                f"got {rows.get('framework')!r}")
    suite.check("device stays absent -- it is still CPU-only", "device" not in rows)


def test_norse_warning_is_not_repeated_per_layer() -> None:
    """ex2 selects norse's own 'super' surrogate, whose alpha norse 1.1.0 ignores. The
    warning is worth printing; printing it once per LIF layer, four networks over, read
    as a dozen separate problems."""
    from frameworks.adapters import norse_lif

    norse_lif.reset_alpha_warning()
    _code, out = run_script("check_network",
                            ["--config", "experiments/ex2/config.yaml", "--all"])
    hits = out.count("IGNORES alpha")
    suite.check("the whole --all run warns at most once", hits <= 1, f"{hits} times")
    suite.check("torch's namedtuple pytree noise is filtered at the norse import",
                "is a subclass of `collections.namedtuple`" not in out)


def test_check_network_fails_when_a_neuron_drifts_from_its_config() -> None:
    """The inline MISMATCH marker has to reach the exit code, or a CI step and a reader
    skimming the last line would both call a broken neuron spec green."""
    import torch

    import frameworks.adapters.snntorch_lif as snntorch_lif

    real_build = snntorch_lif.build_lif

    def drifting(cfg):
        lif = real_build(cfg)
        lif.beta = torch.tensor(0.123)   # as if the constructor had ignored the argument
        return lif

    snntorch_lif.build_lif = drifting
    try:
        code, out = run_script("check_network",
                               ["--config", "experiments/ex2/config.yaml", "--all"])
    finally:
        snntorch_lif.build_lif = real_build

    suite.check("a drifted neuron fails the run", code != 0, f"exit {code}")
    suite.check("the verdict names the neuron check",
                "every neuron holds the value its config asked for" in out)
    suite.check("OVERALL is FAIL", "OVERALL: FAIL" in out)
    suite.check("the offending value is named", "torch.beta" in out, out[-600:])
    suite.check("weights still pass -- only the neuron drifted",
                "PASS  every framework starts from identical weights" in out)

    # And the clean config must still pass, so the check above is not just noise.
    code, out = run_script("check_network",
                           ["--config", "experiments/ex2/config.yaml", "--all"])
    suite.check("the unmodified config still passes", code == 0, f"exit {code}")
    suite.check("both checks report PASS", out.count("  PASS  ") == 2, out[-400:])


def test_inference_mode_can_be_stated_instead_of_asked() -> None:
    """The prompt sits BETWEEN training and testing, so it blocks after the expensive
    part -- a Colab cell would train for ten minutes and then wait for a keypress.
    Stating the mode skips it; omitting it keeps the prompt, the same rule dataset.name
    uses for the dataset question."""
    from learning.utilities import select_inference_mode

    suite.check("--inference stats means statistics only",
                select_inference_mode("stats") is False)
    suite.check("--inference visual means show the window",
                select_inference_mode("visual") is True)
    suite.expect_raises("an unknown mode raises rather than guessing", ValueError,
                        lambda: select_inference_mode("graphs"),
                        must_mention=["stats", "visual"])

    # Omitted -> the prompt runs. With no stdin it must fall back, not crash: that is
    # the path a backgrounded Colab cell or a CI job takes.
    saved_stdin = sys.stdin
    sys.stdin = io.StringIO("")          # empty -> input() raises EOFError
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            fallback = select_inference_mode()
    finally:
        sys.stdin = saved_stdin
    suite.check("no preset and no stdin falls back to statistics only", fallback is False)
    suite.check("and the prompt was actually printed", "Inference output" in out.getvalue())

    # argparse must reject a bad value before any work starts. learning/main.py runs at
    # module level, so it is driven through parse_args() the way the flag-set test is.
    import learning.main as main_module

    saved_argv, saved_err = sys.argv, sys.stderr
    sys.argv = ["main.py", "--inference", "graphs"]
    sys.stderr = io.StringIO()
    try:
        main_module.parse_args()
        rejected = False
    except SystemExit:
        rejected = True
    finally:
        sys.argv, sys.stderr = saved_argv, saved_err
    suite.check("argparse rejects an unknown --inference value", rejected)

    sys.argv = ["main.py", "--inference", "visual"]
    try:
        suite.check("a valid value parses through", main_module.parse_args().inference == "visual")
    finally:
        sys.argv = saved_argv


def main() -> int:
    return suite.run([
        test_each_script_advertises_exactly_the_flags_it_can_act_on,
        test_equivalence_check_roots,
        test_equivalence_check_formats,
        test_check_network_has_no_output_roots,
        test_shape_without_download_from_the_registry,
        test_shape_falls_back_to_the_convolution_block,
        test_shape_raises_on_a_typo,
        test_check_network_all_passes_and_reports,
        test_check_network_single_framework_shows_shapes,
        test_check_network_honours_the_overlay,
        test_check_network_banner_claims_only_what_it_uses,
        test_norse_warning_is_not_repeated_per_layer,
        test_check_network_fails_when_a_neuron_drifts_from_its_config,
        test_equivalence_pins_cpu,
        test_equivalence_takes_no_framework_or_seed_flag,
        test_input_patterns_stay_below_threshold,
        test_constant_step_shape,
        test_poisson_is_reproducible_and_sparse,
        test_drive_returns_one_reading_per_timestep,
        test_equivalence_reports_agreement_on_the_base_config,
        test_equivalence_measures_but_does_not_judge,
        test_equivalence_states_the_yardstick_is_not_a_gate,
        test_main_parses_no_arguments,
        test_main_parses_the_full_flag_set,
        test_main_has_no_device_flag,
        test_inference_mode_can_be_stated_instead_of_asked,
        test_entrypoints_follow_an_architecture_change,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
