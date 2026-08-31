"""Unit tests for the run surface: config loading, overlays, the CLI, and dataset
selection.

    python tests/unit_cli_config.py

Two properties are being protected, and they pull in opposite directions:

  1. `python learning/main.py` with NO flags must behave exactly as this pipeline
     always has -- three base config files, a dataset prompt, output to the `output:`
     paths. Every new flag defaults to that.

  2. `--config config/exN.yaml --experiment exN` must route a per-experiment run
     without any config file knowing which machine it is on.

Plus one rule shared with the rest of the merge: nothing is silently ignored. A typo'd
overlay section, an unknown dataset name, or a flag that cannot take effect all raise.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from _harness import Suite
from event_data_workflow.dataset_registry import (
    DATASET_REGISTRY, UnknownDataset, lookup_dataset, normalise_dataset_name,
)
from skeleton.cli import CliError, add_common_args, build, config_hash, output_dirs, run_banner
from skeleton.config_loader import (
    BASE_FILES, ConfigError, check_no_section_collisions, deep_merge, load_base,
    load_config, load_overlay,
)
from skeleton.snn_config import Settings
from skeleton.workflow_config import WorkflowSettings

suite = Suite("unit_cli_config")

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse(argv: list[str], **kwargs) -> argparse.Namespace:
    return add_common_args(argparse.ArgumentParser(), **kwargs).parse_args(argv)


def write_overlay(tmpdir: Path, name: str, text: str) -> Path:
    path = tmpdir / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. deep_merge
# ---------------------------------------------------------------------------
def test_deep_merge() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3, "list": [1, 2]}
    over = {"a": {"y": 20}, "c": 4, "list": [9]}
    merged = deep_merge(base, over)
    suite.check("nested keys merge, not replace", merged["a"] == {"x": 1, "y": 20},
                str(merged["a"]))
    suite.check("untouched keys survive", merged["b"] == 3)
    suite.check("new keys are added", merged["c"] == 4)
    suite.check("lists are REPLACED, not concatenated", merged["list"] == [9],
                str(merged["list"]))
    suite.check("the base dict is not mutated", base["a"] == {"x": 1, "y": 2})


# ---------------------------------------------------------------------------
# 2. the three base files
# ---------------------------------------------------------------------------
def test_base_loads_and_covers_every_section() -> None:
    base = load_base()
    # No "architecture": that block held the legacy MLP params (input_size,
    # hidden_size, hidden_layers, leak, override, network_struct, simulator), which
    # described a network this pipeline no longer builds. Its one live key, the slice
    # duration, moved to temporal_slicing in data_workflow.yaml.
    for section in ["training", "output", "dataset",
                    "convolution", "neuron_types", "neuron",
                    "framing", "temporal_slicing", "augmentation", "cache",
                    "resource_policy"]:
        suite.check(f"base config has '{section}'", section in base)


def test_no_section_collisions_in_the_shipped_files() -> None:
    """The whole flat-overlay design rests on this. Asserted, not assumed -- adding
    `training:` to data_workflow.yaml later would silently let one file win."""
    try:
        load_base()
        suite.check("the three base files share no top-level section", True)
    except ConfigError as error:
        suite.check("the three base files share no top-level section", False, str(error))

    suite.expect_raises("a collision is detected when one exists", ConfigError,
                        lambda: check_no_section_collisions({
                            "a.yaml": {"training": {}}, "b.yaml": {"training": {}},
                        }), must_mention=["training"])


def test_base_files_are_all_read() -> None:
    suite.check("all three base filenames are declared", len(BASE_FILES) == 3,
                str(sorted(BASE_FILES)))
    for name in BASE_FILES:
        suite.check(f"{name} exists on disk", (REPO_ROOT / "configuration" / name).is_file())


# ---------------------------------------------------------------------------
# 3. overlays
# ---------------------------------------------------------------------------
def test_overlay_overrides_only_what_it_names() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        overlay = write_overlay(Path(tmp), "ov.yaml",
                                "training:\n  epochs: 99\nneuron:\n  snntorch:\n    beta: 0.5\n")
        merged = load_config(overlay)
        base = load_base()
        suite.check("the named key is overridden", merged["training"]["epochs"] == 99)
        suite.check("a sibling in the same block survives",
                    merged["training"]["batch_size"] == base["training"]["batch_size"])
        suite.check("a nested key is overridden",
                    merged["neuron"]["snntorch"]["beta"] == 0.5)
        suite.check("the other frameworks' neurons are untouched",
                    merged["neuron"]["norse"] == base["neuron"]["norse"])
        suite.check("sections in the OTHER base files survive",
                    merged["convolution"] == base["convolution"])
        suite.check("data_workflow sections survive", merged["framing"] == base["framing"])


def test_overlay_reaches_every_base_file() -> None:
    """The point of merging the three: one flat overlay can address any of them."""
    with tempfile.TemporaryDirectory() as tmp:
        overlay = write_overlay(Path(tmp), "ov.yaml",
                                "training:\n  epochs: 7\n"       # SNN_module.yaml
                                "convolution:\n  conv1_out: 16\n"  # network_architecture.yaml
                                "framing:\n  n_time_bins: 25\n")   # data_workflow.yaml
        cfg = Settings(config=load_config(overlay))
        wf = WorkflowSettings(config=load_config(overlay))
        suite.check("overlay reaches SNN_module.yaml", cfg.EPOCHS == 7, str(cfg.EPOCHS))
        suite.check("overlay reaches network_architecture.yaml", cfg.CONV1_OUT == 16,
                    str(cfg.CONV1_OUT))
        suite.check("overlay reaches data_workflow.yaml", wf.N_TIME_BINS == 25,
                    str(wf.N_TIME_BINS))


def test_extends_chain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        write_overlay(tmpdir, "parent.yaml", "training:\n  epochs: 5\n  batch_size: 64\n")
        child = write_overlay(tmpdir, "child.yaml",
                              "extends: parent.yaml\ntraining:\n  epochs: 9\n")
        merged = load_config(child)
        suite.check("child overrides the parent", merged["training"]["epochs"] == 9)
        suite.check("the parent's other keys are inherited",
                    merged["training"]["batch_size"] == 64)
        suite.check("'extends' itself does not leak into the config",
                    "extends" not in merged)


def test_circular_extends_is_caught() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        write_overlay(tmpdir, "a.yaml", "extends: b.yaml\n")
        write_overlay(tmpdir, "b.yaml", "extends: a.yaml\n")
        suite.expect_raises("a circular extends chain raises", ConfigError,
                            lambda: load_overlay(tmpdir / "a.yaml"),
                            must_mention=["circular"])


def test_typo_in_an_overlay_section_raises() -> None:
    """A misspelled section is invisible otherwise: the run proceeds on base values
    and the experiment quietly does not happen."""
    with tempfile.TemporaryDirectory() as tmp:
        overlay = write_overlay(Path(tmp), "ov.yaml", "trianing:\n  epochs: 3\n")
        suite.expect_raises("an unknown overlay section raises", ConfigError,
                            lambda: load_config(overlay), must_mention=["trianing"])


def test_missing_overlay_file_raises() -> None:
    suite.expect_raises("a missing --config file raises", ConfigError,
                        lambda: load_config("config/does_not_exist.yaml"),
                        must_mention=["not found"])


def test_shipped_ex2_overlay_is_valid() -> None:
    # Each experiment's config lives IN its experiment folder, beside the README that
    # describes it -- not in a separate config/ tree that would drift away from it.
    path = REPO_ROOT / "experiments" / "ex2" / "config.yaml"
    if not path.is_file():
        suite.check("experiments/ex2/config.yaml exists", False, "missing")
        return
    merged = load_config(path)
    cfg = Settings(config=merged)
    suite.check("ex2 names its dataset", cfg.DATASET_NAME == "N-MNIST", str(cfg.DATASET_NAME))
    suite.check("ex2 makes sinabs leak-free", cfg.NEURON["sinabs"]["tau_mem"] == float("inf"),
                str(cfg.NEURON["sinabs"]["tau_mem"]))
    # ex2 is "each framework OUT OF THE BOX", so it varies all four neurons, not just
    # sinabs -- that is the experiment. Asserted rather than assumed, because an ex2
    # that only moved one framework would silently be measuring something else.
    base_neuron = load_base()["neuron"]
    varied = [fw for fw in base_neuron if cfg.NEURON[fw] != base_neuron[fw]]
    suite.check("ex2 varies every framework's neuron, not just one",
                len(varied) == len(base_neuron), f"varied: {sorted(varied)}")


# ---------------------------------------------------------------------------
# 4. Settings / WorkflowSettings construction
# ---------------------------------------------------------------------------
def test_experiment_configs_live_in_their_experiment_folder() -> None:
    """The layout rule: config.yaml sits beside the README of the experiment it
    describes, so the two cannot drift apart and a folder is self-contained. There is
    no separate config/ tree."""
    experiments = REPO_ROOT / "experiments"
    suite.check("no separate config/ directory at the repo root",
                not (REPO_ROOT / "config").exists())
    if not experiments.is_dir():
        suite.check("experiments/ exists", False, "missing")
        return
    for folder in sorted(p for p in experiments.iterdir() if p.is_dir()):
        config = folder / "config.yaml"
        if config.is_file():
            suite.check(f"{folder.name}/config.yaml parses", isinstance(load_config(config), dict))


def _git_ignored(paths: list[str]) -> set[str] | None:
    """Which of `paths` git would ignore. None when git cannot answer (no git on PATH,
    or the tree was unpacked from a zip rather than cloned).

    NUL-separated bytes, not text mode: on Windows `text=True` rewrites every "\\n" the
    pipe carries into "\\r\\n", git takes the "\\r" as part of the pathname, and then
    nothing matches. -z also stops git quoting paths it finds unusual.
    """
    import subprocess

    try:
        done = subprocess.run(
            ["git", "check-ignore", "-z", "--stdin"], cwd=REPO_ROOT,
            input=b"\0".join(p.encode("utf-8") for p in paths),
            capture_output=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode not in (0, 1):  # 128 = not a git repository
        return None
    return {p for p in done.stdout.decode("utf-8").split("\0") if p}


def test_generated_experiment_output_is_ignored() -> None:
    """What a RUN writes is generated; what DESCRIBES the experiment is source.

    Asked of git itself rather than of the .gitignore text. A pattern can read correctly
    and still not do the job: `experiments/*/*/` looks equivalent to the rule actually
    used but excludes the parent DIRECTORY, and git cannot re-include a file underneath
    an excluded directory -- so the .gitkeep exception would silently stop working and
    a fresh clone would come without the folders. Only check-ignore knows.
    """
    run = "20260831_norse_seed0"
    generated = [
        f"experiments/ex2/results/runs.csv",          # the append-only schema
        f"experiments/ex2/results/{run}/test.csv",    # per-run CSVs
        f"experiments/ex2/results/runs/{run}.json",
        "experiments/ex2/plots/EQ_poisson.png",       # equivalence figures
        f"experiments/ex2/plots/{run}/loss.png",      # per-run diagnostics
        "experiments/ex2/figures/F1_accuracy.png",    # make_plots.py comparisons
        "experiments/ex2/equivalence/EQ_poisson.png",
        "experiments/ex9/some_future_folder/x.npz",   # a subfolder nobody has invented
    ]
    source = [
        "experiments/ex2/config.yaml",
        "experiments/ex2/README.md",
        "experiments/ex2/ex2_design.md",
        "experiments/ex2/plots/.gitkeep",             # holds the folder in a fresh clone
        "experiments/ex2/results/.gitkeep",
    ]
    ignored = _git_ignored(generated + source)
    if ignored is None:
        suite.check("git cannot answer here -- .gitignore left unchecked", True, "skipped")
        return
    for path in generated:
        suite.check(f"generated, so ignored: {path}", path in ignored)
    for path in source:
        suite.check(f"source, so tracked: {path}", path not in ignored)


def test_no_argument_construction_still_works() -> None:
    """The colleague's `python learning/main.py` path."""
    cfg, wf = Settings(), WorkflowSettings()
    suite.check("Settings() loads with no arguments", cfg.FRAMEWORK is not None)
    suite.check("WorkflowSettings() loads with no arguments", wf.N_TIME_BINS > 0)
    suite.check("no overlay recorded", cfg.overlay_path is None)


def test_config_and_overlay_are_mutually_exclusive() -> None:
    suite.expect_raises("Settings refuses both config and overlay", ValueError,
                        lambda: Settings(config={}, overlay="x.yaml"))
    suite.expect_raises("WorkflowSettings refuses both", ValueError,
                        lambda: WorkflowSettings(config={}, overlay="x.yaml"))


# ---------------------------------------------------------------------------
# 5. the CLI
# ---------------------------------------------------------------------------
def test_defaults_reproduce_the_original_behaviour() -> None:
    cfg, wf, info = build(parse([]))
    base = load_base()
    suite.check("no --config means base only", info["config_path"] is None)
    suite.check("no --experiment means unrouted", info["routed"] is False)
    suite.check("framework comes from the config",
                cfg.FRAMEWORK == base["training"]["framework"], cfg.FRAMEWORK)
    suite.check("seed comes from the config", cfg.SEED == base["training"]["seed"],
                str(cfg.SEED))
    suite.check("cache path comes from the config", wf.CACHE_PATH == base["cache"]["path"],
                wf.CACHE_PATH)
    suite.check("no overrides recorded", info["overrides"] == {}, str(info["overrides"]))
    suite.check("output goes to the output: paths",
                str(info["plots_dir"]).replace("\\", "/").endswith("outputs/plots"),
                str(info["plots_dir"]))


def test_cli_overrides_win_over_config() -> None:
    cfg, wf, info = build(parse(["--framework", "sinabs", "--seed", "42",
                                 "--cache-root", "/tmp/xyz"]))
    suite.check("--framework wins", cfg.FRAMEWORK == "sinabs", cfg.FRAMEWORK)
    suite.check("--seed wins", cfg.SEED == 42, str(cfg.SEED))
    suite.check("--cache-root wins", wf.CACHE_PATH == "/tmp/xyz", wf.CACHE_PATH)
    suite.check("every override is recorded for the banner",
                set(info["overrides"]) == {"framework", "seed", "cache_root"},
                str(info["overrides"]))


def test_cli_beats_overlay_which_beats_base() -> None:
    """The full precedence chain, in one test."""
    with tempfile.TemporaryDirectory() as tmp:
        overlay = write_overlay(Path(tmp), "ov.yaml", "training:\n  framework: norse\n")
        base_framework = load_base()["training"]["framework"]
        suite.check("the overlay differs from the base, so the test is not vacuous",
                    base_framework != "norse", base_framework)

        cfg, _, _ = build(parse(["--config", str(overlay)]))
        suite.check("overlay beats base", cfg.FRAMEWORK == "norse", cfg.FRAMEWORK)

        cfg, _, _ = build(parse(["--config", str(overlay), "--framework", "sinabs"]))
        suite.check("CLI beats overlay", cfg.FRAMEWORK == "sinabs", cfg.FRAMEWORK)


def test_experiment_routes_output() -> None:
    cfg, _, info = build(parse(["--experiment", "ex7"]))
    suite.check("routed flag set", info["routed"] is True)
    for key, tail in [("results_dir", "ex7/results"),
                      ("equivalence_dir", "ex7/equivalence"),
                      ("plots_dir", "ex7/plots")]:
        suite.check(f"{key} routed into the experiment tree",
                    str(info[key]).replace("\\", "/").endswith(tail), str(info[key]))


def test_results_root_moves_the_tree() -> None:
    _, _, info = build(parse(["--experiment", "ex7", "--results-root", "/drive/runs"]))
    suite.check("--results-root relocates the tree",
                str(info["results_dir"]).replace("\\", "/") == "/drive/runs/ex7/results",
                str(info["results_dir"]))


def test_results_root_without_experiment_raises() -> None:
    """The no-silent-no-op rule. Passing this on Colab and finding results in ./outputs
    after the runtime died is exactly the failure this prevents."""
    suite.expect_raises("--results-root without --experiment raises", CliError,
                        lambda: build(parse(["--results-root", "/drive/runs"])),
                        must_mention=["--experiment"])


def test_unknown_framework_is_rejected_by_argparse() -> None:
    try:
        parse(["--framework", "brian2"])
    except SystemExit:
        suite.check("argparse rejects an unknown --framework", True)
    else:
        suite.check("argparse rejects an unknown --framework", False, "accepted it")


def test_optional_argument_groups() -> None:
    """Each script asks only for the flags it can act on."""
    args = parse(["--config", "x"], framework=False, seed=False)
    suite.check("framework can be omitted from a parser", not hasattr(args, "framework"))
    suite.check("seed can be omitted from a parser", not hasattr(args, "seed"))
    args = parse([], results_root=False, cache_root=False)
    suite.check("roots can be omitted from a parser", not hasattr(args, "results_root"))


def test_output_dirs_directly() -> None:
    cfg = Settings()
    _, _, _, routed = output_dirs(None, None, cfg)
    suite.check("no experiment means not routed", routed is False)
    results, equivalence, plots, routed = output_dirs("ex1", "runs", cfg)
    suite.check("experiment means routed", routed is True)
    suite.check("results path built from root + experiment",
                str(results).replace("\\", "/") == "runs/ex1/results", str(results))
    suite.check("equivalence path built the same way",
                str(equivalence).replace("\\", "/") == "runs/ex1/equivalence")
    suite.check("plots path built the same way",
                str(plots).replace("\\", "/") == "runs/ex1/plots")


# ---------------------------------------------------------------------------
# 7. fail-loud config: no silent defaults, no silent typos
# ---------------------------------------------------------------------------
def test_a_missing_key_raises_instead_of_defaulting() -> None:
    """The pattern being replaced was conv.get("conv1_out", 12): delete the key from the
    YAML and the code quietly supplied a literal, so the file stopped describing the
    run. Every key is now required."""
    from skeleton.strict import ConfigKeyError

    for section_name, key in [("convolution", "conv1_out"), ("training", "epochs"),
                              ("training", "seed"), ("framing", "n_time_bins"),
                              ("resource_policy", "max_batch_size")]:
        broken = load_base()
        broken[section_name] = {k: v for k, v in broken[section_name].items() if k != key}
        suite.expect_raises(
            f"a missing {section_name}.{key} raises", ConfigKeyError,
            lambda b=broken: (Settings(config=b), WorkflowSettings(config=b)),
            must_mention=[key])


def test_a_missing_section_raises() -> None:
    """An overlay that dropped a whole section would otherwise take every value in it
    from the code."""
    from skeleton.strict import ConfigKeyError

    for section_name in ("convolution", "training", "framing", "resource_policy"):
        broken = {k: v for k, v in load_base().items() if k != section_name}
        suite.expect_raises(
            f"a missing '{section_name}:' section raises", ConfigKeyError,
            lambda b=broken: (Settings(config=b), WorkflowSettings(config=b)),
            must_mention=[section_name])


def test_a_quoted_number_is_not_silently_accepted() -> None:
    """`epochs: "5"` is a string. int() would swallow it; the type check does not --
    the same strictness the neuron block has always had."""
    from skeleton.strict import ConfigKeyError

    broken = load_base()
    broken["training"] = {**broken["training"], "epochs": "5"}
    suite.expect_raises("a quoted epochs raises", ConfigKeyError,
                        lambda: Settings(config=broken), must_mention=["unquote"])

    broken = load_base()
    broken["training"] = {**broken["training"], "use_amp": "true"}
    suite.expect_raises("a quoted boolean raises", ConfigKeyError,
                        lambda: Settings(config=broken), must_mention=["true or false"])


def test_null_is_a_stated_choice_but_the_key_must_exist() -> None:
    """null has a documented meaning for a few keys (no denoising, duration unknown, ask
    for the dataset). An ABSENT key does not -- that distinction is the point."""
    from skeleton.strict import ConfigKeyError

    ok = load_base()
    ok["framing"] = {**ok["framing"], "denoise_filter_time_us": None}
    suite.check("framing.denoise_filter_time_us: null is accepted",
                WorkflowSettings(config=ok).DENOISE_FILTER_TIME_US is None)

    broken = load_base()
    broken["framing"] = {k: v for k, v in broken["framing"].items()
                         if k != "denoise_filter_time_us"}
    suite.expect_raises("but removing the key entirely raises", ConfigKeyError,
                        lambda: WorkflowSettings(config=broken),
                        must_mention=["denoise_filter_time_us"])


def test_a_misspelled_overlay_key_raises_with_a_suggestion() -> None:
    """The failure this whole change exists for. MEASURED before the fix: `conv1_ou: 64`
    left CONV1_OUT at 12 and `n_time_bin: 40` left N_TIME_BINS at 16, with no error --
    the run reported success having ignored the thing the experiment was about."""
    with tempfile.TemporaryDirectory() as tmp:
        overlay = write_overlay(Path(tmp), "typo.yaml",
                                "convolution:\n  conv1_ou: 64\n"
                                "framing:\n  n_time_bin: 40\n")
        suite.expect_raises("a misspelled overlay key raises", ConfigError,
                            lambda: load_config(overlay),
                            must_mention=["conv1_ou", "n_time_bin", "conv1_out"])


def test_a_correct_overlay_key_is_accepted() -> None:
    """The check must not block real overrides -- including a deeply nested one."""
    with tempfile.TemporaryDirectory() as tmp:
        overlay = write_overlay(Path(tmp), "fine.yaml",
                                "convolution:\n  conv1_out: 64\n"
                                "framing:\n  n_time_bins: 40\n"
                                "training:\n  optimizer:\n    lr: 0.01\n")
        merged = load_config(overlay)
        suite.check("conv1_out override applied", Settings(config=merged).CONV1_OUT == 64)
        suite.check("n_time_bins override applied",
                    WorkflowSettings(config=merged).N_TIME_BINS == 40)
        suite.check("a nested optimizer.lr override applied",
                    Settings(config=merged).LEARNING_RATE == 0.01)


def test_the_shipped_experiment_overlays_all_pass_the_key_check() -> None:
    """A guard against this check being stricter than the configs it has to accept."""
    shipped = sorted(REPO_ROOT.glob("experiments/*/config.yaml"))
    suite.check("there is at least one experiment overlay to check", bool(shipped))
    for path in shipped:
        try:
            load_config(path)
            ok, detail = True, ""
        except ConfigError as exc:
            ok, detail = False, str(exc)
        suite.check(f"{path.parent.name}/config.yaml passes the key check", ok, detail)


def test_dataset_owned_shape_refuses_to_guess() -> None:
    """sensor_h/sensor_w/in_channels/num_classes were removed from
    network_architecture.yaml because apply_dataset_shape() overwrote all four on every
    real run. Reading one before a dataset is applied must raise, not hand back a stale
    34 that produces a plausible-looking but wrong network."""
    cfg = Settings()
    for name in ("SENSOR_H", "SENSOR_W", "IN_CHANNELS", "NUM_CLASSES", "FC_IN"):
        suite.expect_raises(f"cfg.{name} raises before a dataset is applied",
                            AttributeError, lambda n=name: getattr(cfg, n),
                            must_mention=["apply_dataset_shape"])

    cfg.apply_dataset_shape(sensor_h=128, sensor_w=128, in_channels=2, num_classes=11)
    suite.check("after apply_dataset_shape the sensor is set", cfg.SENSOR_H == 128)
    suite.check("classes come from the registry value", cfg.NUM_CLASSES == 11)
    suite.check("FC_IN is recomputed for the new shape", cfg.FC_IN == 26912, str(cfg.FC_IN))

    base = load_base()
    for key in ("sensor_h", "sensor_w", "in_channels", "num_classes"):
        suite.check(f"convolution.{key} is gone from the base config",
                    key not in base["convolution"])


def test_display_survives_an_unset_shape() -> None:
    """A REPORT must never be the thing that stops a run: display() is called before the
    dataset is known in some paths, and it guards every dataset-owned value."""
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        Settings().display()
    text = buffer.getvalue()
    suite.check("display() does not raise with no dataset applied", "ARCHITECTURE" in text)
    suite.check("and says the sensor is not known yet", "not set yet" in text, text[:400])


# ---------------------------------------------------------------------------
# 6. config hash and banner
# ---------------------------------------------------------------------------
def test_config_hash_tracks_content() -> None:
    base = load_base()
    suite.check("the hash is stable for identical content",
                config_hash(base) == config_hash(load_base()))
    changed = deep_merge(base, {"training": {"epochs": base["training"]["epochs"] + 1}})
    suite.check("a changed value changes the hash", config_hash(base) != config_hash(changed))
    suite.check("the hash is short enough to print", len(config_hash(base)) == 12)


def test_banner_states_what_decides_the_run() -> None:
    cfg, _, info = build(parse(["--framework", "norse", "--seed", "5", "--experiment", "ex3"]))
    text = run_banner("main.py", cfg, info)
    for token in ["norse", "5", "ex3", info["config_hash"]]:
        suite.check(f"banner states {token!r}", token in text)
    suite.check("banner names the cli overrides", "cli overrides" in text)

    cfg, _, info = build(parse([]))
    text = run_banner("main.py", cfg, info)
    suite.check("banner says when no overlay was used", "base only" in text)
    suite.check("banner says when the dataset will be prompted", "will prompt" in text)


def test_banner_omits_rows_a_script_does_not_use() -> None:
    """A banner that reports a framework, a seed and a dataset the run never touched is
    worse than one that stays quiet -- the reader cannot tell the values that were used
    from the ones that were merely present in the config. equivalence_check builds ALL
    FOUR frameworks, the poisson pattern carries its own fixed seed, and no dataset is
    ever loaded, so it drops all three."""
    cfg, _, info = build(parse(["--experiment", "ex2"]))

    full = run_banner("x.py", cfg, info, writes_results=False)
    for label in ("framework", "seed", "dataset"):
        suite.check(f"by default the banner still prints {label}", f"  {label:<14}" in full)

    trimmed = run_banner("x.py", cfg, info, writes_results=False,
                         omit=("framework", "seed", "dataset"))
    for label in ("framework", "seed", "dataset"):
        suite.check(f"omit drops the {label} row", f"  {label:<14}" not in trimmed)
    suite.check("what identifies the run survives -- config hash",
                info["config_hash"] in trimmed)
    suite.check("the experiment survives", "ex2" in trimmed)
    suite.check("device survives: make_cfg really does pin it", "device" in trimmed)


# ---------------------------------------------------------------------------
# 7. dataset selection (D17)
# ---------------------------------------------------------------------------
def test_dataset_defaults_to_prompt() -> None:
    suite.check("dataset.name is null in the shipped config",
                load_base()["dataset"]["name"] is None)
    suite.check("Settings leaves DATASET_NAME as None", Settings().DATASET_NAME is None)


def test_dataset_name_from_config_skips_the_prompt() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        overlay = write_overlay(Path(tmp), "ov.yaml", "dataset:\n  name: DVS128 Gesture\n")
        cfg = Settings(config=load_config(overlay))
        suite.check("dataset.name reaches Settings",
                    cfg.DATASET_NAME == "DVS128 Gesture", str(cfg.DATASET_NAME))


def test_lookup_accepts_the_forms_people_actually_write() -> None:
    for written, expected in [
        ("N-MNIST", "N-MNIST"), ("n-mnist", "N-MNIST"), ("nmnist", "N-MNIST"),
        ("N MNIST", "N-MNIST"), ("n_mnist", "N-MNIST"),
        ("DVS128 Gesture", "DVS128 Gesture"), ("dvs128_gesture", "DVS128 Gesture"),
        ("1", "N-MNIST"), ("4", "DVS128 Gesture"),
    ]:
        suite.check(f"lookup {written!r} -> {expected}",
                    lookup_dataset(written)["name"] == expected,
                    lookup_dataset(written)["name"])


def test_every_registry_entry_is_findable_by_its_own_name() -> None:
    for key, entry in DATASET_REGISTRY.items():
        suite.check(f"registry entry {key} findable by name",
                    lookup_dataset(entry["name"])["name"] == entry["name"])
        suite.check(f"registry entry {key} findable by number",
                    lookup_dataset(key)["name"] == entry["name"])


def test_typo_raises_with_a_suggestion() -> None:
    """The whole point of D17: a typo must fail at startup, not download the wrong
    dataset or silently fall back to N-MNIST."""
    for typo, expected_hint in [
        ("N-MNSIT", "N-MNIST"), ("nmnst", "N-MNIST"), ("dvs-gesture", "DVS128 Gesture"),
    ]:
        suite.expect_raises(f"typo {typo!r} raises and suggests {expected_hint}",
                            UnknownDataset, lambda t=typo: lookup_dataset(t),
                            must_mention=[typo, expected_hint])


def test_unrelated_name_raises_without_a_bogus_suggestion() -> None:
    suite.expect_raises("an unrelated name raises", UnknownDataset,
                        lambda: lookup_dataset("cifar10"), must_mention=["cifar10"])
    suite.expect_raises("an out-of-range number raises", UnknownDataset,
                        lambda: lookup_dataset("99"), must_mention=["99"])
    suite.expect_raises("an empty-ish name raises", UnknownDataset,
                        lambda: lookup_dataset("   x   "))


def test_error_lists_the_available_datasets() -> None:
    try:
        lookup_dataset("nope")
    except UnknownDataset as error:
        message = str(error)
        missing = [e["name"] for e in DATASET_REGISTRY.values() if e["name"] not in message]
        suite.check("the error lists every available dataset", not missing,
                    f"missing {missing}")


def test_normalise() -> None:
    suite.check("normalise folds case, spaces, hyphens and underscores",
                len({normalise_dataset_name(x)
                     for x in ["N-MNIST", "n mnist", "N_MNIST", "nmnist"]}) == 1)


def main() -> int:
    return suite.run([
        test_deep_merge,
        test_base_loads_and_covers_every_section,
        test_no_section_collisions_in_the_shipped_files,
        test_base_files_are_all_read,
        test_overlay_overrides_only_what_it_names,
        test_overlay_reaches_every_base_file,
        test_extends_chain,
        test_circular_extends_is_caught,
        test_typo_in_an_overlay_section_raises,
        test_missing_overlay_file_raises,
        test_shipped_ex2_overlay_is_valid,
        test_experiment_configs_live_in_their_experiment_folder,
        test_generated_experiment_output_is_ignored,
        test_a_missing_key_raises_instead_of_defaulting,
        test_a_missing_section_raises,
        test_a_quoted_number_is_not_silently_accepted,
        test_null_is_a_stated_choice_but_the_key_must_exist,
        test_a_misspelled_overlay_key_raises_with_a_suggestion,
        test_a_correct_overlay_key_is_accepted,
        test_the_shipped_experiment_overlays_all_pass_the_key_check,
        test_dataset_owned_shape_refuses_to_guess,
        test_display_survives_an_unset_shape,
        test_no_argument_construction_still_works,
        test_config_and_overlay_are_mutually_exclusive,
        test_defaults_reproduce_the_original_behaviour,
        test_cli_overrides_win_over_config,
        test_cli_beats_overlay_which_beats_base,
        test_experiment_routes_output,
        test_results_root_moves_the_tree,
        test_results_root_without_experiment_raises,
        test_unknown_framework_is_rejected_by_argparse,
        test_optional_argument_groups,
        test_output_dirs_directly,
        test_config_hash_tracks_content,
        test_banner_states_what_decides_the_run,
        test_banner_omits_rows_a_script_does_not_use,
        test_dataset_defaults_to_prompt,
        test_dataset_name_from_config_skips_the_prompt,
        test_lookup_accepts_the_forms_people_actually_write,
        test_every_registry_entry_is_findable_by_its_own_name,
        test_typo_raises_with_a_suggestion,
        test_unrelated_name_raises_without_a_bogus_suggestion,
        test_error_lists_the_available_datasets,
        test_normalise,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
