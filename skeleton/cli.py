"""One command-line surface, shared by every entry point.

TWO WAYS TO RUN, BOTH SUPPORTED
-------------------------------
    python learning/main.py
        No flags. The three files in configuration/ are the whole config, the dataset
        is chosen at the prompt, and output goes to ./outputs -- exactly as this
        pipeline has always worked.

    python learning/main.py --config experiments/ex2/config.yaml --experiment ex2 --framework sinabs
        One overlay config per experiment, output routed into experiments/ex2/.

THE DIVIDING LINE
-----------------
    the CONFIG describes the experiment   -- neuron, architecture, framing, epochs
    the COMMAND LINE describes this run   -- which folder, which machine, which
                                             framework, which seed

That split is what makes the same config file run unchanged on a laptop and on Colab.
A path inside a config file would mean editing the science to change the machine.

PRECEDENCE
    CLI argument  >  --config overlay  >  the three base files  >  code default

NO SILENTLY-IGNORED FLAGS
-------------------------
`--results-root` without `--experiment` is refused rather than ignored. Passing
`--results-root /content/drive/MyDrive/runs`, seeing a green run, and then finding the
results went to ./outputs and died with the Colab runtime is exactly the kind of quiet
failure this pipeline has been getting rid of.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from skeleton.config_loader import ConfigError, load_config
from skeleton.snn_config import FW_TO_CFG_KEY, Settings
from skeleton.workflow_config import WorkflowSettings

# Where output lands when no --experiment is given: this pipeline's own existing
# behaviour, driven by the `output:` block in SNN_module.yaml.
DEFAULT_RESULTS_ROOT = "experiments"


class CliError(Exception):
    """The combination of flags given cannot be honoured."""


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    framework: bool = True,
    seed: bool = True,
    experiment: bool = True,
    results_root: bool = True,
    cache_root: bool = True,
) -> argparse.ArgumentParser:
    """Attach the shared flags. Each script asks only for the ones it can act on.

    The two roots are separate switches, not one, because a script can legitimately
    need one and not the other. `equivalence_check.py` writes FIGURES but reads no
    dataset: it needs `--results-root` (so those figures can land on mounted Drive
    instead of a Colab runtime that is about to disappear) and has no use for
    `--cache-root`. Collapsing both into a single `roots` flag is what previously left
    it unable to write anywhere durable.
    """
    parser.add_argument(
        "--config", default=None, metavar="PATH",
        help="experiment overlay merged over the three files in configuration/. "
             "States only what differs; everything else is inherited. Omit to run "
             "on the base config alone.",
    )
    if framework:
        parser.add_argument(
            "--framework", default=None, choices=sorted(FW_TO_CFG_KEY),
            help="overrides training.framework from the config",
        )
    if seed:
        parser.add_argument(
            "--seed", type=int, default=None, metavar="N",
            help="overrides training.seed (default 0). Fixes weight init and batch "
                 "order, so a rerun reproduces.",
        )
    if experiment:
        parser.add_argument(
            "--experiment", default=None, metavar="exN",
            help="route output to <results-root>/exN/{results,equivalence,plots}. "
                 "Without it, output goes to the output: paths in SNN_module.yaml, "
                 "as it always has.",
        )
    if results_root:
        parser.add_argument(
            "--results-root", default=None, metavar="PATH",
            help=f"where the experiment tree lives (default {DEFAULT_RESULTS_ROOT}). "
                 "Point it at mounted Drive on Colab so output survives the runtime. "
                 "Requires --experiment.",
        )
    if cache_root:
        parser.add_argument(
            "--cache-root", default=None, metavar="PATH",
            help="overrides cache.path for the frame cache. Useful where the default "
                 "location is small or ephemeral.",
        )
    return parser


def config_hash(config: dict[str, Any]) -> str:
    """Short digest of the fully-merged config, so a results row can prove which
    settings produced it without storing the whole thing."""
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def output_dirs(experiment: str | None, results_root: str | None, cfg: Settings):
    """(results_dir, equivalence_dir, plots_dir, routed).

    routed=False means no --experiment was given, so these are the pipeline's own
    `output:` paths and nothing about the layout has changed.
    """
    if experiment is None:
        return (Path(cfg.DATA_DIR), Path(cfg.OUTPUT_DIR), Path(cfg.PLOT_DIR), False)
    base = Path(results_root or DEFAULT_RESULTS_ROOT) / experiment
    return (base / "results", base / "equivalence", base / "plots", True)


def build(args: argparse.Namespace) -> tuple[Settings, WorkflowSettings, dict[str, Any]]:
    """Turn parsed arguments into (Settings, WorkflowSettings, run info).

    One place, so every script resolves the config and applies overrides identically.
    """
    experiment = getattr(args, "experiment", None)
    results_root = getattr(args, "results_root", None)
    cache_root = getattr(args, "cache_root", None)

    if results_root is not None and experiment is None:
        raise CliError(
            "--results-root only means something together with --experiment. Without "
            "--experiment, output goes to the output: paths in SNN_module.yaml and "
            "--results-root would be silently ignored. Add --experiment exN, or drop "
            "--results-root."
        )

    config = load_config(args.config)
    cfg = Settings(config=config)
    wf = WorkflowSettings(config=config)

    # ---- CLI overrides, applied after the config so they always win -------------
    overrides: dict[str, Any] = {}
    framework = getattr(args, "framework", None)
    if framework is not None:
        cfg.FRAMEWORK = framework
        overrides["framework"] = framework
    seed = getattr(args, "seed", None)
    if seed is not None:
        cfg.SEED = seed
        overrides["seed"] = seed
    if cache_root is not None:
        wf.CACHE_PATH = cache_root
        overrides["cache_root"] = cache_root

    results_dir, equivalence_dir, plots_dir, routed = output_dirs(experiment, results_root, cfg)
    info = {
        "config_path": args.config,
        "config_hash": config_hash(config),
        "experiment": experiment,
        "results_dir": results_dir,
        "equivalence_dir": equivalence_dir,
        "plots_dir": plots_dir,
        "routed": routed,
        "overrides": overrides,
    }
    return cfg, wf, info


def run_banner(script: str, cfg: Settings, info: dict[str, Any], *,
               extra: dict[str, Any] | None = None, writes_results: bool = True) -> str:
    """The identity block every entry point prints before doing anything.

    One shared formatter so every script announces the same facts the same way. The
    point is that a scrolled-back terminal, or a snippet pasted into a lab notebook,
    still says WHICH config and WHICH experiment produced what follows -- the two
    things that decide whether a number means anything.
    """
    lines = ["=" * 74, script]

    def row(label: str, value: Any) -> None:
        lines.append(f"  {label:<14}{value}")

    row("config", f"{info['config_path'] or 'base only (configuration/)'}"
                  f"   hash {info['config_hash']}")
    if info["experiment"]:
        row("experiment", info["experiment"])
    elif writes_results:
        row("experiment", "none  -> output: paths from SNN_module.yaml")
    row("framework", cfg.FRAMEWORK)
    row("seed", cfg.SEED)
    row("dataset", cfg.DATASET_NAME or "not set -- will prompt")
    row("device", cfg.DEVICE)
    if writes_results:
        row("writing to", info["results_dir"])
    if info["overrides"]:
        row("cli overrides", ", ".join(f"{k}={v}" for k, v in info["overrides"].items()))
    for label, value in (extra or {}).items():
        row(label, value)

    lines.append("=" * 74)
    return "\n".join(lines)


__all__ = [
    "CliError", "ConfigError", "DEFAULT_RESULTS_ROOT",
    "add_common_args", "build", "config_hash", "output_dirs", "run_banner",
]
