"""Draw every figure for one experiment, straight from its result CSVs.

Ported from the SNNs_2 comparison pipeline. Figures are never edited by hand: everything
is regenerated from `runs.csv` / `epochs.csv` / `layers.csv`, so a figure can always be
traced back to the rows that produced it, and re-running after another seed arrives is a
one-liner.

    python make_plots.py --experiment ex1
    python make_plots.py --experiment ex1 --formats png
    python make_plots.py --results-dir experiments/ex1/results --out-dir /tmp/figs

Reads ONLY the three CSVs, so it needs no GPU, no dataset and no model -- run it on a
laptop against results copied off Colab.

Self-contained on purpose: it does not import skeleton.cli, because that resolves output
directories from a live Settings object and this script has no run to configure. Paths are
resolved from --experiment / --results-dir alone.

SEED COUNT: several figures compare across seeds (F2's within-seed slopegraph, F6's
effect-vs-noise). With a single seed those cannot say anything, so they are SKIPPED with a
note rather than drawn misleadingly. One seed per framework is a perfectly valid run; you
just get the F1/F3/F4/F5 families, and the seed-comparison families appear once a second
seed exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from plots import apply_rcparams, load
from plots import figures as F
from plots.style import set_formats

DEFAULT_RESULTS_ROOT = "experiments"

# (family, function, needs_multiple_seeds). Order is the order they are written.
FAMILIES = [
    ("F0  fairness evidence",        F.f0_start_state,      False),
    ("F1  per-metric distribution",  F.f1_overview,         False),
    ("F1  per-metric distribution",  F.f1_speed,            False),
    ("F1  per-metric distribution",  F.f1_energy,           False),
    ("F2  paired within-seed views", F.f2_slopegraph,       True),
    ("F2  paired within-seed views", F.f2_differences,      True),
    ("F3  profile and trade-offs",   F.f3_profile,          False),
    ("F3  profile and trade-offs",   F.f3_tradeoff,         False),
    ("F4  training dynamics",        F.f4_learning_curves,  False),
    ("F4  training dynamics",        F.f4_epoch_time,       False),
    ("F4  training dynamics",        F.f4_sparsification,   False),
    ("F4  training dynamics",        F.f4_loss,             False),
    ("F5  structure",                F.f5_layer_activity,   False),
    ("F5  structure",                F.f5_spike_budget,     False),
    ("F6  effect vs noise",          F.f6_effect_vs_noise,  True),
    ("F6  effect vs noise",          F.f6_idle_baseline,    False),
    ("F6  effect vs noise",          F.f6_sensor,           False),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None, metavar="exN",
                         help=f"read <results-root>/exN/results and write "
                              f"<results-root>/exN/figures. Default root: "
                              f"{DEFAULT_RESULTS_ROOT}")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT,
                         help="root holding the per-experiment folders")
    parser.add_argument("--results-dir", default=None,
                         help="read the three CSVs from here instead (overrides "
                              "--experiment)")
    parser.add_argument("--out-dir", default=None,
                         help="write figures here instead of <results-dir>/../figures")
    parser.add_argument("--formats", default="png,pdf",
                         help="comma-separated output formats, e.g. png or png,pdf")
    parser.add_argument("--condition", default="framework",
                         help="the column that separates the things being compared. "
                              "'framework' for a framework comparison; point it at "
                              "another column to compare variants or datasets instead.")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.experiment:
        results_dir = Path(args.results_root) / args.experiment / "results"
    else:
        raise SystemExit(
            "give --experiment exN, or --results-dir pointing at the folder holding "
            "runs.csv / epochs.csv / layers.csv"
        )
    out_dir = Path(args.out_dir) if args.out_dir else results_dir.parent / "figures"
    return results_dir, out_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir, out_dir = resolve_paths(args)
    set_formats(tuple(f.strip() for f in args.formats.split(",") if f.strip()))
    apply_rcparams()

    try:
        results = load(results_dir, condition=args.condition)
    except FileNotFoundError as exc:
        print(f"[PLOTS] {exc}")
        print(f"[PLOTS] nothing to draw. Run a training run first -- it writes "
              f"runs.csv / epochs.csv / layers.csv into {results_dir}")
        return 1

    conditions = list(results.conditions)
    seeds = list(results.blocks)
    print(f"[PLOTS] {results_dir}")
    print(f"[PLOTS] {len(results.runs)} run(s)  |  {args.condition}: "
          f"{', '.join(str(c) for c in conditions)}  |  seed(s): "
          f"{', '.join(str(s) for s in seeds)}")

    single_seed = len(seeds) < 2
    if single_seed:
        print("[PLOTS] one seed only -- the seed-comparison families (F2, F6 effect vs "
              "noise) are skipped, since they have nothing to compare across.")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped: list[str] = []
    failed: list[str] = []

    for family, function, needs_seeds in FAMILIES:
        name = function.__name__
        if needs_seeds and single_seed:
            skipped.append(name)
            continue
        try:
            paths = function(results, out_dir) or []
            written.extend(paths)
            for path in paths:
                print(f"  {family:32} {path.name}")
        except Exception as exc:
            # One unusable figure must not cost the other sixteen -- a missing optional
            # metric (energy on a CPU run, layers.csv from a run that measured none) is
            # the usual cause.
            failed.append(f"{name}: {type(exc).__name__}: {exc}")

    print(f"\n[PLOTS] {len(written)} file(s) -> {out_dir}")
    if skipped:
        print(f"[PLOTS] skipped (needs >= 2 seeds): {', '.join(skipped)}")
    for message in failed:
        print(f"[PLOTS] !! {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
