"""Pull Colab results into an experiment folder, MERGING instead of overwriting.

    python collect_results.py --from "G:/My Drive/snn_results/ex2" --experiment ex2
    python collect_results.py --from temp_colab --experiment ex2 --dry-run

The round trip is: Colab writes to Google Drive -> Drive syncs, or you download the
folder once -> this script files everything into experiments/exN/.

WHY MERGE RATHER THAN COPY. The three schema CSVs are append-only ACROSS runs: one
runs.csv holds every framework and every seed of an experiment, which is what makes it
a comparison table. A Colab session that restarts begins a fresh runs.csv containing
only its own rows, and a second Colab account has its own from the start. Copying
either over the local file silently deletes the earlier runs -- no error, and nothing
downstream notices that the comparison is now missing half its arms.

So this concatenates, keyed per file:

    runs.csv     run_id
    epochs.csv   run_id + epoch
    layers.csv   run_id + layer_index

Re-running is therefore always safe: a row already present is skipped, never
duplicated. --dry-run reports exactly what would change and writes nothing.

Ported from the SNNs_2 comparison pipeline, onto this pipeline's output layout -- which
also nests per-run folders under results/<run_id>/ and plots/<run_id>/, so those are
brought across too.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skeleton.cli import DEFAULT_RESULTS_ROOT, output_dirs  # noqa: E402

# The three append-only schema files, and what makes a row unique in each.
ROW_KEYS = {
    "runs.csv":   ("run_id",),
    "epochs.csv": ("run_id", "epoch"),
    "layers.csv": ("run_id", "layer_index"),
}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file() or path.stat().st_size == 0:
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def row_key(row: dict[str, str], filename: str) -> tuple:
    return tuple(row.get(column, "") for column in ROW_KEYS[filename])


def _runs_csv(folder: Path) -> Path:
    """runs.csv, whether the folder is a full experiment tree or a flat download."""
    nested = folder / "results" / "runs.csv"
    return nested if nested.is_file() else folder / "runs.csv"


def guard_experiment_mismatch(source: Path, target: Path, experiment: str,
                              force: bool) -> None:
    """Refuse to file one experiment's runs under another experiment's name.

    The failure this prevents: `--from <drive>/ex2 --experiment ex1` appends ex2's rows
    into ex1's runs.csv. Nothing else catches it. Row keys are run_ids, which carry a
    timestamp but no experiment name, so the rows do not collide -- they silently
    coexist, and every later mean, plot and conclusion is computed over a mixture of
    two experiments.

    The signal is `config_path`. If the incoming rows and the existing rows have NO
    config in common they are different experiments. Overlap is treated as fine,
    because one experiment legitimately spans several configs and must stay mergeable.
    """
    incoming_header, incoming = read_rows(_runs_csv(source))
    existing_header, existing = read_rows(_runs_csv(target))
    if not incoming or not existing:
        return  # nothing to compare against; the schema check covers the rest
    if "config_path" not in (incoming_header or []) or \
       "config_path" not in (existing_header or []):
        return

    incoming_configs = {r["config_path"] for r in incoming if r.get("config_path")}
    existing_configs = {r["config_path"] for r in existing if r.get("config_path")}
    if not incoming_configs or not existing_configs or (incoming_configs & existing_configs):
        return

    message = (
        f"\nREFUSING TO MERGE: this looks like a different experiment.\n\n"
        f"  target  {target}  was produced by: {', '.join(sorted(existing_configs))}\n"
        f"  incoming rows were produced by:    {', '.join(sorted(incoming_configs))}\n\n"
        f"  No config in common, so these are almost certainly two different\n"
        f"  experiments. Merging would file one under the other's name, and\n"
        f"  nothing downstream would ever flag it.\n\n"
        f"  If --experiment is wrong, fix it. If this really is the same experiment\n"
        f"  run from a renamed config, re-run with --force."
    )
    if force:
        print(message.replace("REFUSING TO MERGE", "WARNING (--force given)") + "\n")
        return
    raise SystemExit(message)


def merge_csv(source: Path, target: Path, dry_run: bool) -> str:
    src_header, src_rows = read_rows(source)
    if not src_rows:
        return "nothing to merge"

    dst_header, dst_rows = read_rows(target)
    if dst_header and dst_header != src_header:
        missing = sorted(set(src_header) - set(dst_header))
        extra = sorted(set(dst_header) - set(src_header))
        raise SystemExit(
            f"\nERROR: {target.name} has a different schema on each side, so merging "
            f"would misalign every row.\n"
            f"  incoming has, local does not: {missing or 'none'}\n"
            f"  local has, incoming does not: {extra or 'none'}\n"
            "Written by different code versions. Rename the local file to keep it, "
            "then re-run."
        )

    seen = {row_key(row, source.name) for row in dst_rows}
    added = [row for row in src_rows if row_key(row, source.name) not in seen]

    if not dry_run and added:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=src_header)
            if not dst_rows:
                writer.writeheader()
            writer.writerows(added)

    skipped = len(src_rows) - len(added)
    return f"+{len(added)} new row(s)" + (f", {skipped} already present" if skipped else "")


def copy_files(sources: list[Path], target_dir: Path, dry_run: bool) -> int:
    """Copy files that are not already there at the same size. Never overwrites a
    differing file silently -- same size is treated as same file, which is enough for
    write-once artefacts like a finished PNG or a per-run CSV."""
    copied = 0
    for path in sources:
        destination = target_dir / path.name
        if destination.exists() and destination.stat().st_size == path.stat().st_size:
            continue
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        copied += 1
    return copied


def copy_run_folders(source_root: Path, target_root: Path, dry_run: bool) -> str:
    """The per-run subfolders this pipeline nests under results/ and plots/.

    training_results.csv, batch_metrics.csv, test.csv and the seven diagnostic PNGs all
    carry fixed names, so each run keeps its own folder keyed by run_id. Bringing them
    across one folder at a time keeps that structure intact.
    """
    if not source_root.is_dir():
        return "source folder absent"
    folders = sorted(p for p in source_root.iterdir()
                     if p.is_dir() and p.name != "runs")
    if not folders:
        return "none found"
    total = 0
    for folder in folders:
        total += copy_files(sorted(f for f in folder.iterdir() if f.is_file()),
                            target_root / folder.name, dry_run)
    return f"{len(folders)} run folder(s), {total} file(s) copied"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", required=True, metavar="PATH",
                        help="folder holding the Colab output: a mounted Drive folder, "
                             "or a manual download")
    parser.add_argument("--experiment", required=True, metavar="exN",
                        help="target experiment folder name, e.g. ex2")
    parser.add_argument("--results-root", default=None, metavar="PATH",
                        help=f"where the experiment tree lives (default: "
                             f"{DEFAULT_RESULTS_ROOT}). Use the same value the runs used.")
    parser.add_argument("--force", action="store_true",
                        help="merge even when the incoming rows look like a different "
                             "experiment. Only when you are certain -- see the message "
                             "it overrides.")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen, write nothing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source = Path(args.source)
    if not source.is_dir():
        raise SystemExit(f"source folder not found: {source.resolve()}")

    # The same routing every other script uses, so results land exactly where a local
    # run would have put them. cfg is unused when --experiment is given.
    results_dir, equivalence_dir, plots_dir, _routed = output_dirs(
        args.experiment, args.results_root, cfg=None)

    print("=" * 74)
    print("collect_results.py")
    print(f"  {'experiment':<14}{args.experiment}")
    print(f"  {'reading':<14}{source.resolve()}")
    print(f"  {'writing to':<14}{results_dir.parent.resolve()}")
    print(f"  {'mode':<14}{'DRY RUN (nothing written)' if args.dry_run else 'merging'}")
    print("=" * 74)
    print()

    guard_experiment_mismatch(source, results_dir.parent, args.experiment, args.force)

    # The Colab side may be a full experiment tree or a flat download. Accept both.
    results_src = source / "results" if (source / "results").is_dir() else source
    plots_src = source / "plots" if (source / "plots").is_dir() else source
    equiv_src = source / "equivalence" if (source / "equivalence").is_dir() else plots_src

    print("append-only CSVs")
    for name in ROW_KEYS:
        candidate = results_src / name
        if not candidate.is_file():
            print(f"  {name:<12} not found in source")
            continue
        print(f"  {name:<12} {merge_csv(candidate, results_dir / name, args.dry_run)}")

    runs_src = results_src / "runs"
    per_run = sorted(runs_src.glob("*.json")) if runs_src.is_dir() else []
    print(f"\nper-run JSON  {copy_files(per_run, results_dir / 'runs', args.dry_run)} "
          f"copied of {len(per_run)} found")

    print(f"per-run CSVs  {copy_run_folders(results_src, results_dir, args.dry_run)}")
    print(f"per-run plots {copy_run_folders(plots_src, plots_dir, args.dry_run)}")

    # Equivalence figures: EQ_*.png in this pipeline, equivalence_* in the older one.
    equiv = sorted(p for p in equiv_src.glob("EQ_*") if p.is_file())
    equiv += sorted(p for p in equiv_src.glob("equivalence_*") if p.is_file())
    target = equivalence_dir if equiv_src.name == "equivalence" else plots_dir
    print(f"equivalence   {copy_files(equiv, target, args.dry_run)} copied of "
          f"{len(equiv)} found")

    if not args.dry_run:
        merged = results_dir / "runs.csv"
        _, rows = read_rows(merged)
        print(f"\n{merged} now holds {len(rows)} run(s):")
        for row in rows:
            print(f"  {row.get('framework', '?'):<14} seed {row.get('seed', '?'):<3} "
                  f"acc {row.get('test_accuracy_pct', '?'):<8} {row.get('run_id', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
