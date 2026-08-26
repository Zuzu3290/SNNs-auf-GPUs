"""Loading the three result CSVs and describing what each metric means.

Two jobs:

1. Read `runs.csv` / `epochs.csv` / `layers.csv` into dataframes, adding the few
   derived columns that are cheaper to compute here than to store (mean dynamic
   power, energy in kJ, spikes per neuron per inference).
2. Hold the METRICS registry -- the single place that knows a metric's label, unit,
   and whether higher or lower is better. Every figure reads from it, so adding a
   metric to the plots is one entry, not an edit in six files.

Nothing here is experiment-specific. `condition` is the generic grouping column:
normally `framework`, but a future experiment comparing variants or datasets can
point it elsewhere without touching any figure code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CSV_FILES = ("runs.csv", "epochs.csv", "layers.csv")


@dataclass(frozen=True)
class Metric:
    """One measurable quantity, and how to present it."""

    column: str
    label: str
    unit: str
    better: str  # "up", "down", or "neutral"
    group: str
    plain: str  # one line, for a non-specialist
    decimals: int = 2

    def fmt(self, value: float) -> str:
        return f"{value:.{self.decimals}f}"


# The registry. Order here is the order panels appear in the overview figures.
METRICS: dict[str, Metric] = {
    "accuracy": Metric(
        "test_accuracy_pct",
        "Test accuracy",
        "%",
        "up",
        "quality",
        "Share of 10,000 unseen digits labelled correctly.",
    ),
    "train_time": Metric(
        "train_time_s",
        "Training time",
        "s",
        "down",
        "speed",
        "Wall-clock time to complete all training epochs.",
        decimals=1,
    ),
    "epoch_time": Metric(
        "train_time_per_epoch_s",
        "Time per epoch",
        "s",
        "down",
        "speed",
        "Training time divided by the number of epochs.",
        decimals=1,
    ),
    "throughput": Metric(
        "inference_throughput_samples_per_s",
        "Throughput",
        "samples/s",
        "up",
        "speed",
        "How many digits per second when fed in large batches (the factory question).",
        decimals=0,
    ),
    "latency": Metric(
        "inference_latency_bs1_ms",
        "Latency, median",
        "ms",
        "down",
        "speed",
        "Time to answer ONE digit, middle value (the queue question).",
    ),
    "latency_p90": Metric(
        "inference_latency_bs1_p90_ms",
        "Latency, 90th pct",
        "ms",
        "down",
        "speed",
        "9 of 10 single digits answered faster than this -- catches the slow ones.",
    ),
    "spike_rate": Metric(
        "spike_rate_pct",
        "Spike rate",
        "%",
        "down",
        "activity",
        "Share of firing opportunities a neuron used. Lower = same job, less activity.",
        decimals=3,
    ),
    "spikes_per_neuron": Metric(
        "spikes_per_neuron_per_inference",
        "Spikes / neuron / digit",
        "spikes",
        "down",
        "activity",
        "The same as spike rate, in the unit the SNN literature uses.",
        decimals=3,
    ),
    "memory_train": Metric(
        "peak_memory_train_mb",
        "Peak memory, training",
        "MB",
        "down",
        "memory",
        "Most GPU memory held at any one instant while training.",
        decimals=1,
    ),
    "memory_reserved": Metric(
        "peak_reserved_train_mb",
        "Reserved memory, training",
        "MB",
        "down",
        "memory",
        "Memory PyTorch claimed from the driver -- what the card actually gives up.",
        decimals=0,
    ),
    "memory_infer": Metric(
        "peak_memory_infer_mb",
        "Peak memory, inference",
        "MB",
        "down",
        "memory",
        "Same high-water mark, but only answering rather than learning.",
        decimals=1,
    ),
    "energy_total": Metric(
        "train_energy_kj",
        "Energy, total",
        "kJ",
        "down",
        "energy",
        "All electricity the GPU drew during training, idle included.",
    ),
    "energy_dynamic": Metric(
        "train_energy_dynamic_kj",
        "Energy, dynamic",
        "kJ",
        "down",
        "energy",
        "Total minus what idling for the same time would have cost.",
    ),
    "power_dynamic": Metric(
        "mean_dynamic_power_w",
        "Mean dynamic power",
        "W",
        "down",
        "energy",
        "How hard the GPU worked on average, rather than for how long.",
    ),
}

# Metrics that may never feed a ranking or trade-off figure. Energy is recorded but
# deliberately not interpreted: it comes from a GPU-internal sensor, it is largely
# time x power so it partly restates the speed result, and in experiment 1 the
# instrument contradicted itself between seeds.
UNINTERPRETED_GROUPS = frozenset({"energy"})


def interpretable(keys: list[str]) -> list[str]:
    """Filter out metrics not yet allowed to support a conclusion."""
    return [k for k in keys if METRICS[k].group not in UNINTERPRETED_GROUPS]


@dataclass
class Results:
    """One experiment's three tables, plus the grouping column to compare on."""

    runs: pd.DataFrame
    epochs: pd.DataFrame
    layers: pd.DataFrame
    condition: str = "framework"

    @property
    def conditions(self) -> list[str]:
        """Distinct conditions, in the fixed display order where known."""
        from .style import FRAMEWORK_ORDER

        present = list(self.runs[self.condition].unique())
        known = [c for c in FRAMEWORK_ORDER if c in present]
        rest = sorted(c for c in present if c not in known)
        return known + rest

    @property
    def blocks(self) -> list[int]:
        """The paired-comparison blocks. One seed = one Colab session = one block."""
        return sorted(self.runs["seed"].unique())

    def values(self, metric_key: str, condition: str) -> pd.Series:
        """Every run's value for one metric under one condition, ordered by seed."""
        column = METRICS[metric_key].column
        subset = self.runs[self.runs[self.condition] == condition].sort_values("seed")
        return subset[column].dropna()

    def value_at(self, metric_key: str, condition: str, block: int) -> float | None:
        """One run's value -- the (condition, block) cell of the paired design."""
        column = METRICS[metric_key].column
        subset = self.runs[
            (self.runs[self.condition] == condition) & (self.runs["seed"] == block)
        ]
        if subset.empty or pd.isna(subset[column].iloc[0]):
            return None
        return float(subset[column].iloc[0])

    def has(self, metric_key: str) -> bool:
        """Whether this metric exists and holds at least one real value."""
        column = METRICS[metric_key].column
        return column in self.runs.columns and self.runs[column].notna().any()

    def hardware(self) -> str:
        names = sorted(str(x) for x in self.runs["gpu_name"].dropna().unique())
        return " / ".join(names) if names else "unknown GPU"

    def note(self, extra: str = "") -> str:
        """The provenance line stamped on every figure."""
        n_blocks = len(self.blocks)
        seeds = ", ".join(str(b) for b in self.blocks)
        parts = [
            f"n = {n_blocks} seeds (seed {seeds})",
            f"{len(self.runs)} runs",
            self.hardware(),
        ]
        if extra:
            parts.append(extra)
        return " · ".join(parts)


def _derive(runs: pd.DataFrame) -> pd.DataFrame:
    """Add columns that are cheaper to compute than to store.

    Kept out of `runs.csv` on purpose: they are exact functions of columns already
    there, so storing them would create two things that could disagree.
    """
    runs = runs.copy()
    if "train_energy_j" in runs:
        runs["train_energy_kj"] = runs["train_energy_j"] / 1000.0
    if "train_energy_dynamic_j" in runs:
        runs["train_energy_dynamic_kj"] = runs["train_energy_dynamic_j"] / 1000.0
    if {"train_energy_dynamic_j", "train_energy_duration_s"} <= set(runs.columns):
        runs["mean_dynamic_power_w"] = (
            runs["train_energy_dynamic_j"] / runs["train_energy_duration_s"]
        )
    if {"spike_rate_pct", "time_steps"} <= set(runs.columns):
        # A neuron gets one firing opportunity per timestep, so the percentage and
        # the spikes-per-neuron figure are the same measurement in two units.
        runs["spikes_per_neuron_per_inference"] = (
            runs["spike_rate_pct"] / 100.0 * runs["time_steps"]
        )
    return runs


def load(results_dir: Path, condition: str = "framework") -> Results:
    """Read one experiment's results folder."""
    frames = {}
    for filename in CSV_FILES:
        path = results_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        frames[filename] = pd.read_csv(path)

    runs = _derive(frames["runs.csv"])
    if "seed" in runs:
        runs["seed"] = runs["seed"].astype(int)

    return Results(
        runs=runs,
        epochs=frames["epochs.csv"],
        layers=frames["layers.csv"],
        condition=condition,
    )


def summary(results: Results, metric_key: str) -> pd.DataFrame:
    """mean / std / n per condition, in the fixed display order.

    Standard DEVIATION, not standard error. Weissgerber et al. (2015) point out
    that SE = SD/sqrt(n) shrinks with sample size and so visually magnifies
    differences between groups. SD answers the question actually being asked here:
    how much does this number move between runs?
    """
    rows = []
    for condition in results.conditions:
        series = results.values(metric_key, condition)
        rows.append(
            {
                "condition": condition,
                "mean": series.mean(),
                "std": series.std(ddof=1) if len(series) > 1 else 0.0,
                "n": len(series),
                "min": series.min(),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)
