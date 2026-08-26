"""The six figure families.

F0 fairness evidence · F1 per-metric distribution · F2 paired within-block views
F3 profile and trade-offs · F4 training dynamics · F5 structure
F6 measurement quality

Every function takes a `Results` and an output directory and returns the paths it
wrote, so `make_plots.py` stays a list of calls.

Each figure carries a plain-language subtitle saying what to look for. These are
read by people outside neuromorphic computing, and Rougier et al. (2014) rule 4 is
that the caption is part of the figure rather than an optional extra.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import METRICS, Results, interpretable, summary
from .primitives import (
    difference_dots,
    dot_interval,
    pairs_of,
    pooled_sd,
    slopegraph,
)
from .style import (
    FAINT_GREY,
    GRID_GREY,
    OKABE_ITO,
    TEXT_GREY,
    better_label,
    identity,
    legend_frameworks,
    provenance,
    save,
    title_block,
    wrap,
)

# Metrics shown in the overview and paired grids. Deliberately not every column:
# one figure, one message.
HEADLINE = [
    "accuracy",
    "train_time",
    "throughput",
    "latency",
    "spike_rate",
    "memory_train",
]


# --- helpers -----------------------------------------------------------------


def _series(results: Results, metric_key: str) -> dict[str, list[float]]:
    return {c: list(results.values(metric_key, c)) for c in results.conditions}


def _getter(results: Results, metric_key: str):
    def value_of(condition: str, block: int):
        return results.value_at(metric_key, condition, block)

    return value_of


def _with_meta(frame: pd.DataFrame, results: Results) -> pd.DataFrame:
    """Attach framework and seed to a per-epoch or per-layer table."""
    keys = results.runs[["run_id", results.condition, "seed"]]
    return frame.merge(keys, on="run_id", how="left")


def _present(results: Results, keys: list[str]) -> list[str]:
    return [k for k in keys if results.has(k)]


# --- F0: fairness evidence ---------------------------------------------------


def f0_start_state(results: Results, out_dir: Path) -> list[Path]:
    """F0.2 -- did every framework start each seed from the same weights?

    A grid of seed x framework, coloured by which starting-weight fingerprint the
    run used. Uniform colour across a row means the frameworks were given identical
    initial weights; a change of colour BETWEEN rows means the seed argument
    actually took effect. Both need to be true, and both are easy to get wrong
    silently, which is why this is a figure and not a footnote.
    """
    if "weight_fingerprint" not in results.runs.columns:
        return []

    conditions, blocks = results.conditions, results.blocks
    fingerprints = {}
    for _, row in results.runs.iterrows():
        fingerprints[(row[results.condition], int(row["seed"]))] = str(
            row["weight_fingerprint"]
        )

    unique = sorted(set(fingerprints.values()))
    palette = [OKABE_ITO["sky_blue"], OKABE_ITO["orange"], OKABE_ITO["bluish_green"]]
    colour_of = {fp: palette[i % len(palette)] for i, fp in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(7.6, 1.15 * len(blocks) + 2.0))

    for row_index, block in enumerate(blocks):
        for col_index, condition in enumerate(conditions):
            fingerprint = fingerprints.get((condition, block))
            if fingerprint is None:
                continue
            ax.add_patch(
                plt.Rectangle(
                    (col_index - 0.46, row_index - 0.42),
                    0.92,
                    0.84,
                    facecolor=colour_of[fingerprint],
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.9,
                )
            )
            ax.text(
                col_index,
                row_index,
                fingerprint[:12],
                ha="center",
                va="center",
                fontsize=8.5,
                family="monospace",
                color="black",
            )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([identity(c).label for c in conditions])
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"seed {b}" for b in blocks])
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.set_ylim(-0.6, len(blocks) - 0.4)
    ax.invert_yaxis()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.annotate(
        "same colour across a row  →  all frameworks started from identical weights\n"
        "different colour between rows  →  changing the seed really did change the start",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 26),
        textcoords="offset points",
        fontsize=8.5,
        color=TEXT_GREY,
    )

    title_block(
        fig,
        "F0.2  Every run's starting weights",
        "The fairness check: a fingerprint (SHA-256) of all 18,254 trainable parameters before training begins.",
    )
    provenance(fig, results.note("fingerprint = sha256 of initial trainable parameters"))
    fig.subplots_adjust(top=0.74, bottom=0.12)
    return save(fig, out_dir, "F0.2_start_state_audit")


# --- F1: per-metric distribution ---------------------------------------------


def _dot_panel(ax, results: Results, metric_key: str, annotate: bool = False,
               compact: bool = True) -> None:
    metric = METRICS[metric_key]
    dot_interval(
        ax,
        results.conditions,
        _series(results, metric_key),
        annotate=annotate,
        decimals=metric.decimals,
    )
    ax.set_ylabel(
        better_label(metric.label, metric.unit, metric.better, compact=compact), fontsize=9
    )
    ax.set_title(
        wrap(metric.plain, 46), fontsize=8.5, fontweight="normal", color=TEXT_GREY, loc="left"
    )
    # Leave room on the right for the mean +/- SD annotation.
    if annotate:
        ax.set_xlim(-0.6, len(results.conditions) - 0.15)


def f1_overview(results: Results, out_dir: Path) -> list[Path]:
    """F1.1 -- every headline metric, every individual run.

    The workhorse. Replaces a summary table: instead of "610 +/- 46 s" the reader
    sees the three runs that produced it and can judge the spread themselves.

    No bar charts. Weissgerber et al. (2015) show that with a handful of
    observations a bar plus error bar hides the very thing you need -- many
    different datasets give the same bar -- and recommend plotting every point.
    """
    keys = _present(results, HEADLINE)
    if not keys:
        return []

    columns = 3
    rows = int(np.ceil(len(keys) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.7 * columns, 3.7 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        _dot_panel(ax, results, key, annotate=True)
    for ax in axes[len(keys) :]:
        ax.set_visible(False)

    title_block(
        fig,
        "F1.1  All results, every individual run",
        "Each hollow marker is one training run. Thick line = average of the three. "
        "Thin vertical line = ±1 standard deviation (how much the number moved between runs).",
    )
    provenance(
        fig,
        results.note(
            "each marker = 1 run · thick tick = mean · thin bar = ±1 SD (not SE, not CI) · "
            "↑ higher is better, ↓ lower is better"
        ),
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.93), w_pad=2.4, h_pad=2.2)
    return save(fig, out_dir, "F1.1_overview_all_metrics")


def f1_group(results: Results, keys: list[str], title: str, subtitle: str,
             out_dir: Path, name: str, banner: str | None = None) -> list[Path]:
    """One metric group at a larger size, for the report body.

    The overview grid (F1.1) is for scanning everything at once; these are for
    dropping a single topic into a document at a readable size.
    """
    keys = _present(results, keys)
    if not keys:
        return []

    fig, axes = plt.subplots(1, len(keys), figsize=(4.7 * len(keys), 4.6))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        _dot_panel(ax, results, key, annotate=True)

    title_block(fig, title, subtitle, subtitle_y=0.945)
    if banner:
        fig.text(
            0.01, 0.875, banner, ha="left", va="top", fontsize=8.5, color="#8A5A00",
            bbox=dict(facecolor="#FFF6E5", edgecolor="#E69F00", linewidth=0.9, pad=5),
        )
    provenance(fig, results.note("each marker = 1 run · thick tick = mean · thin bar = ±1 SD"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.80 if banner else 0.90))
    return save(fig, out_dir, name)


def f1_speed(results: Results, out_dir: Path) -> list[Path]:
    """F1.2 -- the three speed questions side by side.

    Separated from the overview because "fast" is not one thing, and the three
    numbers disagree about who wins. Throughput and latency follow MLPerf's Offline
    and Single-Stream definitions respectively.
    """
    return f1_group(
        results,
        ["train_time", "throughput", "latency"],
        "F1.2  Speed, asked three different ways",
        "These three do not have to agree, and here they do not. Training time is one long job; throughput is "
        "digits per second in bulk; latency is one digit answered alone. A framework can lead one and trail another.",
        out_dir,
        "F1.2_speed",
    )


def f1_energy(results: Results, out_dir: Path) -> list[Path]:
    """F1.3 -- energy, shown with an explicit do-not-conclude banner.

    Kept as its own figure precisely so the caveat travels with the numbers. If
    energy appeared only inside the overview grid, someone would crop the panel.
    """
    return f1_group(
        results,
        ["energy_total", "energy_dynamic", "power_dynamic"],
        "F1.3  Energy — recorded, not interpreted",
        "Total is everything the GPU drew; dynamic subtracts what idling for the same time would have cost; "
        "power is how hard it worked rather than for how long.",
        out_dir,
        "F1.3_energy",
        banner="NO CONCLUSION IS DRAWN FROM THESE NUMBERS in this project. They come from the GPU's own sensor "
        "(~10 readings/second, see F6.3), they restate the speed result in part, and the baseline they depend on\n"
        "drifts between runs (see F6.2). Energy is revisited once later experiments give it something to be "
        "compared against. Shown here for the record only.",
    )


# --- F2: paired within-block views -------------------------------------------


def f2_slopegraph(results: Results, out_dir: Path) -> list[Path]:
    """F2.1 -- does the ordering hold inside every block?

    All three frameworks run back-to-back in one session per seed, so a seed is a
    BLOCK: the machine, its temperature and its neighbours are shared. Comparing
    inside a block cancels that drift.

    Crossing lines mean the ordering flipped and no ranking survives. Parallel lines
    that all rise or fall together mean the block moved, not the frameworks.
    """
    keys = _present(results, HEADLINE)
    if not keys or len(results.blocks) < 2:
        return []

    columns = 3
    rows = int(np.ceil(len(keys) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.7 * columns, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        metric = METRICS[key]
        slopegraph(ax, results.conditions, results.blocks, _getter(results, key))
        ax.set_ylabel(
            better_label(metric.label, metric.unit, metric.better, compact=True), fontsize=9
        )
        ax.set_title(
            wrap(metric.plain, 46), fontsize=8.5, fontweight="normal",
            color=TEXT_GREY, loc="left",
        )
    for ax in axes[len(keys) :]:
        ax.set_visible(False)

    legend_frameworks(axes[0], results.conditions, loc="best", fontsize=8)

    title_block(
        fig,
        "F2.1  The same comparison inside each seed",
        "One line per framework. Lines that CROSS mean the ordering flipped between seeds, so no ranking holds. "
        "Lines that stay parallel but all move together mean the seed moved, not the framework.",
    )
    provenance(
        fig,
        results.note(
            "paired design: all frameworks share one Colab session per seed · "
            "↑ higher is better, ↓ lower is better"
        ),
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.92), w_pad=2.4, h_pad=2.2)
    return save(fig, out_dir, "F2.1_slopegraph_within_seed")


def f2_differences(results: Results, out_dir: Path) -> list[Path]:
    """F2.2 -- pairwise differences against zero. The verdict figure.

    Every dot is one seed's A-minus-B, computed inside that seed. If all the dots
    for a pair sit on one side of zero, that framework won in every block. If they
    straddle zero the two swapped places and no ranking is supportable, however far
    apart their averages look.

    This is the "difference panel" Weissgerber et al. recommend for paired data,
    where a bar chart would wrongly imply the groups are independent.
    """
    keys = _present(results, HEADLINE)
    pairs = pairs_of(results.conditions)
    if not keys or len(pairs) < 1 or len(results.blocks) < 2:
        return []

    columns = 3
    rows = int(np.ceil(len(keys) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.7 * columns, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes, keys):
        metric = METRICS[key]
        difference_dots(ax, pairs, results.blocks, _getter(results, key),
                        decimals=metric.decimals)
        ax.set_ylabel(f"difference [{metric.unit}]", fontsize=9)
        # pad clears the verdict labels parked at the top of the axes
        ax.set_title(metric.label, fontsize=9.5, loc="left", pad=34)
    for ax in axes[len(keys) :]:
        ax.set_visible(False)

    title_block(
        fig,
        "F2.2  Does one framework win in every single seed?",
        "Each dot is one seed's difference between two frameworks, computed inside that seed. All dots on one "
        "side of the black line → a consistent direction. Dots straddling it (grey ×) → they swap places, so no "
        "ordering exists. Note this answers DIRECTION only; whether the gap is large enough to matter is F6.1.",
    )
    provenance(fig, results.note("differences computed WITHIN each seed, then compared"))
    fig.tight_layout(rect=(0, 0.025, 1, 0.92), w_pad=2.4, h_pad=2.6)
    return save(fig, out_dir, "F2.2_paired_differences")


# --- F3: profile and trade-offs ----------------------------------------------


def f3_profile(results: Results, out_dir: Path) -> list[Path]:
    """F3.1 -- who is good at what, on one normalized scale.

    Each metric is rescaled so the best framework scores 1.0 and worse ones score
    below it, whichever direction "better" runs in. That makes metrics with
    completely different units comparable on one axis.

    These are ratios, so the summary across metrics is a GEOMETRIC mean, following
    Fleming & Wallace (1986, CACM 29(3):218-221): only the geometric mean is
    invariant to which framework you happen to normalize against, so it cannot be
    gamed by changing the baseline.

    Energy is excluded on purpose -- it is recorded but not yet interpreted, so it
    must not contribute to a ranking.
    """
    keys = interpretable(_present(results, HEADLINE))
    if not keys:
        return []

    scores: dict[str, list[float]] = {c: [] for c in results.conditions}
    for key in keys:
        metric = METRICS[key]
        means = {c: float(np.mean(results.values(key, c))) for c in results.conditions}
        if metric.better == "up":
            best = max(means.values())
            for c, v in means.items():
                scores[c].append(v / best if best else np.nan)
        else:
            best = min(means.values())
            for c, v in means.items():
                scores[c].append(best / v if v else np.nan)

    x = np.arange(len(keys))
    fig, (ax, ax_geo) = plt.subplots(
        1, 2, figsize=(6.0 + 1.5 * len(keys), 4.6), gridspec_kw={"width_ratios": [len(keys), 1.4]}
    )

    ax.axhline(1.0, color=TEXT_GREY, linewidth=1.0, linestyle="-", alpha=0.6)
    ax.annotate("1.0 = best framework on that metric", xy=(1.0, 1.0),
                xycoords=("axes fraction", "data"), xytext=(-4, 6),
                textcoords="offset points", fontsize=8, color=TEXT_GREY, ha="right")

    for condition in results.conditions:
        ident = identity(condition)
        ax.plot(x, scores[condition], color=ident.colour, marker=ident.marker,
                linestyle=ident.linestyle, markersize=7, markerfacecolor="white",
                markeredgewidth=1.6)

    ax.set_xticks(x)
    ax.set_xticklabels([METRICS[k].label for k in keys], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("relative score  (1.0 = best, lower = worse)")
    ax.set_ylim(min(0.55, min(min(v) for v in scores.values()) - 0.05), 1.05)
    ax.grid(axis="x", visible=False)
    legend_frameworks(ax, results.conditions, loc="lower left", fontsize=8.5)

    geo = {
        c: float(np.exp(np.mean(np.log(np.asarray(v, dtype=float)))))
        for c, v in scores.items()
    }
    dot_interval(ax_geo, results.conditions, {c: [geo[c]] for c in results.conditions},
                 show_sd=False, annotate=True)
    ax_geo.set_ylabel("geometric mean of the scores")
    ax_geo.set_title("overall profile", fontsize=9.5, loc="left")
    ax_geo.set_ylim(ax.get_ylim())
    # A composite score is the easiest figure in the set to over-read: it silently
    # gives equal weight to a metric whose differences are real and one whose
    # differences are inside the noise. Say so on the figure.
    ax_geo.annotate(
        "treat as a summary, not a verdict:\nthis weights every metric equally,\n"
        "including ones whose differences\nare inside the noise (see F6.1)",
        xy=(0.5, 0.02), xycoords="axes fraction", ha="center", va="bottom",
        fontsize=7.4, color=FAINT_GREY, style="italic", linespacing=1.4,
    )

    title_block(
        fig,
        "F3.1  Normalized profile — who is strong where",
        "Every metric rescaled so the best framework scores 1.0, whichever direction 'better' runs in. Lines "
        "track one framework across the metrics (a parallel-coordinates reading) — the horizontal axis is a list "
        "of separate metrics, not a scale, so the slopes carry no meaning of their own.",
    )
    provenance(fig, results.note("means over seeds · energy excluded: recorded but not yet interpreted"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    return save(fig, out_dir, "F3.1_normalized_profile")


def _pareto_panel(ax, results: Results, cost_key: str, quality_key: str) -> None:
    cost, quality = METRICS[cost_key], METRICS[quality_key]

    centroids = {}
    for condition in results.conditions:
        ident = identity(condition)
        xs = list(results.values(cost_key, condition))
        ys = list(results.values(quality_key, condition))
        ax.scatter(xs, ys, s=26, color=ident.colour, alpha=0.35, marker=ident.marker,
                   linewidths=0, zorder=3)
        centroids[condition] = (float(np.mean(xs)), float(np.mean(ys)))

    # Label placement with simple collision avoidance. Two frameworks that land close
    # together -- which is exactly the interesting case -- would otherwise print their
    # names on top of each other.
    span_x = max(1e-9, max(c[0] for c in centroids.values()) - min(c[0] for c in centroids.values()))
    span_y = max(1e-9, max(c[1] for c in centroids.values()) - min(c[1] for c in centroids.values()))
    placed: list[tuple[float, float]] = []

    for condition, (cx, cy) in sorted(centroids.items(), key=lambda kv: kv[1][0]):
        ident = identity(condition)
        ax.scatter([cx], [cy], s=170, facecolor="white", edgecolor=ident.colour,
                   marker=ident.marker, linewidths=2.2, zorder=5)

        crowded = any(
            abs(cx - px) < 0.18 * span_x and abs(cy - py) < 0.25 * span_y
            for px, py in placed
        )
        offset = (9, -16) if crowded else (9, 7)
        ax.annotate(ident.label, (cx, cy), xytext=offset, textcoords="offset points",
                    fontsize=9, color=ident.colour, fontweight="bold")
        placed.append((cx, cy))

    # Pareto frontier: lower cost and higher quality both preferred.
    ordered = sorted(centroids.items(), key=lambda kv: kv[1][0])
    frontier, best_quality = [], -np.inf
    for condition, (cx, cy) in ordered:
        if cy > best_quality:
            frontier.append((cx, cy))
            best_quality = cy
    if len(frontier) > 1:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                color=TEXT_GREY, linewidth=1.2, linestyle="--", alpha=0.7, zorder=2,
                label="Pareto frontier")
        ax.legend(loc="lower right", fontsize=8)

    ax.set_xlabel(better_label(cost.label, cost.unit, cost.better))
    ax.set_ylabel(better_label(quality.label, quality.unit, quality.better))
    ax.annotate(
        "best corner\n(cheap and accurate)",
        xy=(0.015, 0.985), xycoords="axes fraction", xytext=(26, -30),
        textcoords="offset points", fontsize=7.6, color=FAINT_GREY, style="italic",
        va="top", linespacing=1.3,
        arrowprops=dict(arrowstyle="-|>", color=FAINT_GREY, linewidth=0.9,
                        connectionstyle="arc3,rad=0.25"),
    )


def f3_tradeoff(results: Results, out_dir: Path) -> list[Path]:
    """F3.2 -- what does the accuracy cost?

    Quality on the vertical axis, a cost on the horizontal. Faint dots are the
    individual runs; the large hollow marker is the framework average. The dashed
    line joins the points nothing else beats on both axes at once (the Pareto
    frontier) -- those are the only choices a designer would consider.

    Reading it here: if the vertical spread is tiny, accuracy is not in play and the
    decision collapses to cost alone. Energy is deliberately not used as a cost axis
    while it remains uninterpreted.
    """
    quality = "accuracy"
    costs = [k for k in ["latency", "memory_train", "train_time"] if results.has(k)]
    costs = interpretable(costs)
    if not results.has(quality) or not costs:
        return []

    fig, axes = plt.subplots(1, len(costs), figsize=(4.9 * len(costs), 4.5))
    axes = np.atleast_1d(axes).ravel()
    for ax, cost_key in zip(axes, costs):
        _pareto_panel(ax, results, cost_key, quality)

    title_block(
        fig,
        "F3.2  Trade-off: accuracy against what it costs",
        "Small faint markers = individual runs; large hollow markers = framework average. "
        "Up and to the left is better. A flat vertical spread means accuracy is not the deciding factor.",
    )
    provenance(fig, results.note("energy excluded as a cost axis: recorded but not yet interpreted"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    return save(fig, out_dir, "F3.2_tradeoff_pareto")


# --- F4: training dynamics ---------------------------------------------------


def _epoch_curve(ax, results: Results, column: str, ylabel: str, better: str) -> None:
    epochs = _with_meta(results.epochs, results)
    for condition in results.conditions:
        ident = identity(condition)
        subset = epochs[epochs[results.condition] == condition]
        if subset.empty:
            continue
        # Individual seeds behind, faint: the reader sees the raw runs, not only a band.
        for _, run in subset.groupby("run_id"):
            run = run.sort_values("epoch")
            ax.plot(run["epoch"], run[column], color=ident.colour, alpha=0.28,
                    linewidth=1.0, zorder=2)
        grouped = subset.groupby("epoch")[column]
        mean, sd = grouped.mean(), grouped.std(ddof=1)
        ax.fill_between(mean.index, mean - sd, mean + sd, color=ident.colour,
                        alpha=0.13, linewidth=0, zorder=1)
        ax.plot(mean.index, mean, color=ident.colour, marker=ident.marker,
                linestyle=ident.linestyle, markersize=5.5, markerfacecolor="white",
                markeredgewidth=1.4, zorder=4)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(epochs["epoch"].unique()))


def f4_learning_curves(results: Results, out_dir: Path) -> list[Path]:
    """F4.1 -- do they converge the same way, not just to the same place?"""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _epoch_curve(ax, results, "test_accuracy_pct",
                 better_label("Test accuracy", "%", "up"), "up")
    legend_frameworks(ax, results.conditions, loc="lower right")
    title_block(
        fig,
        "F4.1  Learning curves",
        "Bold line = average over seeds, shaded band = ±1 SD, faint lines = the individual runs behind it. "
        "Watch epoch 1: that is where frameworks differ most, and where the difference then disappears.",
    )
    provenance(fig, results.note("band = ±1 SD across seeds"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    return save(fig, out_dir, "F4.1_learning_curves")


def f4_epoch_time(results: Results, out_dir: Path) -> list[Path]:
    """F4.2 -- the diagnostic to check BEFORE trusting any timing figure.

    Epoch times should be flat. A tall first epoch means that run also built the
    dataset cache while being timed, which inflates its training time and makes it
    incomparable. That failure has already invalidated one run in this project, so
    it gets its own figure rather than a footnote.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _epoch_curve(ax, results, "epoch_train_time_s",
                 better_label("Time for this epoch", "s", "down"), "down")
    legend_frameworks(ax, results.conditions, loc="best")
    ax.annotate(
        "What to look for: a gentle drift like this is normal warm-up —\n"
        "caches fill, clocks settle. The failure being screened for is a FIRST\n"
        "epoch several times taller than the rest, which means that run was\n"
        "still building the dataset cache while the clock was running.",
        xy=(0.98, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=7.8, color=TEXT_GREY, style="italic", linespacing=1.4,
        bbox=dict(facecolor="white", edgecolor=GRID_GREY, pad=3.5),
    )
    title_block(
        fig,
        "F4.2  Per-epoch time stability (a validity check)",
        "Not a result — a diagnostic, and it should be read before trusting any timing number. Every epoch does "
        "identical work, so large steps between epochs mean something other than the framework was being timed.",
    )
    provenance(fig, results.note("one faint line per run · bold = mean"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    return save(fig, out_dir, "F4.2_epoch_time_stability")


def f4_sparsification(results: Results, out_dir: Path) -> list[Path]:
    """F4.3 -- does the network get quieter as it gets better?"""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _epoch_curve(ax, results, "spike_rate_pct",
                 better_label("Spike rate", "%", "down"), "down")
    legend_frameworks(ax, results.conditions, loc="best")
    title_block(
        fig,
        "F4.3  Spiking activity falls while accuracy rises",
        "Training does not only make the network more accurate — it finds a QUIETER solution. "
        "Fewer spikes for the same job is the central efficiency argument for spiking networks.",
    )
    provenance(fig, results.note("band = ±1 SD across seeds · measured on the test set"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    return save(fig, out_dir, "F4.3_sparsification")


def f4_loss(results: Results, out_dir: Path) -> list[Path]:
    """F4.4 -- are all frameworks really optimising the same objective?"""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    _epoch_curve(axes[0], results, "train_loss", "Training loss  (lower ↓)", "down")
    axes[0].set_title("on data it is learning from", fontsize=9.5, loc="left")
    _epoch_curve(axes[1], results, "test_loss", "Test loss  (lower ↓)", "down")
    axes[1].set_title("on data it has never seen", fontsize=9.5, loc="left")
    legend_frameworks(axes[1], results.conditions, loc="best")
    title_block(
        fig,
        "F4.4  Loss curves",
        "Loss is the quantity training actually minimises. Curves of the same shape confirm all three "
        "frameworks are solving the identical optimisation problem. Test loss turning upward would mean overfitting.",
    )
    provenance(fig, results.note("band = ±1 SD across seeds"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    return save(fig, out_dir, "F4.4_loss_curves")


# --- F5: structure -----------------------------------------------------------


def f5_layer_activity(results: Results, out_dir: Path) -> list[Path]:
    """F5.1 -- where the spiking happens, and which layers deserve trust.

    Marker AREA is the neuron count. That single choice does the explaining: the
    output layer's dot is almost invisible next to the first layer's, so it is
    immediately obvious why its percentage bounces around. An average over 10
    neurons is far noisier than an average over 10,800, and readers should not have
    to be told twice.
    """
    layers = _with_meta(results.layers, results)
    if layers.empty:
        return []

    indices = sorted(layers["layer_index"].unique())
    neurons = {
        int(i): int(layers[layers["layer_index"] == i]["neurons"].iloc[0]) for i in indices
    }

    # One panel per layer, each with its own y-scale. A single shared axis does not
    # work here: the 10-neuron output layer sits around 10 % while the two large
    # layers sit around 2 %, so sharing the axis would compress exactly the
    # comparison the figure exists to show. The differing scales are stated on the
    # figure, because independent axes are easy to misread.
    fig, axes = plt.subplots(1, len(indices), figsize=(3.9 * len(indices), 4.8))
    axes = np.atleast_1d(axes).ravel()

    for ax, layer_index in zip(axes, indices):
        count = neurons[int(layer_index)]
        subset = layers[layers["layer_index"] == layer_index]
        series = {
            c: list(subset[subset[results.condition] == c]["spike_rate_pct"])
            for c in results.conditions
        }
        # Marker area tracks the neuron count, so the unreliable layer looks slight.
        size = 26 + 300 * (count / max(neurons.values())) ** 0.45
        dot_interval(ax, results.conditions, series, dot_size=size, decimals=3)
        ax.set_title(
            f"layer {int(layer_index)} — {count:,} neurons",
            fontsize=9.5,
            loc="left",
        )
        ax.set_ylabel(better_label("Spike rate", "%", "down", compact=True), fontsize=9)
        if count < 100:
            # Open a clear band below the data rather than writing over it.
            low, high = ax.get_ylim()
            ax.set_ylim(low - 0.55 * (high - low), high)
            ax.annotate(
                f"only {count} neurons — an average over so few\n"
                "bounces around far more than one over thousands.\n"
                "Widest spread in the figure. Do not over-read it.",
                xy=(0.5, 0.015), xycoords="axes fraction", ha="center", va="bottom",
                fontsize=7.6, color=TEXT_GREY, style="italic", linespacing=1.35,
                bbox=dict(facecolor="white", edgecolor=GRID_GREY, pad=3.5),
            )

    legend_frameworks(axes[0], results.conditions, loc="best", fontsize=8.5)

    title_block(
        fig,
        "F5.1  Spiking activity layer by layer",
        "One marker per run; marker size shows how many neurons the layer has — which is also how much to trust "
        "it. NOTE each panel has its own vertical scale, because the 10-neuron output layer fires several times "
        "more often than the two large layers and would otherwise flatten them.",
    )
    provenance(fig, results.note("one marker per run · tick = mean · bar = ±1 SD · independent y-scales"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.87))
    return save(fig, out_dir, "F5.1_layer_spike_rate")


def f5_spike_budget(results: Results, out_dir: Path) -> list[Path]:
    """F5.2 -- where the spike budget is actually spent.

    Rates mislead about totals. A huge layer at a modest rate emits far more spikes
    than a tiny layer at a high one, and it is the total that costs energy on
    neuromorphic hardware. Bars are appropriate here: these are exact counts, not
    estimates carrying uncertainty.
    """
    layers = _with_meta(results.layers, results)
    if layers.empty:
        return []

    indices = sorted(layers["layer_index"].unique())
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    width = 0.8 / len(results.conditions)
    shades = [1.0, 0.62, 0.34]

    for position, condition in enumerate(results.conditions):
        ident = identity(condition)
        subset = layers[layers[results.condition] == condition]
        base = position * width - 0.4 + width / 2
        bottom = 0.0
        for depth, layer_index in enumerate(indices):
            cell = subset[subset["layer_index"] == layer_index]["total_spikes"]
            if cell.empty:
                continue
            value = float(cell.mean()) / 1e6
            ax.bar(base, value, width=width * 0.9, bottom=bottom,
                   color=ident.colour, alpha=shades[depth % len(shades)],
                   edgecolor="white", linewidth=1.0,
                   label=f"layer {int(layer_index)}" if position == 0 else None)
            bottom += value

    ax.set_xticks([p * width - 0.4 + width / 2 for p in range(len(results.conditions))])
    ax.set_xticklabels([identity(c).label for c in results.conditions])
    ax.set_ylabel("total spikes across the test set  [millions, ↓ better]")
    ax.grid(axis="x", visible=False)
    ax.legend(title="stacked by layer (darkest = first layer)", loc="upper right",
              fontsize=8, title_fontsize=8)

    title_block(
        fig,
        "F5.2  Total spikes, and which layer emits them",
        "Rates can mislead about totals: a large layer at a modest rate emits far more spikes than a small layer "
        "at a high rate — and it is the total that costs energy on neuromorphic hardware.",
    )
    provenance(fig, results.note("mean over seeds · exact counts, so bars are appropriate here"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    return save(fig, out_dir, "F5.2_spike_budget")


# --- F6: measurement quality -------------------------------------------------


def f6_effect_vs_noise(results: Results, out_dir: Path) -> list[Path]:
    """F6.1 -- which of our differences actually clear the noise?

    For each metric: how far apart are the frameworks, measured in units of that
    metric's OWN run-to-run spread? A gap of 0.5 SD is invisible against the noise;
    a gap of 10 SD is unmistakable. Expressing everything in SD units puts metrics
    with different units on one comparable axis.

    This turns the noise-floor argument from a paragraph of prose into a picture,
    and it is the figure that decides which claims the experiment may make.
    """
    keys = _present(results, list(METRICS))
    rows = []
    for key in keys:
        series = _series(results, key)
        means = {c: float(np.mean(v)) for c, v in series.items() if len(v)}
        if len(means) < 2:
            continue
        spread = max(means.values()) - min(means.values())
        noise = pooled_sd(series)
        if not np.isfinite(noise):
            continue
        rows.append(
            {
                "key": key,
                "label": METRICS[key].label,
                "ratio": spread / noise if noise > 0 else np.inf,
                "group": METRICS[key].group,
            }
        )
    if not rows:
        return []

    frame = pd.DataFrame(rows).sort_values("ratio")
    # Infinite means zero measured noise (peak memory) -- plot it just past the axis.
    finite_max = frame.loc[np.isfinite(frame["ratio"]), "ratio"].max()
    cap = float(finite_max) * 1.35 if np.isfinite(finite_max) else 10.0
    plotted = frame["ratio"].replace(np.inf, cap)

    fig, ax = plt.subplots(figsize=(9.2, 0.44 * len(frame) + 3.4))
    y = np.arange(len(frame))

    ax.axvspan(0, 1.0, color=OKABE_ITO["vermillion"], alpha=0.08, linewidth=0)
    for threshold, colour in [(1.0, OKABE_ITO["vermillion"]), (2.0, OKABE_ITO["orange"])]:
        ax.axvline(threshold, color=colour, linewidth=1.2, linestyle="--", alpha=0.9)

    colours = ["#BBBBBB" if g == "energy" else OKABE_ITO["blue"] for g in frame["group"]]
    # Bars whose noise measured exactly zero have no finite length. Drawing them at
    # the axis maximum would read as a real value, so they are hatched: the label
    # carries the meaning and the hatching says "not to scale".
    hatches = ["" if np.isfinite(r) else "///" for r in frame["ratio"]]
    for i, (value, colour, hatch) in enumerate(zip(plotted, colours, hatches)):
        ax.barh(i, value, color=colour, alpha=0.85, height=0.62, hatch=hatch,
                edgecolor="white" if hatch else "none", linewidth=0)

    for i, (ratio, group) in enumerate(zip(frame["ratio"], frame["group"])):
        text = "no measurable noise" if not np.isfinite(ratio) else f"{ratio:.1f}×"
        if group == "energy":
            text += "  (not interpreted)"
        ax.annotate(text, (plotted.iloc[i], i), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, va="center", color=TEXT_GREY)

    # Threshold labels sit above the plot area, horizontal, so they never sit on a bar.
    for threshold, colour, text in [
        (1.0, OKABE_ITO["vermillion"], "1 SD\nsame size as the noise"),
        (2.0, OKABE_ITO["orange"], "2 SD\nclearly above it"),
    ]:
        ax.annotate(text, xy=(threshold, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 8), textcoords="offset points", fontsize=7.8,
                    color=colour, ha="center", va="bottom", linespacing=1.3)

    ax.set_yticks(y)
    ax.set_yticklabels(frame["label"], fontsize=9)
    ax.set_xlabel("framework spread ÷ that metric's own run-to-run spread   [in SD units, → more trustworthy]")
    ax.set_xlim(0, cap * 1.22)
    ax.grid(axis="y", visible=False)

    title_block(
        fig,
        "F6.1  Which differences are bigger than the noise?",
        "Bars in the shaded zone are smaller than the metric's own variation between runs — those differences "
        "cannot be claimed. Long bars are safe to rank. Grey bars are energy: measured, but not interpreted. "
        "Hatched bars had no measurable run-to-run variation at all, so their length is symbolic.",
    )
    provenance(fig, results.note("noise = mean of the per-framework SDs"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    return save(fig, out_dir, "F6.1_effect_size_vs_noise")


def f6_idle_baseline(results: Results, out_dir: Path) -> list[Path]:
    """F6.2 -- why the energy numbers are not interpreted.

    Dynamic energy is a SUBTRACTION: total energy minus (idle power x duration). So
    it is only as trustworthy as the idle figure being subtracted. Each arrow here
    is one run's idle power before training and again afterwards. Arrows that do not
    all point the same way mean the baseline is not a stable property of the
    machine -- and a couple of watts over ten minutes is more than a kilojoule.
    """
    needed = {"idle_power_cold_w", "idle_power_after_train_w"}
    if not needed <= set(results.runs.columns):
        return []

    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    # Fixed display order, not alphabetical: the schema promises the same
    # left-to-right framework order in every figure.
    order = {c: i for i, c in enumerate(results.conditions)}
    frame = (
        results.runs.assign(_order=results.runs[results.condition].map(order))
        .sort_values(["_order", "seed"])
        .reset_index(drop=True)
    )

    flagged = 0
    for x, (_, row) in enumerate(frame.iterrows()):
        ident = identity(row[results.condition])
        cold, hot = float(row["idle_power_cold_w"]), float(row["idle_power_after_train_w"])
        ax.annotate("", xy=(x, hot), xytext=(x, cold),
                    arrowprops=dict(arrowstyle="-|>", color=ident.colour,
                                    linewidth=1.6, alpha=0.85))
        ax.scatter([x], [cold], s=34, color=ident.colour, marker="o", zorder=4)
        ax.scatter([x], [hot], s=52, facecolor="white", edgecolor=ident.colour,
                   marker=ident.marker, linewidths=1.6, zorder=5)

        # An absent warning arrives from the CSV as NaN, and str(NaN) is the
        # non-empty text "nan" -- so test for missing data before testing the string.
        warning = row.get("energy_warnings")
        if pd.notna(warning) and str(warning).strip():
            flagged += 1
            ax.annotate("⚠", (x, max(cold, hot)), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=13,
                        color=OKABE_ITO["vermillion"])

    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(
        [f"{identity(r[results.condition]).label}\nseed {int(r['seed'])}"
         for _, r in frame.iterrows()],
        fontsize=8,
    )
    ax.set_ylabel("GPU idle power  [W]")
    ax.grid(axis="x", visible=False)

    caption = (
        "arrow tail ● = idle before training (cold)   →   head = idle after training (hot)\n"
        "Dynamic energy subtracts this baseline, so wherever it drifts the energy figure drifts with it. "
        "Two watts over ten minutes is more than a kilojoule."
    )
    if flagged:
        caption += f"\n⚠ {flagged} run(s) whose baseline the measurement itself flagged as unstable."
    ax.annotate(caption, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 12),
                textcoords="offset points", fontsize=8.2, color=TEXT_GREY,
                linespacing=1.45, va="bottom")

    title_block(
        fig,
        "F6.2  The idle baseline moves — which is why energy is not interpreted",
        "A validity figure, not a result. Energy is recorded in this project; no conclusion is drawn from it yet.",
    )
    provenance(fig, results.note("idle power measured before and immediately after each run"))
    fig.subplots_adjust(top=0.74, bottom=0.13)
    return save(fig, out_dir, "F6.2_idle_baseline_stability")


def f6_sensor(results: Results, out_dir: Path) -> list[Path]:
    """F6.3 -- how often the power sensor actually reports.

    Energy is computed by integrating power samples over time, so the sampling rate
    bounds how good that integral can be. The GPU's sensor is widely assumed to
    refresh every 10-20 ms; measured here it is roughly 100 ms, i.e. about ten times
    a second. Worth a figure because it is a cheap, verifiable fact that constrains
    every energy number in the project.
    """
    column = "nvml_update_interval_ms"
    if column not in results.runs.columns or results.runs[column].isna().all():
        return []

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    dot_interval(ax, results.conditions,
                 {c: list(results.runs[results.runs[results.condition] == c][column].dropna())
                  for c in results.conditions},
                 annotate=True, decimals=1)
    ax.axhline(20, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.2)
    ax.annotate("20 ms — the refresh rate commonly assumed", (0.02, 20),
                xytext=(0, 6), textcoords="offset points", fontsize=8,
                color=OKABE_ITO["vermillion"])
    ax.set_ylabel("measured sensor refresh interval  [ms, ↓ better]")

    title_block(
        fig,
        "F6.3  How often the GPU power sensor actually updates",
        "Energy is an integral of power samples, so this sets a floor on how precise any energy figure can be. "
        "Measured at roughly 100 ms — about ten readings a second, not the 20 ms often assumed.",
    )
    provenance(fig, results.note("measured per run by polling until the reading changes"))
    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    return save(fig, out_dir, "F6.3_sensor_interval")
