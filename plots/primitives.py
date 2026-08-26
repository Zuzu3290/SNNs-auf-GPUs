"""The three reusable marks every figure family is built from.

Keeping them here rather than inline in each figure is what makes the schema a
schema: change the dot size once and thirty figures agree.

1. `dot_interval` -- one dot per run, a mean tick, a +/-1 SD bar. This replaces the
   bar-chart-with-error-bars that Weissgerber et al. (2015, PLOS Biology
   13(4):e1002128) argue against for small samples: "Summary statistics ... are only
   meaningful when there are enough data to summarize." With three runs, showing all
   three costs nothing and hides nothing.

2. `slopegraph` -- one line per condition across the paired blocks. Answers a
   question no unpaired plot can: does the ordering hold WITHIN each block?

3. `difference_dots` -- pairwise differences against a zero line. This is the
   "difference panel" the same paper recommends for paired data, where a plain bar
   chart "erroneously suggest[s] that the groups being compared are independent".
"""

from __future__ import annotations

import numpy as np

from .style import FAINT_GREY, TEXT_GREY, identity


def _offsets(count: int, spread: float = 0.13) -> np.ndarray:
    """Symmetric, DETERMINISTIC horizontal offsets for overlapping points.

    Random jitter would make the figure change every time it is regenerated, which
    breaks the promise that figures are reproducible from the CSVs. Evenly spaced
    offsets read just as clearly and are stable.
    """
    if count <= 1:
        return np.zeros(1)
    return np.linspace(-spread, spread, count)


def dot_interval(
    ax,
    conditions: list[str],
    series_by_condition: dict[str, list[float]],
    *,
    show_mean: bool = True,
    show_sd: bool = True,
    dot_size: float = 42.0,
    annotate: bool = False,
    decimals: int = 2,
) -> None:
    """Every individual run as a dot, plus mean and +/-1 SD.

    The mean is a wide horizontal tick rather than a bar, so no ink implies a
    baseline at zero -- which lets the y-axis crop to the interesting range without
    the distortion a truncated bar would cause.
    """
    for x, condition in enumerate(conditions):
        values = [v for v in series_by_condition.get(condition, []) if v is not None]
        if not values:
            continue
        ident = identity(condition)
        array = np.asarray(values, dtype=float)

        if show_sd and len(array) > 1:
            sd = array.std(ddof=1)
            ax.vlines(
                x,
                array.mean() - sd,
                array.mean() + sd,
                color=ident.colour,
                linewidth=1.4,
                alpha=0.55,
                zorder=2,
            )
        if show_mean:
            ax.hlines(
                array.mean(),
                x - 0.22,
                x + 0.22,
                color=ident.colour,
                linewidth=2.4,
                zorder=3,
            )

        ax.scatter(
            x + _offsets(len(array)),
            array,
            s=dot_size,
            facecolor="white",
            edgecolor=ident.colour,
            marker=ident.marker,
            linewidths=1.5,
            zorder=4,
        )

        if annotate:
            label = f"{array.mean():.{decimals}f}"
            if len(array) > 1:
                label += f"\n±{array.std(ddof=1):.{decimals}f}"
            ax.annotate(
                label,
                (x + 0.28, array.mean()),
                fontsize=8,
                color=TEXT_GREY,
                va="center",
                ha="left",
            )

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([identity(c).label for c in conditions])
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.grid(axis="x", visible=False)


def slopegraph(
    ax,
    conditions: list[str],
    blocks: list[int],
    value_of,
    *,
    block_label: str = "seed",
) -> None:
    """One line per condition across the paired blocks.

    `value_of(condition, block) -> float | None`.

    Reading it: lines that cross mean the ordering flipped, so there is no ranking.
    Lines that stay parallel while all moving together mean the BLOCK moved, not the
    conditions -- which is the signature of a shared-machine or shared-seed effect
    rather than a framework difference.
    """
    x = np.arange(len(blocks))
    for condition in conditions:
        ident = identity(condition)
        y = [value_of(condition, block) for block in blocks]
        ax.plot(
            x,
            y,
            color=ident.colour,
            marker=ident.marker,
            linestyle=ident.linestyle,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{block_label} {b}" for b in blocks])
    ax.set_xlim(-0.35, len(blocks) - 0.65)
    ax.grid(axis="x", visible=False)


def difference_dots(
    ax,
    pairs: list[tuple[str, str]],
    blocks: list[int],
    value_of,
    *,
    decimals: int = 2,
) -> None:
    """Within-block differences for each pair of conditions, against a zero line.

    `value_of(condition, block) -> float | None`.

    This is the plot that decides whether a ranking exists. Every dot is one block's
    A-minus-B. If the dots sit entirely on one side of zero the ordering held in
    every block; if they straddle zero the two conditions swapped places and no
    ranking is supportable, however far apart their averages look.
    """
    ax.axhline(0.0, color=TEXT_GREY, linewidth=1.2, zorder=2)

    # Collect first, draw second: the verdict labels need to know the final y-range
    # so they can be parked above the data instead of on top of a marker.
    collected: list[tuple[int, np.ndarray, bool]] = []
    for x, (left, right) in enumerate(pairs):
        diffs = []
        for block in blocks:
            a, b = value_of(left, block), value_of(right, block)
            if a is not None and b is not None:
                diffs.append(a - b)
        if diffs:
            array = np.asarray(diffs, dtype=float)
            consistent = bool((array > 0).all() or (array < 0).all())
            collected.append((x, array, consistent))

    if not collected:
        return

    for x, array, consistent in collected:
        # Colour by verdict, not by condition: this figure is about the pair.
        colour = identity(pairs[x][0]).colour if consistent else FAINT_GREY
        ax.vlines(x, array.min(), array.max(), color=colour, linewidth=1.2, alpha=0.5)
        # 'x' is an unfilled marker, so it takes `color`, not a face/edge pair.
        if consistent:
            ax.scatter(
                x + _offsets(len(array)),
                array,
                s=46,
                facecolor="white",
                edgecolor=colour,
                marker="o",
                linewidths=1.6,
                zorder=4,
            )
        else:
            ax.scatter(
                x + _offsets(len(array)),
                array,
                s=52,
                color=colour,
                marker="x",
                linewidths=1.8,
                zorder=4,
            )

    # Reserve a band at the top for the verdict labels.
    low = min(float(a.min()) for _, a, _ in collected)
    high = max(float(a.max()) for _, a, _ in collected)
    low, high = min(low, 0.0), max(high, 0.0)
    span = (high - low) or 1.0
    ax.set_ylim(low - 0.10 * span, high + 0.42 * span)
    label_y = high + 0.36 * span

    for x, array, consistent in collected:
        colour = identity(pairs[x][0]).colour if consistent else FAINT_GREY
        # State the typical size next to the direction. A consistent SIGN only says
        # an ordering exists; whether the ordering is big enough to matter is a
        # separate question, answered by the effect-size-vs-noise figure (F6.1).
        typical = float(np.median(np.abs(array)))
        ax.annotate(
            ("same direction\nin all "
             + f"{len(array)} seeds\ntypically {typical:.{decimals}f}")
            if consistent
            else "crosses zero\nthey swap places\nno ordering",
            (x, label_y),
            fontsize=7.4,
            color=colour if consistent else TEXT_GREY,
            ha="center",
            va="center",
            linespacing=1.35,
        )

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(
        [f"{identity(a).label}\n− {identity(b).label}" for a, b in pairs], fontsize=8.5
    )
    ax.set_xlim(-0.6, len(pairs) - 0.4)
    ax.grid(axis="x", visible=False)


def pairs_of(conditions: list[str]) -> list[tuple[str, str]]:
    """Every unordered pair, in display order."""
    return [
        (conditions[i], conditions[j])
        for i in range(len(conditions))
        for j in range(i + 1, len(conditions))
    ]


def pooled_sd(series_by_condition: dict[str, list[float]]) -> float:
    """A single noise estimate for a metric: the mean of the per-condition SDs.

    Averaging the within-condition spreads keeps the estimate free of the very
    between-condition differences it is being used to judge.
    """
    sds = [
        float(np.std(np.asarray(v, dtype=float), ddof=1))
        for v in series_by_condition.values()
        if len([x for x in v if x is not None]) > 1
    ]
    return float(np.mean(sds)) if sds else float("nan")
