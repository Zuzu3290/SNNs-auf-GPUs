"""Fixed visual identity, shared by every figure of every experiment.

The point of putting this in one file is consistency ACROSS experiments. A reader
who learns that green-dotted-triangle means Norse in experiment 1 must not have to
relearn it in experiment 5. So nothing here is chosen per-figure.

Palette: Okabe-Ito, the colourblind-safe set popularised for scientific figures by
Wong (2011), Nature Methods 8:441, doi:10.1038/nmeth.1618. Colour is never the only
channel -- every series also carries a distinct marker and linestyle, so figures
survive greyscale printing and colour vision deficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on Colab or in CI
import matplotlib.pyplot as plt

# --- Okabe-Ito ---------------------------------------------------------------
# The full eight. Framework colours are taken from here and then never reused for
# anything else, so a later experiment comparing variants has spare colours.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Reserved for extra conditions (variants, datasets) in future experiments.
# reddish_purple was moved OUT of this list when sinabs claimed it below, to keep
# the rule that a framework colour is never reused for anything else.
SPARE_COLOURS = [OKABE_ITO["orange"], OKABE_ITO["sky_blue"]]

GRID_GREY = "#CCCCCC"
TEXT_GREY = "#444444"
FAINT_GREY = "#888888"


@dataclass(frozen=True)
class Identity:
    """How one framework is drawn, everywhere, forever."""

    key: str
    label: str
    colour: str
    marker: str
    linestyle: str


# Order matters: it is the left-to-right order in every categorical figure, and it
# matches the column order of the tables in the reports.
FRAMEWORKS: dict[str, Identity] = {
    "snntorch": Identity("snntorch", "snnTorch", OKABE_ITO["blue"], "o", "-"),
    "spikingjelly": Identity(
        "spikingjelly", "SpikingJelly", OKABE_ITO["vermillion"], "s", "--"
    ),
    "norse": Identity("norse", "Norse", OKABE_ITO["bluish_green"], "^", ":"),
    # reddish_purple is the only one of the remaining Okabe-Ito colours that is
    # not easily confused with a colour already in use: sky_blue reads as blue
    # (snnTorch) and orange reads as vermillion (SpikingJelly).
    #
    # Adding a fourth entry here is safe for the existing experiments. Results
    # .conditions filters FRAMEWORK_ORDER by what is actually PRESENT in the data
    # (see src/plots/data.py), so ex1 and ex2 figures drawn from runs that contain
    # no sinabs rows are unchanged.
    "sinabs": Identity("sinabs", "Sinabs", OKABE_ITO["reddish_purple"], "D", "-."),
}

FRAMEWORK_ORDER = list(FRAMEWORKS)


def identity(key: str) -> Identity:
    """Look up how to draw a condition, tolerating unknown ones.

    Unknown keys get a spare colour rather than crashing, so an experiment that
    introduces a new variant still plots. It will look deliberately different from
    the three frameworks, which is the correct signal.
    """
    if key in FRAMEWORKS:
        return FRAMEWORKS[key]
    index = abs(hash(key)) % len(SPARE_COLOURS)
    return Identity(key, key, SPARE_COLOURS[index], "D", "-.")


def apply_rcparams() -> None:
    """Global defaults.

    Rougier et al. (2014) rule 5 is "do not trust the defaults" -- matplotlib is
    tuned for screens, not for a printed thesis. Bigger text, thinner spines, no
    top/right box, faint grid behind the data rather than over it.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": TEXT_GREY,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID_GREY,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.color": TEXT_GREY,
            "ytick.color": TEXT_GREY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.8,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
        }
    )


def title_block(fig, title: str, subtitle: str, subtitle_y: float = 0.955) -> None:
    """A bold title plus a plain-language "what to look for" line.

    The subtitle is not decoration. These figures are read by people outside
    neuromorphic computing, and a one-line statement of what the figure is for
    removes most of the guesswork. Rougier's rule 4: the caption is part of the
    figure.

    The subtitle is wrapped to the figure's own width. Left unwrapped, a long line
    of text widens the saved bounding box beyond the axes and the figure comes out
    stretched -- the text, not the data, would be setting the aspect ratio.
    """
    fig.suptitle(title, x=0.01, ha="left", va="top", y=0.995)

    # ~9.5pt italic averages a little under 5 characters per inch of figure width.
    width_chars = max(60, int(fig.get_size_inches()[0] * 15.5))
    fig.text(
        0.01,
        subtitle_y,
        wrap(subtitle, width_chars),
        ha="left",
        va="top",
        fontsize=9.5,
        color=TEXT_GREY,
        style="italic",
        linespacing=1.45,
    )


def provenance(fig, note: str) -> None:
    """Stamp what the numbers are, at the bottom of the figure.

    Wilke's chapter on uncertainty insists the meaning of an interval be stated
    wherever it is drawn -- an error bar alone is ambiguous between standard
    deviation, standard error and confidence interval. Putting it on the figure
    means the figure survives being pasted into a slide without its caption.
    """
    fig.text(0.01, 0.005, note, ha="left", va="bottom", fontsize=7.5, color=FAINT_GREY)


def better_label(label: str, unit: str, better: str, compact: bool = False) -> str:
    """Axis label with an explicit better-is direction.

    Several of these metrics invert -- higher accuracy is good, higher latency is
    bad -- and a reader should not have to remember which.

    `compact` uses a bare arrow, for grids where the spelled-out version would
    collide with the neighbouring panel. The arrow is decoded once in the figure's
    footer, so the meaning is never left to guesswork.
    """
    if compact:
        arrow = {"up": " ↑", "down": " ↓"}
    else:
        arrow = {"up": "  (higher is better ↑)", "down": "  (lower is better ↓)"}
    base = f"{label} [{unit}]" if unit else label
    return base + arrow.get(better, "")


def wrap(text: str, width: int = 44) -> str:
    """Soft-wrap a panel description so it cannot run into the next panel."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width))


def legend_frameworks(ax, keys: list[str], **kwargs) -> None:
    """A legend built from the fixed identities, showing colour+marker+linestyle."""
    handles = [
        plt.Line2D(
            [],
            [],
            color=identity(k).colour,
            marker=identity(k).marker,
            linestyle=identity(k).linestyle,
            markersize=6,
            label=identity(k).label,
        )
        for k in keys
    ]
    ax.legend(handles=handles, **kwargs)


# Output formats for this run. PNG is what the markdown report embeds.
_FORMATS: tuple[str, ...] = ("png",)


def set_formats(formats: tuple[str, ...]) -> None:
    """Choose the output formats once, for every figure in the run.

    Set here rather than threaded through all seventeen figure functions: the
    format is a property of the invocation, not of any individual figure.
    """
    global _FORMATS
    _FORMATS = tuple(formats)


def save(fig, out_dir: Path, name: str, formats: tuple[str, ...] | None = None) -> list[Path]:
    """Write one figure in every requested format and close it.

    PNG only by default -- that is what the markdown report embeds. Vector output
    is still available with `--formats png,pdf` if a thesis template ever wants it.

    Closing matters: a script that builds thirty figures without closing them will
    exhaust memory and emit a warning per figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats or _FORMATS:
        path = out_dir / f"{name}.{extension}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written
