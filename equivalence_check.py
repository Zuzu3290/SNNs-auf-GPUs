"""Draw what each framework's LIF neuron actually does, side by side.

Drives ONE spiking layer per framework with byte-identical input and plots the result:
input current, membrane trajectory, and spike raster. Prints the same numbers it draws.

    python equivalence_check.py
    python equivalence_check.py --experiment ex1        # figures -> experiments/ex1/plots
    python equivalence_check.py --out-dir /tmp/figs --formats png

This is a LOOK, not a gate. It reports divergences and draws them; it does not decide
whether a divergence is acceptable and does not fail. Read the figure and the numbers.

There is no per-framework code below. Every adapter satisfies the same BaseLIF contract
(forward one timestep, reset, membrane), so the driver is written once and each framework
is just a different `lif_factory` result.

INPUT AMPLITUDES ARE NOT ARBITRARY -- both sit BELOW the threshold, deliberately:

  constant_step 0.15  With input gain 1.0 an amplitude above threshold fires on every
                      step, so the membrane trace is a flat line at 0 and the figure shows
                      nothing. At 0.15 the membrane climbs toward 0.15/(1-0.9) = 1.5,
                      crosses, resets, repeats -- a real sawtooth that exercises the decay
                      AND the reset, which is what there is to look at.

  poisson       0.6   At amplitude == threshold the neuron parks exactly ON the threshold,
                      where SpikingJelly's `>=` comparison disagrees with the others' `>`.
                      That shows up as a difference between frameworks that is really a
                      boundary artefact, not a translation error.

WHAT THE MEMBRANE PANEL SHOWS: the POST-reset membrane, i.e. the value carried into the
next timestep, which is 0 immediately after a spike. It therefore never sits above the
threshold line -- the crossing that caused a spike happened before the reset was applied.
Read the raster underneath to see exactly when each neuron fired. (A pre-reset view would
need a state SETTER on BaseLIF to re-seed a shadow neuron each step; the adapters expose a
getter only.)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")  # no display on Colab or a headless box
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from frameworks.adapters import ADAPTERS, lif_factory  # noqa: E402
from skeleton import Settings  # noqa: E402
from skeleton.neuron_spec import describe  # noqa: E402

REFERENCE = "torch"  # cfg.FRAMEWORK key for snnTorch, the comparison baseline

# Line styles chosen so four overlapping traces stay tellable apart: the reference is
# thick and translucent underneath, the others progressively thinner on top.
STYLE = {
    "torch":  {"color": "#1f77b4", "linewidth": 4.5, "linestyle": "-",  "alpha": 0.45},
    "sj":     {"color": "#d62728", "linewidth": 2.8, "linestyle": "--"},
    "norse":  {"color": "#2ca02c", "linewidth": 1.6, "linestyle": ":"},
    # Goldenrod rather than a pink-purple: next to #d62728 red, a thin purple line reads
    # as a second red. Pure yellow is invisible at this width on white.
    "sinabs": {"color": "#DAA520", "linewidth": 1.0, "linestyle": "-."},
}
LABEL = {"torch": "snnTorch", "sj": "SpikingJelly", "norse": "Norse", "sinabs": "Sinabs"}


class Trace(NamedTuple):
    """One framework's response, per timestep."""
    membrane: list[float]   # post-reset: the state carried into the next step
    spikes: list[float]     # 1.0 where it fired


def drive(framework: str, cfg, current: list[float]) -> Trace:
    """One LIF layer of `framework`, fed one timestep at a time.

    Framework-agnostic: BaseLIF hides whether the state comes back from forward(), sits on
    the module, or lives in a registered buffer.
    """
    # The factory is per-layer-slot: it re-reads that slot's configured neuron type and
    # raises unless it is 'lif'. "lif1" stands in for any hidden slot here -- this probes
    # ONE neuron, not the network.
    layer = lif_factory(framework, cfg)("lif1")
    layer.reset()
    membrane, spikes = [], []
    for value in current:
        spike = layer(torch.full((1,), float(value)))
        state = layer.membrane()
        spikes.append(float(spike.detach().sum()))
        membrane.append(0.0 if state is None else float(state.detach().sum()))
    return Trace(membrane, spikes)


def constant_step(steps: int = 90, amplitude: float = 0.15, onset: int = 10) -> list[float]:
    """Zero until `onset`, then constant. The membrane climbs to 1.5, so it crosses the
    1.0 threshold and resets -- a sawtooth, not a flat line."""
    return [0.0 if t < onset else amplitude for t in range(steps)]


def poisson(steps: int = 90, rate: float = 0.15, amplitude: float = 0.6,
            seed: int = 1234) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    draws = torch.rand(steps, generator=generator)
    return [(amplitude if float(d) < rate else 0.0) for d in draws]


PATTERNS = {"constant_step": constant_step, "poisson": poisson}


def draw(name: str, current: list[float], traces: dict[str, Trace], threshold: float,
         out_dir: Path, formats: tuple[str, ...], stamp: str,
         caption: list[str]) -> list[Path]:
    steps = list(range(len(current)))
    figure, (ax_in, ax_mem, ax_raster) = plt.subplots(
        3, 1, figsize=(12, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 3, 1.2]},
    )

    # A step, not a line: the signal is discrete, one value per timestep, instantly on and
    # instantly off. A line plot slopes between samples and makes a single one-step event
    # look like a triangle that rises and falls, which it does not.
    ax_in.step(steps, current, where="mid", color="#555555", linewidth=1.4)
    ax_in.set_ylabel("input\ncurrent")
    ax_in.grid(alpha=0.3)
    ax_in.set_title(f"{name}   —   {len(current)} timesteps, "
                     f"max amplitude {max(current):g}, threshold {threshold:g}",
                     fontsize=11, loc="left")

    for framework, trace in traces.items():
        ax_mem.plot(steps, trace.membrane, label=LABEL.get(framework, framework),
                    **STYLE.get(framework, {}))
    ax_mem.axhline(threshold, color="#999999", linestyle="--", linewidth=1.0)
    ax_mem.annotate(f"threshold {threshold:g}", xy=(0.995, threshold),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=8.5, color="#666666")
    ax_mem.set_ylabel("membrane potential\n(after reset)")
    ax_mem.grid(alpha=0.3)
    ax_mem.legend(loc="upper left", frameon=False, fontsize=9, ncol=4)

    for row, (framework, trace) in enumerate(traces.items()):
        fired = [t for t, s in zip(steps, trace.spikes) if s > 0]
        ax_raster.scatter(fired, [row] * len(fired), s=48, marker="|",
                          color=STYLE.get(framework, {}).get("color", "#333333"))
    ax_raster.set_yticks(range(len(traces)))
    ax_raster.set_yticklabels([LABEL.get(f, f) for f in traces], fontsize=8.5)
    ax_raster.set_ylim(-0.6, len(traces) - 0.4)
    ax_raster.set_xlabel("timestep")
    ax_raster.set_ylabel("spikes")
    ax_raster.grid(alpha=0.3, axis="x")

    figure.text(0.01, 0.005, "\n".join(caption), fontsize=7.6, color="#555555",
                va="bottom", linespacing=1.5)
    figure.suptitle("Neuron equivalence — one LIF layer per framework, identical input",
                    fontsize=13, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        path = out_dir / f"EQ_{name}.{suffix}"
        figure.savefig(path, dpi=150)
        written.append(path)
    plt.close(figure)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default=None, metavar="exN",
                         help="write figures to <results-root>/exN/plots")
    parser.add_argument("--results-root", default="experiments")
    parser.add_argument("--out-dir", default=None, help="write figures here instead")
    parser.add_argument("--formats", default="png", help="e.g. png or png,pdf")
    args = parser.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(args.results_root) / args.experiment / "plots" if args.experiment
               else Path("outputs") / "equivalence")
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    stamp = datetime.now().isoformat(timespec="seconds")

    cfg = Settings()
    cfg.DEVICE = "cpu"
    cfg.apply_dataset_shape(sensor_h=34, sensor_w=34, in_channels=2, num_classes=10)

    print("=" * 78)
    print("NEURON EQUIVALENCE — figures and numbers, no verdict")
    print("=" * 78)

    flat = describe(cfg)
    width = max(len(k) for k in flat)
    print("\nconfigured neuron (network_architecture.yaml):")
    for key in sorted(flat):
        print(f"    {key:<{width}} = {flat[key]}")

    threshold = float(flat.get("snntorch.threshold", 1.0))
    all_written: list[Path] = []

    for name, build in PATTERNS.items():
        current = build()
        print(f"\n{'-' * 78}\n{name}   ({len(current)} steps, max {max(current):g}, "
              f"threshold {threshold:g})\n{'-' * 78}")

        traces: dict[str, Trace] = {}
        for framework in ADAPTERS:
            try:
                traces[framework] = drive(framework, cfg, current)
            except Exception as exc:
                print(f"  {LABEL.get(framework, framework):14} could not run — "
                      f"{type(exc).__name__}: {exc}")

        if not traces:
            print("  no framework produced a trace; skipping this pattern")
            continue

        print(f"\n  {'framework':14} {'spikes':>7} {'peak V':>9} "
              f"{'max |dV| vs ref':>17} {'spike diffs':>12}")
        reference = traces.get(REFERENCE)
        for framework, trace in traces.items():
            fired = int(sum(trace.spikes))
            peak = max(trace.membrane) if trace.membrane else 0.0
            if reference is None or framework == REFERENCE:
                d_v, d_s = "reference", "reference"
            else:
                d_v = f"{max(abs(a - b) for a, b in zip(trace.membrane, reference.membrane)):.3e}"
                d_s = str(sum(1 for a, b in zip(trace.spikes, reference.spikes) if a != b))
            print(f"  {LABEL.get(framework, framework):14} {fired:>7} {peak:>9.4f} "
                  f"{d_v:>17} {d_s:>12}")

        caption = [
            f"input: {name}, max amplitude {max(current):g} (below threshold {threshold:g} "
            f"on purpose — above it the neuron fires every step and the trace is flat)",
            "membrane shown POST-reset, so it never sits above the threshold line; the "
            "raster below marks the step each neuron actually fired",
            f"reference for the divergence column: {LABEL.get(REFERENCE, REFERENCE)}   |   "
            f"generated {stamp}",
        ]
        written = draw(name, current, traces, threshold, out_dir, formats, stamp, caption)
        all_written.extend(written)
        for path in written:
            print(f"\n  figure -> {path}")

    print(f"\n{'=' * 78}")
    print(f"{len(all_written)} figure(s) written to {out_dir}")
    print("Numbers above are reported, not judged — read them alongside the figure.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
