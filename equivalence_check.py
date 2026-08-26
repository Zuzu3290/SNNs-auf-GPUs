"""
Do the four frameworks' LIF neurons actually compute the same thing?

Drives ONE spiking layer per framework with byte-identical input and compares the
resulting spike train and membrane trajectory against a reference framework, step by
step. This is what makes the unified `neuron:` spec in network_architecture.yaml
trustworthy -- the spec CLAIMS all four blocks describe one neuron, and nothing enforces
that but this script. Run it after any change to a neuron parameter.

    python equivalence_check.py
    python equivalence_check.py --config experiments/ex2/config.yaml --experiment ex2
    python equivalence_check.py --config experiments/ex1/config.yaml --experiment ex1 \
        --results-root /content/drive/MyDrive/snn_runs --formats png,pdf

It MEASURES and does not judge: no pass/fail verdict, always exits 0. One threshold
cannot serve both uses -- ex1 forces the neurons to agree (so ~1e-07 means the
translation worked), while ex2 deliberately varies one of them (so a large deviation
IS the result). The hard gate on agreement lives in tests/unit_adapters.py and
tests/unit_shared_net.py instead.

There is no per-framework code below. Every adapter satisfies the same BaseLIF contract
(forward one timestep, reset, membrane), so the driver is written once and each framework
is just a different `lif_factory` result.

INPUT AMPLITUDES ARE NOT ARBITRARY -- both sit BELOW the threshold, deliberately:

  constant_step 0.15  With input gain 1.0, an amplitude above threshold fires on every
                      step, so the membrane trace is flat at 0 and the test passes while
                      proving nothing -- four broken neurons would also agree on a flat
                      line. At 0.15 the membrane climbs toward 0.15/(1-0.9) = 1.5,
                      crosses, resets, repeats: a real sawtooth exercising decay AND reset.

  poisson       0.6   At amplitude == threshold the neuron parks exactly ON the threshold,
                      where SpikingJelly's `>=` disagrees with the others' `>`. That
                      produces a failure that looks like a bad translation but is really
                      a boundary artefact.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")  # no display on Colab or a headless box
import matplotlib.pyplot as plt  # noqa: E402
import torch

from frameworks.adapters import ADAPTERS, lif_factory
from skeleton.cli import add_common_args, build, run_banner
from skeleton import Settings
from skeleton.neuron_spec import describe

TOLERANCE = 1e-6
REFERENCE = "torch"  # cfg.FRAMEWORK key for snnTorch


def parse_args():
    """--config, --experiment, --results-root, --formats.

    No --framework: this builds ALL four and compares them, so choosing one would
    defeat the point. No --seed: the only randomness is the poisson pattern, which
    carries its own fixed seed so every framework receives byte-identical input.
    No --device: CPU is hardcoded below -- see make_cfg.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # --results-root IS offered: this writes FIGURES, and on Colab they have to be able
    # to land on mounted Drive rather than a /content that disappears with the runtime.
    # --cache-root is not, because no dataset or cache is touched here.
    add_common_args(parser, framework=False, seed=False, cache_root=False)
    parser.add_argument(
        "--formats", default="png", metavar="LIST",
        help="figure formats, comma-separated: png (default), pdf, svg. "
             "e.g. --formats png,pdf -- PDF is vector, for a written report.",
    )
    return parser.parse_args()


def make_cfg(cfg: Settings) -> Settings:
    """CPU, always, and not configurable.

    This drives ONE neuron for 90 timesteps. There is no training, no dataset and no
    batch -- a GPU would add transfer overhead and float non-determinism to a test
    whose entire purpose is exact comparison, and would make the result depend on
    which machine ran it. Hardcoded rather than exposed as a flag so there is nothing
    to get wrong.
    """
    cfg.DEVICE = "cpu"
    cfg.apply_dataset_shape(sensor_h=34, sensor_w=34, in_channels=2, num_classes=10)
    return cfg


def drive(framework: str, cfg, xs: list[float]) -> tuple[list[float], list[float]]:
    """One LIF layer of `framework`, fed `xs` one timestep at a time.

    Framework-agnostic: BaseLIF hides whether the state comes back from forward(), sits
    on the module, or lives in a buffer.
    """
    layer = lif_factory(framework, cfg)("lif1")
    layer.reset()
    spikes, mems = [], []
    for x in xs:
        spk = layer(torch.full((1,), float(x)))
        mem = layer.membrane()
        spikes.append(float(spk.detach().sum()))
        mems.append(float("nan") if mem is None else float(mem.detach().sum()))
    return spikes, mems


def constant_step(T: int = 90, amplitude: float = 0.15, onset: int = 10) -> list[float]:
    """Zero until `onset`, then constant. Membrane climbs to 1.5, so it crosses the 1.0
    threshold and resets -- a sawtooth, not a flat line."""
    return [0.0 if t < onset else amplitude for t in range(T)]


def poisson(T: int = 90, rate: float = 0.15, amplitude: float = 0.6,
            seed: int = 1234) -> list[float]:
    generator = torch.Generator().manual_seed(seed)
    draws = torch.rand(T, generator=generator)
    return [(amplitude if float(d) < rate else 0.0) for d in draws]


PATTERNS = {"constant_step": constant_step(), "poisson": poisson()}


# Line styles chosen so four overlapping traces stay tellable apart: the reference is
# thick and translucent underneath, the others progressively thinner on top.
STYLE = {
    "torch":  {"color": "#1f77b4", "linewidth": 4.5, "linestyle": "-", "alpha": 0.45},
    "sj":     {"color": "#d62728", "linewidth": 2.8, "linestyle": "--"},
    "norse":  {"color": "#2ca02c", "linewidth": 1.6, "linestyle": ":"},
    # Goldenrod rather than pink-purple: beside #d62728 red, a thin purple line reads as
    # a second red. Pure yellow is invisible at this width on white.
    "sinabs": {"color": "#DAA520", "linewidth": 1.0, "linestyle": "-."},
}
LABEL = {"torch": "snnTorch", "sj": "SpikingJelly", "norse": "Norse", "sinabs": "Sinabs"}


def draw(pattern_name: str, xs: list[float], traces: dict, threshold: float,
         plots_dir: Path, caption: list[str],
         formats: tuple[str, ...] = ("png",)) -> list[Path]:
    """Three stacked panels: input current, membrane trajectory, spike raster.

    The membrane shown is POST-reset -- the value carried into the next timestep, which
    is 0 straight after a spike. It therefore never sits above the threshold line; the
    crossing that caused the spike happened before the reset was applied. That is what
    the raster underneath is for: it marks the step each neuron actually fired.
    """
    if not traces:
        return []
    steps = list(range(len(xs)))
    figure, (ax_in, ax_mem, ax_raster) = plt.subplots(
        3, 1, figsize=(12, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [1, 3, 1.2]},
    )

    # A step, not a line: the signal is discrete, one value per timestep, instantly on
    # and instantly off. A line plot slopes between samples and makes a single one-step
    # event look like a triangle that rises and falls, which it does not.
    ax_in.step(steps, xs, where="mid", color="#555555", linewidth=1.4)
    ax_in.set_ylabel("input\ncurrent")
    ax_in.grid(alpha=0.3)
    ax_in.set_title(f"{pattern_name}   —   T={len(xs)}, max amplitude {max(xs):g}, "
                     f"threshold {threshold:g}", fontsize=11, loc="left")

    for framework, (_spikes, mems) in traces.items():
        ax_mem.plot(steps, mems, label=LABEL.get(framework, framework),
                    **STYLE.get(framework, {}))
    ax_mem.axhline(threshold, color="#999999", linestyle="--", linewidth=1.0)
    ax_mem.annotate(f"threshold {threshold:g}", xy=(0.995, threshold),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=8.5, color="#666666")
    ax_mem.set_ylabel("membrane potential\n(after reset)")
    ax_mem.grid(alpha=0.3)
    ax_mem.legend(loc="upper left", frameon=False, fontsize=9, ncol=4)

    for row, (framework, (spikes, _mems)) in enumerate(traces.items()):
        fired = [t for t, spike in zip(steps, spikes) if spike > 0]
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

    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        path = plots_dir / f"EQ_{pattern_name}.{suffix}"
        figure.savefig(path, dpi=150)
        written.append(path)
    plt.close(figure)
    return written


def main() -> int:
    args = parse_args()
    cfg, _wf, info = build(args)
    make_cfg(cfg)
    # device already shows as cpu in the banner -- make_cfg pinned it, see its docstring
    print(run_banner("equivalence_check.py", cfg, info, writes_results=False))

    print("=" * 78)
    print("NEURON EQUIVALENCE CHECK".center(78))
    print(f"reference = {REFERENCE}   yardstick = {TOLERANCE:g} (not enforced)".center(78))
    print("=" * 78)

    print("\nconfigured neuron (network_architecture.yaml):")
    flat = describe(cfg)
    width = max(len(k) for k in flat)
    for key in sorted(flat):
        print(f"    {key:<{width}} = {flat[key]}")

    # Where the figures go, and in which formats. plots_dir comes from the shared CLI,
    # so --experiment / --results-root route it the same way they route every other
    # output -- which is what lets a Colab run put figures on mounted Drive.
    plots_dir = Path(info["plots_dir"])
    formats = tuple(f.strip().lstrip(".") for f in args.formats.split(",") if f.strip())
    stamp = datetime.now().isoformat(timespec="seconds")
    written: list[Path] = []

    deviations: list[tuple] = []
    for pattern_name, xs in PATTERNS.items():
        print(f"\n{'-' * 78}\npattern: {pattern_name}   (T={len(xs)}, "
              f"max amplitude {max(xs):g}, threshold 1.0)\n{'-' * 78}")

        traces = {}
        for framework in ADAPTERS:
            try:
                traces[framework] = drive(framework, cfg, xs)
            except Exception as exc:
                print(f"  {framework:14} ERROR {type(exc).__name__}: {exc}")
                deviations.append((pattern_name, framework, float("inf"), float("inf"), False))

        if REFERENCE not in traces:
            print(f"  reference {REFERENCE} unavailable -- cannot compare")
            continue

        ref_spikes, ref_mems = traces[REFERENCE]
        fired = int(sum(ref_spikes))
        print(f"  reference fired {fired} spike(s) over {len(xs)} steps"
              + ("   <-- this pattern never crosses threshold, so it says nothing "
                 "about reset" if fired == 0 else ""))

        print(f"\n  {'framework':14} {'max |d spikes|':>15} {'max |d membrane|':>18}"
              f"  {'vs reference':>14}")
        for framework, (spikes, mems) in traces.items():
            d_spk = max(abs(a - b) for a, b in zip(spikes, ref_spikes))
            d_mem = max(abs(a - b) for a, b in zip(mems, ref_mems))
            within = d_spk <= TOLERANCE and d_mem <= TOLERANCE
            print(f"  {framework:14} {d_spk:>15.3e} {d_mem:>18.3e}"
                  f"  {'agrees' if within else 'DIFFERS':>14}")
            deviations.append((pattern_name, framework, d_spk, d_mem, within))

        for figure_path in draw(
            pattern_name, xs, traces, threshold=1.0,
            plots_dir=plots_dir, formats=formats,
            caption=[
                f"input: {pattern_name}, max amplitude {max(xs):g} -- below the threshold "
                "on purpose; above it the neuron fires every step and the trace is flat",
                "membrane shown POST-reset, so it never sits above the threshold line; the "
                "raster below marks the step each neuron actually fired",
                f"divergence reference: {REFERENCE}   |   yardstick {TOLERANCE:g}, "
                f"reported not enforced   |   generated {stamp}",
            ],
        ):
            written.append(figure_path)
            print(f"\n  figure -> {figure_path}")

    # ---- no verdict, and always exit 0 -------------------------------------------
    #
    # Deliberate, and the same choice the SNNs_2 pipeline made. One threshold cannot
    # serve both experiments this script is run for:
    #
    #   ex1  forces all four neurons to agree, so ~1e-07 means the translation worked
    #        and anything larger is a bug
    #   ex2  deliberately gives sinabs its out-of-box behaviour (no leak, multi-spike,
    #        subtract reset), so a LARGE deviation is the experiment's result, not a
    #        failure -- a gate here would report the finding as an error
    #
    # So this measures and reports; deciding what a deviation means is the reader's.
    # The agreement property still has a hard gate, in tests/unit_adapters.py and
    # tests/unit_shared_net.py, which is where CI should look.
    print("\n" + "=" * 78)
    print("MEASURED, NOT JUDGED".center(78))
    print("=" * 78)
    agreeing = [d for d in deviations if d[4]]
    differing = [d for d in deviations if not d[4]]
    print(f"  reference        : {REFERENCE}")
    print(f"  yardstick        : {TOLERANCE:g}  (a useful 'as close as float32 allows',")
    print("                      NOT a pass/fail threshold -- nothing is enforced here)")
    print(f"  within yardstick : {len(agreeing)}/{len(deviations)} framework-patterns")
    print(f"  figures          : {len(written)} written to {plots_dir}")
    if differing:
        print("  differing        : "
              + ", ".join(f"{fw} ({pattern})" for pattern, fw, _, _, _ in differing))
        print("\n  If this config was meant to make the neurons agree, that is a bug to")
        print("  chase. If it deliberately varies one framework's neuron, this IS the")
        print("  result -- read the deviations above against what the config asked for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
