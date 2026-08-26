"""
Do the four frameworks' LIF neurons actually compute the same thing?

Drives ONE spiking layer per framework with byte-identical input and compares the
resulting spike train and membrane trajectory against a reference framework, step by
step. This is what makes the unified `neuron:` spec in network_architecture.yaml
trustworthy -- the spec CLAIMS all four blocks describe one neuron, and nothing enforces
that but this script. Run it after any change to a neuron parameter.

    python equivalence_check.py
    python equivalence_check.py --config config/ex2.yaml --experiment ex2

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from frameworks.adapters import ADAPTERS, lif_factory
from skeleton.cli import add_common_args, build, run_banner
from skeleton import Settings
from skeleton.neuron_spec import describe

TOLERANCE = 1e-6
REFERENCE = "torch"  # cfg.FRAMEWORK key for snnTorch


def parse_args():
    """--config and --experiment only.

    No --framework: this builds ALL four and compares them, so choosing one would
    defeat the point. No --seed: the only randomness is the poisson pattern, which
    carries its own fixed seed so every framework receives byte-identical input.
    No --device: CPU is hardcoded below -- see make_cfg.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # roots=False as well: this script writes no files, so --results-root and
    # --cache-root would be flags that do nothing. --experiment stays, as a label for
    # the banner saying which experiment's config is being checked.
    return add_common_args(parser, framework=False, seed=False, roots=False).parse_args()


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
    if differing:
        print("  differing        : "
              + ", ".join(f"{fw} ({pattern})" for pattern, fw, _, _, _ in differing))
        print("\n  If this config was meant to make the neurons agree, that is a bug to")
        print("  chase. If it deliberately varies one framework's neuron, this IS the")
        print("  result -- read the deviations above against what the config asked for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
