"""Build the network and report what it actually is, without touching a dataset.

    python check_network.py
    python check_network.py --config config/ex2.yaml --framework sinabs --seed 1
    python check_network.py --all

Feeds a dummy tensor of the right shape, so it needs no download, no GPU and no cache.
Run it after changing the architecture or the neuron spec, or when adding a framework,
to confirm that:

  * every layer's output shape is what you expect
  * the flatten width was MEASURED, and agrees with Settings.compute_fc_in()
  * with one seed, every framework starts from byte-identical conv/linear weights --
    which is what makes a cross-framework comparison mean anything
  * the neuron each framework built is the one the config asked for

Ported from the SNNs_2 comparison pipeline, onto this pipeline's Settings and CLI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

from frameworks.adapters import lif_factory
from frameworks.adapters.base import BaseLIF
from frameworks.spiking_net import build_network
from skeleton.cli import add_common_args, build, run_banner
from skeleton.seeding import param_report, seed_model_init, verify_cross_framework_init
from skeleton.snn_config import FW_TO_CFG_KEY

FRAMEWORKS = list(FW_TO_CFG_KEY)


def apply_shape_without_download(cfg) -> str:
    """Give cfg the selected dataset's real sensor size and class count.

    DATASET_REGISTRY stores both as plain data, so this needs no download and no
    tonic dataset object -- which is what lets this script run on a laptop with no
    data present. Deliberately does NOT go through resolve_dataset_entry: that would
    prompt when dataset.name is unset, and a shape check should never block on input.
    """
    from event_data_workflow.dataset_registry import lookup_dataset

    if not cfg.DATASET_NAME:
        # Settings.NUM_CLASSES has no default -- this script's own placeholder, only for
        # inspecting the conv block's shape with no dataset picked.
        cfg.NUM_CLASSES = 10
        return (f"convolution: block ({cfg.SENSOR_H}x{cfg.SENSOR_W}, "
                f"{cfg.NUM_CLASSES} classes) -- no dataset.name set")
    entry = lookup_dataset(cfg.DATASET_NAME)
    width, height, channels = entry["sensor_size"]
    cfg.apply_dataset_shape(sensor_h=height, sensor_w=width, in_channels=channels,
                            num_classes=entry["num_classes"])
    return (f"{entry['name']} ({cfg.SENSOR_H}x{cfg.SENSOR_W}, "
            f"{cfg.NUM_CLASSES} classes) -- from the registry, nothing downloaded")


def build_one(framework: str, cfg):
    seed_model_init(cfg.SEED)
    return build_network(lif_factory(framework, cfg), cfg)


def describe_layers(net, cfg, batch: int) -> None:
    """One line per layer, with the tensor shape as it leaves that layer."""
    print("=" * 74)
    print("layer by layer, shapes for ONE timestep")
    print("=" * 74)

    activation = torch.zeros(batch, cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W)
    print(f"  {'input':<30} {tuple(activation.shape)}")
    with torch.no_grad():
        for layer in net.layers:
            activation = layer(activation)
            label = type(layer).__name__
            if isinstance(layer, nn.Conv2d):
                label += f"({layer.in_channels}->{layer.out_channels}, k{layer.kernel_size[0]})"
            elif isinstance(layer, nn.MaxPool2d):
                label += f"({layer.kernel_size})"
            elif isinstance(layer, nn.Linear):
                label += f"({layer.in_features}->{layer.out_features})"
            print(f"  {label:<30} {tuple(activation.shape)}")
    net.reset()

    linear = [m for m in net.layers if isinstance(m, nn.Linear)][0]
    print()
    print(f"  flatten width measured by dummy forward : {linear.in_features}")
    print(f"  Settings.compute_fc_in() says           : {cfg.FC_IN}")
    print(f"  agree                                   : {linear.in_features == cfg.FC_IN}")


def full_forward(net, cfg, batch: int, timesteps: int) -> None:
    print()
    print("=" * 74)
    print("full forward pass over all timesteps")
    print("=" * 74)
    data = torch.zeros(timesteps, batch, cfg.IN_CHANNELS, cfg.SENSOR_H, cfg.SENSOR_W)
    with torch.no_grad():
        out = net(data)
    print(f"  input  {tuple(data.shape)}   [T, batch, C, H, W]")
    print(f"  output {tuple(out.shape)}   [T, batch, num_classes]")


def compare_all(cfg, batch: int) -> bool:
    """Every framework under one seed, side by side. This is the real check."""
    rows, reports = [], {}
    for framework in FRAMEWORKS:
        net = build_one(framework, cfg)
        report = param_report(net)
        reports[framework] = report
        neuron = net.lif_layers()[0].describe()
        linear = [m for m in net.layers if isinstance(m, nn.Linear)][0]
        rows.append({
            "framework": framework,
            "fingerprint": report["shared_fingerprint"],
            "params": report["total_trainable"],
            "flatten": linear.in_features,
            "neuron": neuron,
        })

    print("=" * 74)
    print(f"ALL FRAMEWORKS, seed {cfg.SEED}")
    print("=" * 74)
    print(f"{'framework':<12}{'fingerprint':>18}{'params':>10}{'flatten':>10}")
    print("-" * 74)
    for row in rows:
        print(f"{row['framework']:<12}{row['fingerprint']:>18}"
              f"{row['params']:>10,}{row['flatten']:>10}")

    print()
    print("=" * 74)
    print("NEURON ACTUALLY BUILT, per framework")
    print("=" * 74)
    print("These are SUPPOSED to differ in NAMING between frameworks -- each one uses its")
    print("own units. They must be the values THIS config asked for. Read them against")
    print("network_architecture.yaml's neuron: block.")
    print()
    for row in rows:
        print(f"  {row['framework']}")
        for key, value in row["neuron"].items():
            print(f"      {key:<22}{value}")
        print()

    passed, problems = verify_cross_framework_init(reports)
    print("=" * 74)
    print(f"  {'PASS' if passed else 'FAIL'}  every framework starts from identical weights")
    for problem in problems:
        print(f"        {problem}")
    print("=" * 74)
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("\nThe frameworks do NOT start from the same place. Any accuracy comparison")
        print("between them would be meaningless until this is fixed.")
    return passed


def parse_args():
    """roots=False: this script writes no files and reads no cache, so --results-root
    and --cache-root would be flags that do nothing. --experiment stays as a label for
    the banner, saying which experiment's config is being checked."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_common_args(parser, roots=False)
    parser.add_argument("--all", action="store_true",
                        help="compare every framework instead of inspecting one")
    parser.add_argument("--batch", type=int, default=4, help="dummy batch size")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="override T for the forward pass (default: framing.n_time_bins)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg, wf, info = build(args)
    # A label, not an output path: this script writes no files, so --experiment only
    # says which experiment's config you are checking.
    shape_note = apply_shape_without_download(cfg)
    print(run_banner("check_network.py", cfg, info, writes_results=False,
                     extra={"shape": shape_note}))
    print()

    if args.all:
        return 0 if compare_all(cfg, args.batch) else 1

    framework = cfg.FRAMEWORK
    net = build_one(framework, cfg)
    describe_layers(net, cfg, args.batch)
    full_forward(net, cfg, args.batch, args.timesteps or wf.N_TIME_BINS)

    report = param_report(net)
    print()
    print(f"  trainable parameters : {report['total_trainable']:,}")
    print(f"  shared fingerprint   : {report['shared_fingerprint']}")
    print()
    print("  neuron actually built:")
    for key, value in net.lif_layers()[0].describe().items():
        print(f"      {key:<22}{value}")
    print()
    print("  (run with --all to check every framework starts from identical weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
