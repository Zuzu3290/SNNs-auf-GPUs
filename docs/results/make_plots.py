"""
Generates comparison plots from the JSON summaries produced by run_benchmark.py.

Usage:
    python docs/results/make_plots.py                    # N-MNIST (default)
    python docs/results/make_plots.py --dataset "ASL-DVS" # matches run_benchmark.py's --dataset
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def dataset_slug(name: str) -> str:
    """Same convention as run_benchmark.py's dataset_slug — must match to find its output."""
    return name.lower().replace(" ", "_").replace("-", "_")


ORDER = ["norse", "torch", "sj", "sinabs", "bindsnet", "spyx"]
COLORS = dict(zip(ORDER, plt.cm.tab10.colors))
LABELS = {
    "norse": "Norse", "torch": "snnTorch", "sj": "SpikingJelly",
    "sinabs": "Sinabs", "bindsnet": "BindsNET", "spyx": "Spyx",
}


def load_results():
    results = {}
    for name in ORDER:
        path = DATA / f"{name}_summary.json"
        if path.exists():
            with open(path) as f:
                results[name] = json.load(f)
    return results


def attach_real_training_energy(results):
    """
    run_one()'s summary JSON doesn't include the per-epoch GPU energy that
    SNNTrainer.train() measures (via GPUStats/NVML) — it's only in the
    {name}_train.csv it writes. Pull it in here rather than re-running training.
    """
    for name, r in results.items():
        csv_path = DATA / f"{name}_train.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        r["train_energy_j"] = sum(float(row["energy_j"]) for row in rows)
        r["train_avg_power_w"] = sum(float(row["avg_power_w"]) for row in rows) / len(rows)


def plot_curve(results, key, title, ylabel, fname):
    plt.figure(figsize=(8, 5))
    for name, r in results.items():
        plt.plot(r[key], label=LABELS[name], color=COLORS[name])
    plt.title(title)
    plt.xlabel("Training iteration")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")


def plot_bar(results, key, title, ylabel, fname, scale=1.0):
    names = list(results.keys())
    values = [results[n][key] * scale for n in names]
    plt.figure(figsize=(8, 5))
    plt.bar([LABELS[n] for n in names], values, color=[COLORS[n] for n in names])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")


def plot_confusion_matrices(results, fname):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, (name, r) in zip(axes, results.items()):
        cm = np.array(r["test_confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(LABELS[name])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground truth")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes[len(results):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=150)
    plt.close()
    print(f"  -> {fname}")


def main():
    global DATA, PLOTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="N-MNIST",
                         help="Must match the --dataset a prior run_benchmark.py run used")
    args = parser.parse_args()
    slug = dataset_slug(args.dataset)

    DATA = HERE / "data" / slug
    PLOTS = HERE / "plots" / slug
    PLOTS.mkdir(parents=True, exist_ok=True)

    results = load_results()
    if not results:
        print(f"No result JSON files found in {DATA} — run `run_benchmark.py --dataset \"{args.dataset}\"` first.")
        return
    print(f"[{args.dataset}] Loaded results for: {list(results.keys())}")
    attach_real_training_energy(results)

    # Classification-only metrics (accuracy, confusion matrix) — only plotted if present,
    # so this doesn't crash on a regression dataset's summaries once those exist.
    has_classification_metrics = all("test_overall_accuracy" in r for r in results.values())

    plot_curve(results, "loss_history", f"{args.dataset} — Training Loss", "Loss", "loss_curves.png")
    plot_curve(results, "spike_rate_history", f"{args.dataset} — Training Spike Rate", "Mean spike rate", "spike_rate_curves.png")
    plot_bar(results, "train_time_s", f"{args.dataset} — Training Wall-Clock Time", "seconds", "train_time.png")
    plot_bar(results, "train_energy_j", f"{args.dataset} — Actual GPU Energy Used During Training (NVML-measured)", "Joules", "train_energy.png")
    plot_bar(results, "train_avg_power_w", f"{args.dataset} — Average GPU Power Draw During Training (NVML-measured)", "Watts", "train_power.png")

    if has_classification_metrics:
        plot_curve(results, "accuracy_history", f"{args.dataset} — Training Accuracy", "Accuracy", "accuracy_curves.png")
        plot_bar(results, "test_overall_accuracy", f"{args.dataset} — Test Accuracy", "Accuracy", "test_accuracy.png", scale=100)
        plot_bar(results, "test_energy_per_sample_pj", f"{args.dataset} — Energy per Sample (neuromorphic model)", "pJ / sample", "test_energy.png")
        plot_bar(results, "test_avg_latency_per_sample_ms", f"{args.dataset} — Inference Latency per Sample", "ms / sample", "test_latency.png")
        plot_bar(results, "test_avg_firing_rate_hz", f"{args.dataset} — Average Firing Rate", "Hz", "test_firing_rate.png")
        if all(r.get("test_framework_ratio") is not None for r in results.values()):
            plot_bar(results, "test_framework_ratio",
                     f"{args.dataset} — Framework Ratio (input activity / output spikes)",
                     "ratio", "framework_ratio.png")
        plot_confusion_matrices(results, "confusion_matrices.png")
    else:
        print("  (skipping accuracy/confusion-matrix plots — not present, likely a regression dataset)")

    print("\nAll plots written to", PLOTS)


if __name__ == "__main__":
    main()
