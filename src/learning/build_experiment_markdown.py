"""
Build a single Markdown file with per-experiment plots + parameters + brief description.
Same content as the notebook, but as a scrollable MD with PNG images instead of executed cells.

Output:
    outputs/experiments_overview/README.md          (main file)
    outputs/experiments_overview/img/*.png          (per-experiment plots)

Usage:
    python src/learning/build_experiment_markdown.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse the EXPERIMENTS dict from the notebook builder so we have a single source of truth
sys.path.insert(0, str(Path(__file__).parent))
from build_experiment_notebook import EXPERIMENTS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs" / "experiments_overview"
IMG_DIR = OUT_DIR / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

TORCH_COLOR = "#1f77b4"
NORSE_COLOR = "#ff7f0e"


def color_for(framework: str) -> str:
    return TORCH_COLOR if framework == "SNNTorch" else NORSE_COLOR


def plot_training_curves(exp, save_path: Path):
    color = color_for(exp["framework"])
    n_ep = len(exp["train_loss"])
    epochs = np.arange(1, n_ep + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, exp["train_loss"], marker="o", color=color, linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training loss"); axes[0].grid(alpha=0.3); axes[0].set_xticks(epochs)

    axes[1].plot(epochs, exp["train_acc"], marker="o", color=color, linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Training accuracy (%)")
    axes[1].set_title("Training accuracy"); axes[1].grid(alpha=0.3); axes[1].set_xticks(epochs)
    if exp["status"] == "ok":
        axes[1].set_ylim(70, 100)

    axes[2].plot(epochs, exp["spike_rate"], marker="o", color=color, linewidth=2)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Average spike rate")
    axes[2].set_title("Spike rate"); axes[2].grid(alpha=0.3); axes[2].set_xticks(epochs)

    if exp["status"] == "crashed":
        for ax in axes:
            ax.text(0.5, 0.95, "(only one epoch completed before crash)",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    color="darkred", style="italic")
    elif exp["status"] == "dead_neurons":
        for ax in axes:
            ax.text(0.5, 0.95, "(network failed to learn — neurons never fired)",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    color="darkred", style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_test_metrics(exp, save_path: Path):
    color = color_for(exp["framework"])
    labels = ["Test accuracy (%)", "Energy per sample (pJ)",
              "Avg spikes per sample", "Avg latency per sample (ms)"]
    values = [exp["test_acc"], exp["energy_per_sample"],
              exp["avg_spikes_per_sample"], exp["avg_latency_ms"]]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    bars = ax.barh(labels, values, color=color, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}" if val < 1000 else f"{val:.0f}",
                va="center", fontsize=11, fontweight="bold")
    ax.set_title("Test-time performance")
    ax.set_xlim(0, max(values) * 1.18)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(exp, save_path: Path):
    color = color_for(exp["framework"])
    classes = list(exp["per_class_f1"].keys())
    f1s = list(exp["per_class_f1"].values())

    fig, ax = plt.subplots(figsize=(11, 4))
    bars = ax.bar(classes, f1s, color=color, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("Digit class")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 (test set)")
    ax.set_xticks(classes)
    ax.set_ylim(min(f1s) - 0.02, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def build_params_table(params: dict) -> str:
    rows = ["| Parameter | Value |", "|---|---|"]
    for k, v in params.items():
        rows.append(f"| {k} | {v} |")
    return "\n".join(rows)


def build_markdown() -> str:
    sections = []

    sections.append(
        "# Per-experiment results — N-MNIST Conv-SNN training\n\n"
        "This document shows the results of each training experiment one after another, "
        "using the same template every time. Each section has:\n\n"
        "1. A short description of what was tried\n"
        "2. The configuration parameters used\n"
        "3. Training curves (loss, accuracy, spike rate) per epoch\n"
        "4. Test-time performance (accuracy, energy per sample, spikes per sample, latency)\n"
        "5. Per-class F1 score on the test set (when available)\n\n"
        "The architecture is the same shallow Convolutional Spiking Neural Network throughout: "
        "two convolution layers (12 and 32 channels, 5×5 kernels) with 2×2 max-pooling, a "
        "fully-connected output layer with 10 classes, and Leaky-Integrate-and-Fire neurons. "
        "Trained on N-MNIST (60,000 training samples, 10,000 test samples), each recording is "
        "framed into 16 time bins of event accumulations.\n\n"
        "Two SNN frameworks are exercised: **SNNTorch** (plots in blue) and **Norse** (plots in orange).\n\n"
        "Scroll through to see how each parameter change affected training and test behavior. "
        "Some runs failed — those sections still show the partial data that was captured before "
        "the failure.\n\n"
        "---"
    )

    for exp in EXPERIMENTS:
        n = exp["n"]
        slug = f"exp{n:02d}"

        # Generate plots
        train_png = IMG_DIR / f"{slug}_training_curves.png"
        plot_training_curves(exp, train_png)

        test_png = None
        if exp["test_acc"] is not None:
            test_png = IMG_DIR / f"{slug}_test_metrics.png"
            plot_test_metrics(exp, test_png)

        f1_png = None
        if exp.get("per_class_f1"):
            f1_png = IMG_DIR / f"{slug}_per_class_f1.png"
            plot_per_class_f1(exp, f1_png)

        # Section markdown
        section = [f"## Experiment {n} — {exp['title']}\n"]
        section.append(exp["config_summary"] + "\n")

        if exp["status"] == "crashed":
            section.append(f"> **Training crashed.** {exp['failure_note']}\n")
        elif exp["status"] == "dead_neurons":
            section.append(f"> **Training failed — dead neurons.** {exp['failure_note']}\n")
        elif exp["status"] == "undertrained":
            section.append(
                "> **Completed but under-trained.** Compare its final training accuracy to other "
                "completed runs to see the effect.\n"
            )

        section.append("### Configuration\n")
        section.append(build_params_table(exp["params"]) + "\n")

        section.append("### Training curves\n")
        section.append(f"![Training curves](img/{train_png.name})\n")

        if test_png:
            section.append("### Test-time performance\n")
            section.append(f"![Test metrics](img/{test_png.name})\n")
        else:
            section.append("*(No test results — training did not complete.)*\n")

        if f1_png:
            section.append("### Per-class F1 (test set)\n")
            section.append(f"![Per-class F1](img/{f1_png.name})\n")

        section.append("---")
        sections.append("\n".join(section))

    return "\n\n".join(sections) + "\n"


def main():
    print(f"Generating per-experiment markdown -> {OUT_DIR}")
    md_text = build_markdown()
    md_path = OUT_DIR / "README.md"
    md_path.write_text(md_text, encoding="utf-8")
    n_pngs = len(list(IMG_DIR.glob("*.png")))
    print(f"Done. Wrote {md_path.name} + {n_pngs} PNGs.")


if __name__ == "__main__":
    main()
