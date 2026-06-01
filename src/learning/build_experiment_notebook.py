"""
Build a Jupyter notebook that shows each experiment's results in the same template,
one after another. Scrolling through reveals what's changing parameter-by-parameter,
no comparison/winner framing — just per-experiment performance views.

Output: outputs/experiments_notebook.ipynb (with executed outputs embedded)

Usage:
    python src/learning/build_experiment_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = OUT_DIR / "experiments_notebook.ipynb"


# ────────────────────────────────────────────────────────────────────────────
# Experiment data — full per-experiment records (training + test + config)
# ────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "n": 1,
        "framework": "SNNTorch",
        "title": "SNNTorch · standard training settings",
        "config_summary": (
            "Cross-entropy loss on summed spikes. Hard reset. No mixed precision. "
            "5 training epochs at constant learning rate."
        ),
        "params": {
            "Framework": "SNNTorch", "Threshold": 0.5, "Membrane decay (beta)": 0.95,
            "Loss function": "cross_entropy", "Reset mode": "zero (hard)",
            "Surrogate gradient": "atan", "Optimizer": "adam",
            "Learning rate": 0.001, "LR scheduler": "none",
            "Weight decay": 0.0001, "Mixed precision (AMP)": "false",
            "Batch size": 64, "Timesteps": 16, "Epochs": 5,
            "Augmentation": "±10° rotation",
        },
        "train_loss": [0.3239, 0.0964, 0.0802, 0.0705, 0.0643],
        "train_acc":  [90.04, 96.86, 97.53, 97.77, 97.98],
        "spike_rate": [0.1131, 0.1369, 0.1322, 0.1356, 0.1422],
        "lr":         [0.001, 0.001, 0.001, 0.001, 0.001],
        "duration":   [968, 124, 124, 124, 126],
        "test_acc": 98.25, "energy_per_sample": 86.81,
        "avg_spikes_per_sample": 25.0, "avg_latency_ms": 0.327,
        "per_class_f1": None,
        "status": "ok",
    },
    {
        "n": 2,
        "framework": "Norse",
        "title": "Norse · standard training settings",
        "config_summary": (
            "Same shared settings as the SNNTorch standard run, but using Norse's "
            "LIF cell with inverse-time-constant parametrization. Hard reset (Norse default)."
        ),
        "params": {
            "Framework": "Norse", "Threshold": 0.5, "tau_mem_inv": "50.0 Hz",
            "Loss function": "cross_entropy", "Reset mode": "zero (hard, only option for Norse)",
            "Surrogate gradient": "SuperSpike (Norse default)", "Optimizer": "adam",
            "Learning rate": 0.001, "LR scheduler": "none",
            "Weight decay": 0.0001, "Mixed precision (AMP)": "false",
            "Batch size": 64, "Timesteps": 16, "Epochs": 5,
            "Augmentation": "±10° rotation",
        },
        "train_loss": [0.4090, 0.1601, 0.1304, 0.1142, 0.1026],
        "train_acc":  [86.82, 95.14, 95.99, 96.53, 96.81],
        "spike_rate": [0.0718, 0.0916, 0.0950, 0.0986, 0.0983],
        "lr":         [0.001, 0.001, 0.001, 0.001, 0.001],
        "duration":   [1208, 160, 165, 161, 159],
        "test_acc": 97.31, "energy_per_sample": 55.76,
        "avg_spikes_per_sample": 16.0, "avg_latency_ms": 0.409,
        "per_class_f1": None,
        "status": "ok",
    },
    {
        "n": 3,
        "framework": "Norse",
        "title": "Norse · mixed precision + 4 parallel data workers",
        "config_summary": (
            "Stress test: turned on mixed precision and used 4 parallel data loader workers "
            "with adaptive caching. Run crashed at the start of epoch 2 due to host RAM exhaustion."
        ),
        "params": {
            "Framework": "Norse", "Threshold": 0.5, "tau_mem_inv": "50.0 Hz",
            "Loss function": "cross_entropy", "Reset mode": "zero",
            "Surrogate gradient": "SuperSpike", "Optimizer": "adam",
            "Learning rate": 0.001, "LR scheduler": "cosine",
            "Mixed precision (AMP)": "TRUE", "Data workers": 4,
            "Batch size": 64, "Timesteps": 16, "Epochs (planned)": 5,
            "Augmentation": "±10° rotation",
        },
        "train_loss": [0.4214],
        "train_acc":  [86.40],
        "spike_rate": [0.0723],
        "lr":         [0.000905],
        "duration":   [983.52],
        "test_acc": None, "energy_per_sample": None,
        "avg_spikes_per_sample": None, "avg_latency_ms": None,
        "per_class_f1": None,
        "status": "crashed",
        "failure_note": "Out-of-memory crash at the start of epoch 2. Each of the 4 data-loader workers held its own copy of the in-memory cache (~4 GB each), exceeding Colab's 12 GB RAM limit.",
    },
    {
        "n": 4,
        "framework": "Norse",
        "title": "Norse · raised threshold to 1.0",
        "config_summary": (
            "Tested raising the firing threshold from 0.5 to 1.0 (Norse library default). "
            "All neurons stopped firing. Loss stayed at random-chance level throughout."
        ),
        "params": {
            "Framework": "Norse", "Threshold": "1.0 (raised)", "tau_mem_inv": "50.0 Hz",
            "Loss function": "cross_entropy", "Reset mode": "zero",
            "Surrogate gradient": "SuperSpike", "Optimizer": "adam",
            "Learning rate": 0.001, "LR scheduler": "cosine",
            "Mixed precision (AMP)": "false",
            "Batch size": 64, "Timesteps": 16, "Epochs (planned)": 3,
            "Augmentation": "±10° rotation",
        },
        "train_loss": [2.3026, 2.3026],
        "train_acc":  [9.88, 9.92],
        "spike_rate": [0.0000, 0.0002],
        "lr":         [0.001, 0.001],
        "duration":   [1050, 539],
        "test_acc": None, "energy_per_sample": None,
        "avg_spikes_per_sample": None, "avg_latency_ms": None,
        "per_class_f1": None,
        "status": "dead_neurons",
        "failure_note": "Network never learned. Loss locked at 2.3026 (= ln(10), the random-chance value for 10 classes). Spike rate near zero throughout — no neuron ever reached the threshold to fire, so no gradient signal flowed back.",
    },
    {
        "n": 5,
        "framework": "Norse",
        "title": "Norse · threshold reverted to 0.5 (verification, 3 epochs)",
        "config_summary": (
            "Repeat of experiment 4 with the only change being threshold lowered back to 0.5. "
            "Confirms that threshold=1.0 was the sole cause of the previous failure. Stopped at 3 epochs."
        ),
        "params": {
            "Framework": "Norse", "Threshold": 0.5, "tau_mem_inv": "50.0 Hz",
            "Loss function": "cross_entropy", "Reset mode": "zero",
            "Surrogate gradient": "SuperSpike", "Optimizer": "adam",
            "Learning rate": 0.001, "LR scheduler": "cosine",
            "Mixed precision (AMP)": "false",
            "Batch size": 64, "Timesteps": 16, "Epochs": 3,
            "Augmentation": "±10° rotation",
        },
        "train_loss": [0.4622, 0.1979, 0.1616],
        "train_acc":  [85.03, 93.98, 95.03],
        "spike_rate": [0.0728, 0.0960, 0.0948],
        "lr":         [0.000750, 0.000250, 0.000000],
        "duration":   [972.49, 482.78, 489.59],
        "test_acc": 95.71, "energy_per_sample": 53.82,
        "avg_spikes_per_sample": 15.38, "avg_latency_ms": 0.387,
        "per_class_f1": None,
        "status": "ok",
    },
    {
        "n": 6,
        "framework": "SNNTorch",
        "title": "SNNTorch · library-recommended training recipe",
        "config_summary": (
            "Switched to SNNTorch's library-canonical training recipe: mse_count_loss with explicit "
            "per-class firing-rate targets (80% / 20%), soft (subtract) reset, mixed precision on, "
            "threshold raised to library default of 1.0. Cosine learning rate schedule."
        ),
        "params": {
            "Framework": "SNNTorch", "Threshold": "1.0 (library default)",
            "Membrane decay (beta)": 0.95,
            "Loss function": "mse_count_loss (target 80%/20%)",
            "Reset mode": "subtract (soft, SNNTorch native)",
            "Surrogate gradient": "atan", "Optimizer": "adam",
            "Learning rate": 0.001, "LR scheduler": "cosine",
            "Weight decay": 0.0001, "Mixed precision (AMP)": "TRUE",
            "Batch size": 64, "Timesteps": 16, "Epochs": 5,
            "Augmentation": "±10° rotation",
        },
        "train_loss": [0.1422, 0.0902, 0.0815, 0.0741, 0.0691],
        "train_acc":  [91.15, 96.97, 97.57, 97.85, 97.94],
        "spike_rate": [0.2366, 0.2366, 0.2352, 0.2351, 0.2361],
        "lr":         [0.000905, 0.000655, 0.000345, 0.000095, 0.000000],
        "duration":   [964.42, 490.74, 478.49, 480.82, 484.85],
        "test_acc": 98.17, "energy_per_sample": 128.60,
        "avg_spikes_per_sample": 36.74, "avg_latency_ms": 0.329,
        "per_class_f1": {
            0: 0.986, 1: 0.991, 2: 0.981, 3: 0.984, 4: 0.988,
            5: 0.980, 6: 0.986, 7: 0.977, 8: 0.974, 9: 0.969,
        },
        "status": "ok",
    },
    {
        "n": 7,
        "framework": "Norse",
        "title": "Norse · lower learning rate (community recommendation for SuperSpike)",
        "config_summary": (
            "Reduced learning rate from 0.001 to 0.0002 — a recommendation in the Norse community "
            "for stabilizing the SuperSpike surrogate gradient. Combined with cosine decay this left "
            "the network with very small effective updates and produced an under-trained model."
        ),
        "params": {
            "Framework": "Norse", "Threshold": 0.5, "tau_mem_inv": "50.0 Hz",
            "Loss function": "cross_entropy", "Reset mode": "zero",
            "Surrogate gradient": "SuperSpike", "Optimizer": "adam",
            "Learning rate": "0.0002 (lowered)",
            "LR scheduler": "cosine",
            "Weight decay": 0.0001, "Mixed precision (AMP)": "false",
            "Batch size": 64, "Timesteps": 16, "Epochs": 5,
            "Augmentation": "±10° rotation",
        },
        "train_loss": [0.8260, 0.3154, 0.2493, 0.2255, 0.2145],
        "train_acc":  [73.37, 90.65, 92.52, 93.20, 93.57],
        "spike_rate": [0.0389, 0.0611, 0.0671, 0.0717, 0.0712],
        "lr":         [0.000181, 0.000131, 0.000069, 0.000019, 0.000000],
        "duration":   [989.63, 507.99, 492.32, 490.93, 493.29],
        "test_acc": 94.05, "energy_per_sample": 41.03,
        "avg_spikes_per_sample": 11.72, "avg_latency_ms": 0.373,
        "per_class_f1": {
            0: 0.967, 1: 0.981, 2: 0.929, 3: 0.918, 4: 0.948,
            5: 0.933, 6: 0.963, 7: 0.942, 8: 0.904, 9: 0.915,
        },
        "status": "undertrained",
    },
    {
        "n": 8,
        "framework": "SNNTorch",
        "title": "SNNTorch · library-recommended recipe + longer training (8 epochs)",
        "config_summary": (
            "Same library-canonical config as experiment 6, but trained for 8 epochs with cosine "
            "learning-rate decay instead of 5 epochs."
        ),
        "params": {
            "Framework": "SNNTorch", "Threshold": "1.0 (library default)",
            "Membrane decay (beta)": 0.95,
            "Loss function": "mse_count_loss (target 80%/20%)",
            "Reset mode": "subtract (soft)",
            "Surrogate gradient": "atan", "Optimizer": "adam",
            "Learning rate": 0.001, "LR scheduler": "cosine",
            "Weight decay": 0.0001, "Mixed precision (AMP)": "TRUE",
            "Batch size": 64, "Timesteps": 16, "Epochs": 8,
            "Augmentation": "±10° rotation",
        },
        "train_loss": [0.1511, 0.0841, 0.0716, 0.0640, 0.0585, 0.0546, 0.0520, 0.0505],
        "train_acc":  [90.30, 96.80, 97.31, 97.64, 97.77, 97.99, 98.04, 98.11],
        "spike_rate": [0.2377, 0.2424, 0.2419, 0.2429, 0.2430, 0.2432, 0.2431, 0.2431],
        "lr":         [0.000962, 0.000854, 0.000691, 0.000500, 0.000309, 0.000146, 0.000038, 0.000000],
        "duration":   [1048.55, 524.94, 514.51, 511.42, 513.26, 514.43, 513.08, 513.49],
        "test_acc": 98.17, "energy_per_sample": 133.81,
        "avg_spikes_per_sample": 38.23, "avg_latency_ms": 0.342,
        "per_class_f1": {
            0: 0.987, 1: 0.990, 2: 0.984, 3: 0.983, 4: 0.985,
            5: 0.976, 6: 0.987, 7: 0.981, 8: 0.978, 9: 0.964,
        },
        "status": "ok",
    },
    {
        "n": 9,
        "framework": "Norse",
        "title": "Norse · library-recommended membrane time constant + 8 epochs + cosine LR",
        "config_summary": (
            "Raised the inverse membrane time constant from 50 to 100 Hz (Norse's library default), "
            "kept learning rate at the empirically-validated 0.001, trained for 8 epochs with cosine "
            "decay. This combination doubled per-step input gain at the membrane."
        ),
        "params": {
            "Framework": "Norse", "Threshold": 0.5,
            "tau_mem_inv": "100.0 Hz (raised to library default)",
            "Loss function": "cross_entropy", "Reset mode": "zero",
            "Surrogate gradient": "SuperSpike", "Optimizer": "adam",
            "Learning rate": 0.001, "LR scheduler": "cosine",
            "Weight decay": 0.0001, "Mixed precision (AMP)": "false",
            "Batch size": 64, "Timesteps": 16, "Epochs": 8,
            "Augmentation": "±10° rotation",
        },
        "train_loss": [0.2995, 0.1231, 0.0976, 0.0825, 0.0718, 0.0650, 0.0581, 0.0545],
        "train_acc":  [90.60, 96.31, 97.04, 97.41, 97.76, 98.03, 98.21, 98.30],
        "spike_rate": [0.0784, 0.0956, 0.0981, 0.1003, 0.0965, 0.0944, 0.0928, 0.0909],
        "lr":         [0.000962, 0.000854, 0.000691, 0.000500, 0.000309, 0.000146, 0.000038, 0.000000],
        "duration":   [1009.09, 527.59, 528.73, 524.07, 539.35, 528.84, 528.77, 538.55],
        "test_acc": 98.29, "energy_per_sample": 51.87,
        "avg_spikes_per_sample": 14.82, "avg_latency_ms": 0.390,
        "per_class_f1": {
            0: 0.986, 1: 0.990, 2: 0.983, 3: 0.983, 4: 0.987,
            5: 0.983, 6: 0.985, 7: 0.982, 8: 0.974, 9: 0.974,
        },
        "status": "ok",
    },
]


# ────────────────────────────────────────────────────────────────────────────
# Notebook cell templates
# ────────────────────────────────────────────────────────────────────────────
SETUP_CODE = '''\
# Run this cell once at the top.
import matplotlib.pyplot as plt
import numpy as np

TORCH_COLOR = "#1f77b4"
NORSE_COLOR = "#ff7f0e"

def plot_training_curves(epochs_data, framework, status="ok"):
    """3-panel plot: train loss, train accuracy, spike rate per epoch."""
    color = TORCH_COLOR if framework == "SNNTorch" else NORSE_COLOR
    n_ep = len(epochs_data["train_loss"])
    epochs = np.arange(1, n_ep + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, epochs_data["train_loss"], marker="o", color=color, linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3); axes[0].set_xticks(epochs)

    axes[1].plot(epochs, epochs_data["train_acc"], marker="o", color=color, linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Training accuracy (%)")
    axes[1].set_title("Training accuracy")
    axes[1].grid(alpha=0.3); axes[1].set_xticks(epochs)
    if status == "ok":
        axes[1].set_ylim(70, 100)

    axes[2].plot(epochs, epochs_data["spike_rate"], marker="o", color=color, linewidth=2)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Average spike rate")
    axes[2].set_title("Spike rate")
    axes[2].grid(alpha=0.3); axes[2].set_xticks(epochs)

    if status == "crashed":
        for ax in axes:
            ax.text(0.5, 0.95, "(only one epoch completed before crash)",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    color="darkred", style="italic")
    elif status == "dead_neurons":
        for ax in axes:
            ax.text(0.5, 0.95, "(network failed to learn — neurons never fired)",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    color="darkred", style="italic")

    plt.tight_layout()
    plt.show()


def plot_test_metrics(test_acc, energy_per_sample, avg_spikes_per_sample,
                      avg_latency_ms, framework):
    """4 horizontal bars showing test-time performance."""
    color = TORCH_COLOR if framework == "SNNTorch" else NORSE_COLOR
    labels = ["Test accuracy (%)", "Energy per sample (pJ)",
              "Avg spikes per sample", "Avg latency per sample (ms)"]
    values = [test_acc, energy_per_sample, avg_spikes_per_sample, avg_latency_ms]

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
    plt.show()


def plot_per_class_f1(per_class_f1, framework):
    """10 bars, one per class, showing F1 score."""
    color = TORCH_COLOR if framework == "SNNTorch" else NORSE_COLOR
    classes = list(per_class_f1.keys())
    f1s = list(per_class_f1.values())

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
    plt.show()


def show_config(params):
    """Pretty-print config table as Markdown."""
    from IPython.display import Markdown, display
    rows = ["| Parameter | Value |", "|---|---|"]
    for k, v in params.items():
        rows.append(f"| {k} | {v} |")
    display(Markdown("\\n".join(rows)))
'''


INTRO_MD = """\
# Per-experiment results — N-MNIST Conv-SNN training

This notebook shows the results of each training experiment one after another, using the same plot template every time. Each section has:

1. A short description of what was tried
2. The configuration parameters used
3. Training curves (loss, accuracy, spike rate) per epoch
4. Test-time performance (accuracy, energy per sample, spikes per sample, latency)
5. Per-class F1 score on the test set (when available)

The architecture is the same shallow Convolutional Spiking Neural Network throughout: two convolution layers (12 and 32 channels, 5×5 kernels) with 2×2 max-pooling, a fully-connected output layer with 10 classes, and Leaky-Integrate-and-Fire neurons. Trained on N-MNIST (60,000 training samples, 10,000 test samples), each recording is framed into 16 time bins of event accumulations.

Two SNN frameworks are exercised: **SNNTorch** (blue) and **Norse** (orange).

Scroll through to see how each parameter change affected the model's training and test behavior. Some runs failed — those sections still show the partial data that was captured before the failure.
"""


# ────────────────────────────────────────────────────────────────────────────
# Build the notebook
# ────────────────────────────────────────────────────────────────────────────
def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(INTRO_MD))
    cells.append(nbf.v4.new_code_cell(SETUP_CODE))

    for exp in EXPERIMENTS:
        # Section header
        header = f"## Experiment {exp['n']} — {exp['title']}\n\n{exp['config_summary']}"
        if exp["status"] == "crashed":
            header += "\n\n**Status: training crashed.** " + exp["failure_note"]
        elif exp["status"] == "dead_neurons":
            header += "\n\n**Status: training failed (dead neurons).** " + exp["failure_note"]
        elif exp["status"] == "undertrained":
            header += "\n\n**Status: completed but under-trained.** Compare its final training accuracy to other completed runs to see the effect."
        cells.append(nbf.v4.new_markdown_cell(header))

        # Config table cell
        params_code = f"show_config({repr(exp['params'])})"
        cells.append(nbf.v4.new_code_cell(params_code))

        # Training curves cell
        epochs_data = {
            "train_loss": exp["train_loss"],
            "train_acc": exp["train_acc"],
            "spike_rate": exp["spike_rate"],
        }
        train_code = (
            f"plot_training_curves({repr(epochs_data)}, "
            f"framework={repr(exp['framework'])}, status={repr(exp['status'])})"
        )
        cells.append(nbf.v4.new_code_cell(train_code))

        # Test metrics cell — only for completed runs
        if exp["test_acc"] is not None:
            test_code = (
                f"plot_test_metrics(test_acc={exp['test_acc']}, "
                f"energy_per_sample={exp['energy_per_sample']}, "
                f"avg_spikes_per_sample={exp['avg_spikes_per_sample']}, "
                f"avg_latency_ms={exp['avg_latency_ms']}, "
                f"framework={repr(exp['framework'])})"
            )
            cells.append(nbf.v4.new_code_cell(test_code))
        else:
            cells.append(nbf.v4.new_markdown_cell(
                "*(No test results — training did not complete.)*"
            ))

        # Per-class F1 cell — only when data is available
        if exp.get("per_class_f1"):
            f1_code = (
                f"plot_per_class_f1({repr(exp['per_class_f1'])}, "
                f"framework={repr(exp['framework'])})"
            )
            cells.append(nbf.v4.new_code_cell(f1_code))

        # Separator
        cells.append(nbf.v4.new_markdown_cell("---"))

    nb["cells"] = cells

    # Strip "id" fields from markdown cells if they cause issues
    nb_metadata = nb.get("metadata", {})
    nb_metadata.setdefault("kernelspec", {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    })
    nb_metadata.setdefault("language_info", {
        "name": "python",
        "pygments_lexer": "ipython3",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "mimetype": "text/x-python",
        "file_extension": ".py",
    })
    nb["metadata"] = nb_metadata

    return nb


def main():
    print(f"Building notebook -> {NOTEBOOK_PATH}")
    nb = build_notebook()

    # Execute the notebook so outputs are embedded in the file
    print("Executing notebook (this will run all matplotlib cells)...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()

    # Save executed notebook
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Done. Notebook saved with {len(nb['cells'])} cells (executed).")


if __name__ == "__main__":
    main()
