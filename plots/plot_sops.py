"""SOPS -- synaptic operations per SECOND -- for inference and across training.

SynOps is a count; SOPS is that count divided by a time. Which time is not a detail:
the same runs give 13.9 GSOP/s measured against the forward pass and 0.27 GSOP/s
measured against the test-pass wall clock, because the test loader re-runs
Denoise+ToFrame on every sample single-threaded (data_pipeline.py -- the test split is
uncached and pinned to num_workers=0). Both numbers are real and they answer different
questions, so both are drawn and both are labelled:

  COMPUTE RATE     synaptic ops / forward-pass time   -- what the framework can sustain
  DELIVERED RATE   synaptic ops / test-pass wall clock -- what the pipeline actually gets

For training the time base is the measured step time (forward + backward latency), which
needs no batch count and is recorded per epoch. The operations counted are forward-pass
synaptic operations -- the backward sweep is in the denominator but not the numerator,
because SynOps is a forward-pass quantity. That makes the training figure "forward
synaptic operations delivered per second of training-step compute", which is what it is
labelled.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(r"C:\Users\zuhai\Desktop\Projects\SNN\records")
OUT = Path(r"C:\Users\zuhai\Desktop\Projects\SNN\SNNs-auf-GPUs")
ORDER = ["norse", "torch", "sj", "sinabs"]
COLOUR = {"norse": "#2b6cb0", "torch": "#c53030", "sj": "#2f855a", "sinabs": "#d9822b"}
NICE = {"dvs128_gesture": "DVS128 Gesture", "n_caltech101": "N-Caltech101"}
DATASETS = ["dvs128_gesture", "n_caltech101"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
PJ_PER_MAC = 4.6          # learning/inference.py SYNOPS_ENERGY_PJ_PER_MAC
GIGA = 1e9


def load(dataset):
    runs = {r["run_id"]: r for r in csv.DictReader((ROOT / dataset / "results" / "runs.csv")
                                                   .open(newline="", encoding="utf-8"))
            if r["framework"] in ORDER}
    epochs = {}
    for row in csv.DictReader((ROOT / dataset / "results" / "epochs.csv").open(newline="", encoding="utf-8")):
        if row["run_id"] in runs:
            epochs.setdefault(row["run_id"], []).append(row)

    out = {}
    for run_id, run in runs.items():
        batch = int(run["batch_size"])
        synops_per_sample = float(run["infer_synops_energy_per_sample_pj"]) / PJ_PER_MAC
        # Amortised p50 is batch time / B around the forward call only, so its inverse is
        # the rate the network itself sustains with data already on the device.
        forward_s = float(run["inference_latency_amortised_p50_ms"]) / 1000.0
        rows = epochs.get(run_id, [])
        training = []
        for row in rows:
            # Per-batch synops, so per-SAMPLE synops needs the batch size out -- the
            # calibrated batch differs by framework and would otherwise be counted as speed.
            per_sample = float(row["synops_energy_pj"]) / batch / PJ_PER_MAC
            step_s = (float(row["forward_latency_ms"]) + float(row["backward_latency_ms"])) / 1000.0
            # One step processes one batch, so the operations delivered in that step are
            # per_sample x batch.
            training.append(per_sample * batch / step_s / GIGA)
        out[run["framework"]] = {
            "compute": synops_per_sample / forward_s / GIGA,
            "delivered": synops_per_sample * float(run["inference_throughput_samples_per_s"]) / GIGA,
            "training": training,
            "synops_per_sample": synops_per_sample,
        }
    return out


DATA = {d: load(d) for d in DATASETS}
PRESENT = {d: [f for f in ORDER if f in DATA[d]] for d in DATASETS}


def tidy(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---- inference -----------------------------------------------------------------
figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
for ax, dataset in zip(axes, DATASETS):
    present = PRESENT[dataset]
    positions = np.arange(len(present))
    compute = [DATA[dataset][f]["compute"] for f in present]
    delivered = [DATA[dataset][f]["delivered"] for f in present]
    ax.bar(positions - 0.19, compute, width=0.36, color=[COLOUR[f] for f in present])
    ax.bar(positions + 0.19, delivered, width=0.36, color=[COLOUR[f] for f in present],
           alpha=0.32, hatch="//", edgecolor="white")
    top = max(compute)
    for position, (c, d) in zip(positions, zip(compute, delivered)):
        ax.text(position - 0.19, c + top * 0.02, f"{c:.1f}", ha="center", fontsize=9.5,
                fontweight="bold", color=INK)
        ax.text(position + 0.19, d + top * 0.02, f"{d:.2f}", ha="center", fontsize=9,
                color=MUTED)
    ax.set_xticks(positions)
    ax.set_xticklabels(present, fontsize=11)
    ax.set_ylim(0, top * 1.22)
    ax.set_ylabel("GSOP/s  (giga synaptic operations per second)", fontsize=10)
    ax.set_title(NICE[dataset], fontsize=12, loc="left", pad=8)
    tidy(ax)
# Neutral proxy swatches: a legend keyed to one framework's colour reads as if the
# two bars belonged to that framework rather than to every one of them.
proxies = [Patch(facecolor="#6f7782", label="compute rate (forward pass)"),
           Patch(facecolor="#6f7782", alpha=0.32, hatch="//", edgecolor="white",
                 label="delivered rate (whole test pass)")]
axes[0].legend(handles=proxies, fontsize=9, frameon=False, loc="upper left")
figure.suptitle("Inference SOPS: what the network sustains, and what the pipeline delivers",
                fontsize=14.5, fontweight="bold", y=0.978)
figure.text(0.5, 0.916,
            "solid = synaptic ops / forward-pass time   |   hatched = synaptic ops / test-pass "
            "wall clock, which the uncached single-process test loader dominates",
            ha="center", fontsize=9.8, color=MUTED)
figure.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.09, wspace=0.22)
figure.savefig(OUT / "sops_inference.png", dpi=130, facecolor="white")
plt.close(figure)

# ---- training: trend on top, distribution underneath ----------------------------
figure, axes = plt.subplots(2, 2, figsize=(13, 9.2))
rng = np.random.default_rng(1)
for column, dataset in enumerate(DATASETS):
    present = PRESENT[dataset]
    ax = axes[0][column]
    for framework in present:
        series = DATA[dataset][framework]["training"]
        ax.plot(range(1, len(series) + 1), series, marker="o", ms=4.5, lw=1.9,
                color=COLOUR[framework], label=framework)
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("GSOP/s", fontsize=10.5)
    ax.set_title(f"{NICE[dataset]}  -  across training", fontsize=12, loc="left", pad=8)
    ax.legend(fontsize=9, frameon=False)
    tidy(ax)

    ax = axes[1][column]
    data = [DATA[dataset][f]["training"] for f in present]
    parts = ax.violinplot(data, positions=range(len(present)), widths=0.72,
                          showmeans=False, showextrema=False)
    for body, framework in zip(parts["bodies"], present):
        body.set_facecolor(COLOUR[framework]); body.set_alpha(0.45)
        body.set_edgecolor(COLOUR[framework])
    for position, (framework, values) in enumerate(zip(present, data)):
        ax.scatter(rng.normal(position, 0.055, len(values)), values, s=18,
                   color=COLOUR[framework], zorder=3, edgecolor="white", linewidth=0.5)
        ax.hlines(np.median(values), position - 0.22, position + 0.22, color="black", lw=2, zorder=4)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, fontsize=11)
    ax.set_ylabel("GSOP/s", fontsize=10.5)
    ax.set_title(f"{NICE[dataset]}  -  distribution over the 15 epochs",
                 fontsize=12, loc="left", pad=8)
    tidy(ax)

figure.suptitle("Training SOPS: forward synaptic operations per second of training-step compute",
                fontsize=14.5, fontweight="bold", y=0.980)
figure.text(0.5, 0.940,
            "time base is the measured step time (forward + backward latency), so this is a "
            "compute rate and carries no data-loading time",
            ha="center", fontsize=9.8, color=MUTED)
figure.tight_layout(rect=[0, 0, 1, 0.925])
figure.savefig(OUT / "sops_training.png", dpi=130, facecolor="white")
plt.close(figure)

for name in ("sops_inference", "sops_training"):
    print(f"written: {name}.png  ({(OUT / f'{name}.png').stat().st_size / 1024:.0f} KB)")

print()
for dataset in DATASETS:
    print(f"{dataset}")
    for framework in PRESENT[dataset]:
        entry = DATA[dataset][framework]
        train = np.array(entry["training"])
        print(f"   {framework:7s} inference compute {entry['compute']:7.2f} GSOP/s   "
              f"delivered {entry['delivered']:6.3f} GSOP/s   "
              f"training median {np.median(train):6.2f} GSOP/s  "
              f"(epoch1 {train[0]:.2f} -> last {train[-1]:.2f})")
