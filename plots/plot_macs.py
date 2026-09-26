"""MACs figures: inference endpoint, training trend, training distribution, and fitted normals."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\zuhai\Desktop\Projects\SNN\records")
OUT = Path(r"C:\Users\zuhai\Desktop\Projects\SNN\SNNs-auf-GPUs")
ORDER = ["norse", "torch", "sj", "sinabs"]
COLOUR = {"norse": "#2b6cb0", "torch": "#c53030", "sj": "#2f855a", "sinabs": "#d9822b"}
NICE = {"dvs128_gesture": "DVS128 Gesture", "n_caltech101": "N-Caltech101"}
DATASETS = ["dvs128_gesture", "n_caltech101"]
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
PJ_PER_MAC = 4.6          # learning/inference.py SYNOPS_ENERGY_PJ_PER_MAC
MILLION = 1e6


def load(dataset):
    """Per-framework MACs, normalised per sample so frameworks with different calibrated
    batch sizes are comparable. dense_macs_downstream and the per-epoch synops figure are
    both recorded per BATCH, and the calibrated batch size differs by framework (36 vs 30
    on DVS128) -- comparing them unnormalised would credit the smaller batch with being
    cheaper."""
    runs = {r["run_id"]: r for r in csv.DictReader((ROOT / dataset / "results" / "runs.csv")
                                                   .open(newline="", encoding="utf-8"))
            if r["framework"] in ORDER}
    layers, epochs = {}, {}
    for row in csv.DictReader((ROOT / dataset / "results" / "layers.csv").open(newline="", encoding="utf-8")):
        if row["run_id"] in runs and (row["dense_macs_downstream"] or "").strip():
            layers.setdefault(row["run_id"], []).append(float(row["dense_macs_downstream"]))
    for row in csv.DictReader((ROOT / dataset / "results" / "epochs.csv").open(newline="", encoding="utf-8")):
        if row["run_id"] in runs:
            epochs.setdefault(row["run_id"], []).append(row)

    out = {}
    for run_id, run in runs.items():
        batch = int(run["batch_size"])
        rows = epochs.get(run_id, [])
        # x T, because dense_macs_downstream is the arithmetic for ONE timestep and
        # SynOps sums over all of them (inference.py: rate * macs * T). The honest
        # ceiling is therefore what a dense network would spend across the whole
        # T-step window -- without the factor the "effective" figure can exceed it.
        timesteps = int(float(run["time_steps"]))
        out[run["framework"]] = {
            "dense": sum(layers.get(run_id, [])) * timesteps / batch / MILLION,
            "timesteps": timesteps,
            "inference": float(run["infer_synops_energy_per_sample_pj"]) / PJ_PER_MAC / MILLION,
            "training": [float(e["synops_energy_pj"]) / batch / PJ_PER_MAC / MILLION for e in rows],
            "spike_rate": [float(e["spike_rate_pct"]) for e in rows],
            "cv_isi": [float(e["cv_isi_mean"]) for e in rows],
            "batch": batch,
        }
    return out


DATA = {d: load(d) for d in DATASETS}
PRESENT = {d: [f for f in ORDER if f in DATA[d]] for d in DATASETS}


def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---- 1. inference endpoint ------------------------------------------------------
# A grey bar for the dense cost and a coloured bar drawn inside it, rather than a
# ceiling line: the effective figure is 4-21% of dense, so a line at the ceiling
# would leave every real bar as an unreadable sliver at the axis.
figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
for ax, dataset in zip(axes, DATASETS):
    present = PRESENT[dataset]
    dense = DATA[dataset][present[0]]["dense"]
    positions = np.arange(len(present))
    ax.barh(positions, [dense] * len(present), height=0.62, color="#dfe3e8",
            edgecolor="#c3c9d0", linewidth=0.8)
    for position, framework in zip(positions, present):
        value = DATA[dataset][framework]["inference"]
        ax.barh(position, value, height=0.62, color=COLOUR[framework])
        ax.text(dense * 0.015 + value, position, f"  {value:,.0f}M   {100 * value / dense:.1f}%",
                va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(positions)
    ax.set_yticklabels(present, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, dense * 1.30)
    ax.set_xlabel("MACs per sample (millions)", fontsize=10.5)
    ax.set_title(f"{NICE[dataset]}      dense = {dense:,.0f}M", fontsize=12, loc="left", pad=8)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
figure.suptitle("Arithmetic per inference: what the architecture specifies vs what the spikes require",
                fontsize=14.5, fontweight="bold", y=0.975)
figure.text(0.5, 0.918,
            "grey = dense MACs summed over all T timesteps, identical for all four (same architecture)"
            "   |   coloured = effective MACs after sparsity (SynOps)",
            ha="center", fontsize=10, color=MUTED)
figure.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.11, wspace=0.20)
figure.savefig(OUT / "macs_inference.png", dpi=130, facecolor="white")
plt.close(figure)

# ---- 2. across training ---------------------------------------------------------
# No ceiling line here: at 20-25x the plotted values it would flatten every curve.
# The ceiling is stated per panel instead.
figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, dataset in zip(axes, DATASETS):
    for framework in PRESENT[dataset]:
        series = DATA[dataset][framework]["training"]
        ax.plot(range(1, len(series) + 1), series, marker="o", ms=4.5, lw=1.9,
                color=COLOUR[framework], label=framework)
    dense = DATA[dataset][PRESENT[dataset][0]]["dense"]
    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("MACs per sample (millions)", fontsize=10.5)
    ax.set_title(f"{NICE[dataset]}      dense ceiling = {dense:,.0f}M (off scale)",
                 fontsize=12, loc="left", pad=8)
    ax.legend(fontsize=9, frameon=False)
    tidy(ax)
figure.suptitle("Effective MACs across training", fontsize=14.5, fontweight="bold", y=0.980)
figure.text(0.5, 0.908,
            "the network is free to grow or shrink its own arithmetic cost while it learns",
            ha="center", fontsize=10, color=MUTED)
figure.tight_layout(rect=[0, 0, 1, 0.875])
figure.savefig(OUT / "macs_training.png", dpi=130, facecolor="white")
plt.close(figure)

# ---- 3. training distribution ---------------------------------------------------
figure, axes = plt.subplots(1, 2, figsize=(13, 5.4))
rng = np.random.default_rng(1)
for ax, dataset in zip(axes, DATASETS):
    present = PRESENT[dataset]
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
    ax.set_xticklabels(present, fontsize=10.5)
    ax.set_ylabel("MACs per sample (millions)", fontsize=10.5)
    ax.set_title(NICE[dataset], fontsize=12, loc="left", pad=8)
    tidy(ax)
figure.suptitle("Distribution of effective MACs across training epochs",
                fontsize=14.5, fontweight="bold", y=0.975)
figure.text(0.5, 0.925,
            "each dot is one epoch  |  black bar = median  |  width = how often that value occurred",
            ha="center", fontsize=10, color=MUTED)
figure.tight_layout(rect=[0, 0, 1, 0.90])
figure.savefig(OUT / "macs_training_distribution.png", dpi=130, facecolor="white")
plt.close(figure)

# ---- 4. fitted normals for spike rate and CV(ISI) -------------------------------
METRICS = [("spike_rate", "spike rate (%)"), ("cv_isi", "CV(ISI)")]
figure, axes = plt.subplots(2, 2, figsize=(13, 8.6))
for row, (key, label) in enumerate(METRICS):
    for column, dataset in enumerate(DATASETS):
        ax = axes[row][column]
        pooled = np.concatenate([DATA[dataset][f][key] for f in PRESENT[dataset]])
        span = max(pooled.std() * 3.2, np.ptp(pooled) * 0.6, 1e-6)
        grid = np.linspace(pooled.min() - span, pooled.max() + span, 600)
        rugs = []
        for framework in PRESENT[dataset]:
            values = np.array(DATA[dataset][framework][key])
            mean, sd = values.mean(), values.std(ddof=1)
            # A normal fitted to the 15 epoch values: centre is where the metric settled,
            # width is how much it moved while it settled there. A narrow curve is a
            # framework whose internals barely shifted during training.
            curve = np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            ax.plot(grid, curve, lw=2.2, color=COLOUR[framework],
                    label=f"{framework}   mean {mean:.3g}, sd {sd:.2g}")
            ax.fill_between(grid, curve, color=COLOUR[framework], alpha=0.12)
            rugs.append((values, COLOUR[framework]))
        baseline = -0.045 * ax.get_ylim()[1]
        for values, colour in rugs:
            ax.plot(values, np.full(len(values), baseline), "|", color=colour, ms=8, mew=1.3)
        ax.set_ylim(baseline * 1.6, ax.get_ylim()[1])
        ax.set_xlabel(label, fontsize=10.5)
        ax.set_ylabel("probability density", fontsize=10)
        ax.set_title(f"{NICE[dataset]}  -  {label}", fontsize=11.5, loc="left", pad=8)
        ax.legend(fontsize=8.5, frameon=False)
        tidy(ax)
figure.suptitle("Spike rate and CV(ISI) as fitted normal distributions over training",
                fontsize=14.5, fontweight="bold", y=0.978)
figure.text(0.5, 0.940,
            "one normal per framework, fitted to its 15 epoch values  |  ticks along the bottom "
            "are the epochs themselves",
            ha="center", fontsize=10, color=MUTED)
figure.tight_layout(rect=[0, 0, 1, 0.925])
figure.savefig(OUT / "normal_fit_spikerate_cvisi.png", dpi=130, facecolor="white")
plt.close(figure)

for name in ("macs_inference", "macs_training", "macs_training_distribution",
             "normal_fit_spikerate_cvisi"):
    path = OUT / f"{name}.png"
    print(f"written: {name}.png  ({path.stat().st_size / 1024:.0f} KB)")

print()
for dataset in DATASETS:
    dense = DATA[dataset][PRESENT[dataset][0]]["dense"]
    print(f"{dataset}   dense MACs/sample {dense:,.1f}M")
    for framework in PRESENT[dataset]:
        entry = DATA[dataset][framework]
        train = np.array(entry["training"])
        print(f"   {framework:7s} inference {entry['inference']:9,.1f}M "
              f"({100 * entry['inference'] / dense:5.1f}% of dense)   "
              f"training median {np.median(train):8,.1f}M  "
              f"epoch1 {train[0]:8,.1f}M -> last {train[-1]:8,.1f}M")
