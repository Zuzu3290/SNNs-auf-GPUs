# =============================================================
# evaluate.py — Full evaluation and performance matrix
# =============================================================
# Runs on the test set after training is complete.
# Computes every metric relevant to SNN evaluation.
# Saves:
#   results_dir/evaluation.md     — all metrics in a table
#   results_dir/confusion_matrix.png
#   results_dir/spike_raster.png  — one sample visualised
#
# Called by run.py — do not run this directly.

import os
import time
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Helper: run one batch through model, collect everything ───
def _forward_with_spikes(model, x: torch.Tensor):
    """
    Re-runs the forward pass while collecting hidden + output spikes
    at every timestep. Used for spike-level metrics.

    Returns:
      predictions   : (batch,)       predicted class per sample
      output_spikes : (T, batch, 10) output spike trains
      hidden_spikes : (T, batch, H)  hidden layer spike trains
    """
    x_perm = x.permute(1, 0, 2, 3, 4)
    T, B   = x_perm.shape[0], x_perm.shape[1]
    x_flat = x_perm.reshape(T, B, -1)

    hidden_states = [None] * len(model.lif_cells)
    out_state     = None

    all_hidden = []
    all_output = []

    with torch.no_grad():
        for t in range(T):
            z = x_flat[t]
            for i, cell in enumerate(model.lif_cells):
                if model.neuron_type in ("LIF", "LIFAdEx"):
                    z = model.fc_layers[i](z)
                    z, hidden_states[i] = cell(z, hidden_states[i])
                else:
                    z, hidden_states[i] = cell(z, hidden_states[i])
                z = model.dropouts[i](z)
            all_hidden.append(z.detach().cpu())

            z = model.fc_out(z)
            z, out_state = model.lif_out(z, out_state)
            all_output.append(z.detach().cpu())

    hidden_spikes = torch.stack(all_hidden, dim=0)   # (T, B, H)
    output_spikes = torch.stack(all_output, dim=0)   # (T, B, 10)
    decoded       = model._decode(output_spikes.to(x.device))
    predictions   = decoded.argmax(dim=1).cpu()

    return predictions, output_spikes, hidden_spikes


# ── Metric computations ────────────────────────────────────────

def _confusion_matrix(all_preds, all_labels, n_classes=10):
    """
    n_classes × n_classes matrix.
    cm[true][predicted] = count of samples
    Diagonal = correct predictions.
    Off-diagonal = where the model confused two classes.
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for pred, label in zip(all_preds, all_labels):
        cm[label][pred] += 1
    return cm


def _per_class_accuracy(cm):
    """
    Accuracy per digit class (0–9).
    cm[i][i] / sum(cm[i]) = fraction correctly identified as class i.
    """
    per_class = {}
    for i in range(cm.shape[0]):
        total = cm[i].sum()
        per_class[i] = round(100.0 * cm[i][i] / total, 2) if total > 0 else 0.0
    return per_class


def _spike_sparsity(hidden_spikes: torch.Tensor) -> float:
    """
    Fraction of neurons that stayed SILENT (did not fire) across all
    timesteps and all samples.

    High sparsity (e.g. 85%) = network is efficient — most neurons
    silent most of the time. This is desirable in SNNs because it
    means only the most relevant neurons activate.

    sparsity = 1 - spike_rate
    """
    return 1.0 - hidden_spikes.mean().item()


def _total_spike_count(hidden_spikes: torch.Tensor, output_spikes: torch.Tensor) -> dict:
    """
    Raw spike counts across the entire test set.
    Useful for comparing energy efficiency across neuron types —
    fewer spikes = less computation = more efficient on neuromorphic hardware.
    """
    return {
        "hidden_total": int(hidden_spikes.sum().item()),
        "output_total": int(output_spikes.sum().item()),
    }


def _convergence_epoch(log_rows: list, threshold: float = 90.0) -> int:
    """
    First epoch where training accuracy crossed `threshold`%.
    Returns -1 if it never reached the threshold.
    Tells you how quickly each neuron type starts learning.
    """
    for row in log_rows:
        epoch, _, acc, _, _ = row
        if acc >= threshold:
            return epoch
    return -1


def _avg_inference_time(model, test_loader, device, n_batches=20) -> float:
    """
    Average time (ms) to process one sample.
    Timed over first n_batches batches to get a stable estimate.
    """
    model.eval()
    times = []
    for i, (images, _) in enumerate(test_loader):
        if i >= n_batches:
            break
        images = images.to(device)
        start  = time.perf_counter()
        with torch.no_grad():
            model(images)
        end = time.perf_counter()
        # time per sample in ms
        times.append((end - start) * 1000 / images.shape[0])
    return round(sum(times) / len(times), 4)


# ── Plots ──────────────────────────────────────────────────────

def _plot_confusion_matrix(cm, results_dir: str, label: str):
    """
    Heatmap of the confusion matrix.
    Rows = true digit, columns = predicted digit.
    Bright diagonal = correct predictions.
    Off-diagonal bright = common confusions (e.g. 3↔8, 4↔9).
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xlabel("Predicted digit", fontsize=12)
    ax.set_ylabel("True digit", fontsize=12)
    ax.set_title(f"Confusion Matrix — {label}", fontsize=13)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    # annotate each cell with its count
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i][j]),
                    ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black",
                    fontsize=8)

    plt.tight_layout()
    path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {path}")


def _plot_spike_raster(model, test_loader, device, results_dir: str, label: str):
    """
    Spike raster for a single test sample.
    Shows hidden neuron activity and output neuron activity over T timesteps.
    Lets you visually inspect what the network is doing inside.
    """
    images, labels = next(iter(test_loader))
    sample  = images[0:1].to(device)   # single sample
    true_lbl = labels[0].item()

    preds, out_spk, hid_spk = _forward_with_spikes(model, sample)

    # shapes: (T, 1, H) and (T, 1, 10) → squeeze batch dim
    hid = hid_spk[:, 0, :].numpy()    # (T, H)
    out = out_spk[:, 0, :].numpy()    # (T, 10)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # ── Input image (sum over T and polarity to get 34×34 activity map)
    activity = images[0].sum(dim=0).sum(dim=0).numpy()   # (34, 34)
    axes[0].imshow(activity, cmap="hot")
    axes[0].set_title(
        f"Input Activity Map — True label: {true_lbl}", fontsize=13
    )
    axes[0].axis("off")

    # ── Hidden spike raster (first 64 neurons for clarity)
    axes[1].imshow(hid[:, :64].T, aspect="auto", cmap="binary",
                   interpolation="nearest")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Hidden neuron (first 64)")
    axes[1].set_title("Hidden Layer Spike Raster", fontsize=13)
    axes[1].set_xticks(range(hid.shape[0]))

    # ── Output spike raster (all 10 output neurons)
    predicted = int(out.sum(axis=0).argmax())
    axes[2].imshow(out.T, aspect="auto", cmap="Blues",
                   interpolation="nearest")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Output neuron (digit class)")
    axes[2].set_yticks(range(10))
    axes[2].set_xticks(range(out.shape[0]))
    axes[2].set_title(
        f"Output Layer Spike Raster — Predicted: {predicted}", fontsize=13
    )

    plt.tight_layout()
    path = os.path.join(results_dir, "spike_raster.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Spike raster saved     → {path}")


def _plot_training_curves(log_rows: list, results_dir: str, label: str):
    """
    Three subplots showing how training progressed:
      1. Loss per epoch
      2. Accuracy per epoch
      3. Spike rate per epoch
    """
    epochs    = [r[0] for r in log_rows]
    losses    = [r[1] for r in log_rows]
    accs      = [r[2] for r in log_rows]
    spk_rates = [r[3] for r in log_rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10))

    axes[0].plot(epochs, losses, marker="o", color="crimson")
    axes[0].set_title(f"Training Loss — {label}", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True)

    axes[1].plot(epochs, accs, marker="o", color="steelblue")
    axes[1].set_title("Training Accuracy", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True)

    axes[2].plot(epochs, spk_rates, marker="o", color="seagreen")
    axes[2].set_title("Hidden Layer Spike Rate", fontsize=12)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Spike rate (%)")
    axes[2].grid(True)

    plt.tight_layout()
    path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved  → {path}")


# ── Main evaluate function ─────────────────────────────────────

def evaluate(model, test_loader, config: dict,
             log_rows: list, results_dir: str, device: torch.device):
    """
    Full evaluation on the test set.

    Computes:
      - test accuracy
      - per-class accuracy (all 10 digits)
      - confusion matrix
      - spike sparsity
      - total spike counts (hidden + output)
      - average inference time per sample
      - convergence epoch (first epoch ≥ 90% train accuracy)
      - model parameter count
      - total training time

    Saves:
      training_log.csv      (already saved by train.py)
      training_curves.png
      confusion_matrix.png
      spike_raster.png
      evaluation.md         ← master results table
    """
    model.eval()
    os.makedirs(results_dir, exist_ok=True)

    label = f"{config['neuron_type']}_{config['decoding']}_T{config['timesteps']}"

    print("\nRunning evaluation on test set...")

    # ── Pass through entire test set ──────────────────────────
    all_preds  = []
    all_labels = []
    all_hidden_spikes = []
    all_output_spikes = []

    for images, labels in test_loader:
        images = images.to(device)
        preds, out_spk, hid_spk = _forward_with_spikes(model, images)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_hidden_spikes.append(hid_spk)
        all_output_spikes.append(out_spk)

    # concatenate across all batches along batch dimension
    all_hidden_spikes = torch.cat(all_hidden_spikes, dim=1)  # (T, N_test, H)
    all_output_spikes = torch.cat(all_output_spikes, dim=1)  # (T, N_test, 10)

    # ── Compute all metrics ───────────────────────────────────
    test_acc    = round(100.0 * sum(p == l for p, l in zip(all_preds, all_labels))
                        / len(all_labels), 2)
    cm          = _confusion_matrix(all_preds, all_labels)
    per_class   = _per_class_accuracy(cm)
    sparsity    = round(100.0 * _spike_sparsity(all_hidden_spikes), 2)
    spk_counts  = _total_spike_count(all_hidden_spikes, all_output_spikes)
    conv_epoch  = _convergence_epoch(log_rows, threshold=90.0)
    infer_time  = _avg_inference_time(model, test_loader, device)
    n_params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_time  = round(sum(r[4] for r in log_rows), 1)
    best_train  = max(r[2] for r in log_rows)
    final_loss  = log_rows[-1][1]

    # ── Print summary ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS — {label}")
    print(f"{'='*55}")
    print(f"  Test Accuracy          : {test_acc}%")
    print(f"  Best Train Accuracy    : {best_train}%")
    print(f"  Final Train Loss       : {final_loss}")
    print(f"  Convergence Epoch      : {conv_epoch}  (first ≥90% train acc)")
    print(f"  Spike Sparsity         : {sparsity}%  (% silent neurons)")
    print(f"  Hidden Spike Count     : {spk_counts['hidden_total']:,}")
    print(f"  Output Spike Count     : {spk_counts['output_total']:,}")
    print(f"  Avg Inference Time     : {infer_time} ms/sample")
    print(f"  Trainable Parameters   : {n_params:,}")
    print(f"  Total Training Time    : {total_time}s")
    print(f"{'='*55}")
    print(f"\n  Per-class Accuracy:")
    for digit, acc in per_class.items():
        bar = "█" * int(acc / 5)
        print(f"    Digit {digit}: {acc:6.2f}%  {bar}")

    # ── Save plots ────────────────────────────────────────────
    _plot_confusion_matrix(cm, results_dir, label)
    _plot_spike_raster(model, test_loader, device, results_dir, label)
    _plot_training_curves(log_rows, results_dir, label)

    # ── Save evaluation.md ────────────────────────────────────
    md_path = os.path.join(results_dir, "evaluation.md")
    with open(md_path, "w") as f:
        f.write(f"# Evaluation Results — {label}\n\n")

        f.write("## Config\n\n")
        f.write("| Parameter | Value |\n|---|---|\n")
        for k, v in config.items():
            if k not in ("data_dir", "results_dir"):
                f.write(f"| {k} | {v} |\n")

        f.write("\n## Training Performance\n\n")
        f.write("| Epoch | Avg Loss | Train Accuracy % | Spike Rate % | Time (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for row in log_rows:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n")

        f.write("\n## Test Performance\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Test Accuracy | **{test_acc}%** |\n")
        f.write(f"| Best Train Accuracy | {best_train}% |\n")
        f.write(f"| Final Train Loss | {final_loss} |\n")
        f.write(f"| Convergence Epoch (≥90%) | {conv_epoch} |\n")
        f.write(f"| Spike Sparsity | {sparsity}% |\n")
        f.write(f"| Hidden Spike Count (test set) | {spk_counts['hidden_total']:,} |\n")
        f.write(f"| Output Spike Count (test set) | {spk_counts['output_total']:,} |\n")
        f.write(f"| Avg Inference Time | {infer_time} ms/sample |\n")
        f.write(f"| Trainable Parameters | {n_params:,} |\n")
        f.write(f"| Total Training Time | {total_time}s |\n")

        f.write("\n## Per-Class Accuracy\n\n")
        f.write("| Digit | Accuracy % |\n|---|---|\n")
        for digit, acc in per_class.items():
            f.write(f"| {digit} | {acc} |\n")

        f.write("\n## Outputs\n\n")
        f.write("| File | Description |\n|---|---|\n")
        f.write("| training_log.csv | Loss, accuracy, spike rate per epoch |\n")
        f.write("| training_curves.png | Loss, accuracy, spike rate plots |\n")
        f.write("| confusion_matrix.png | True vs predicted digit heatmap |\n")
        f.write("| spike_raster.png | Hidden and output spike activity |\n")

    print(f"\nEvaluation report saved → {md_path}")

    return {
        "label":         label,
        "test_acc":      test_acc,
        "best_train":    best_train,
        "final_loss":    final_loss,
        "conv_epoch":    conv_epoch,
        "sparsity":      sparsity,
        "hidden_spikes": spk_counts["hidden_total"],
        "output_spikes": spk_counts["output_total"],
        "infer_time":    infer_time,
        "n_params":      n_params,
        "total_time":    total_time,
        "per_class":     per_class,
    }
