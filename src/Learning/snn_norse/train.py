# =============================================================
# train.py — Training loop for the Norse SNN
# =============================================================
# Trains the SNN built by model.py on N-MNIST.
# Tracks per-epoch: loss, accuracy, spike rate, epoch time.
# Saves a training_log.csv to the results folder.
#
# Called by run.py — do not run this directly.

import os
import time
import csv
import torch
import torch.nn as nn


def get_optimizer(model, config: dict):
    """
    Builds the optimizer from config["optimizer"].
    weight_decay applies L2 regularization to all weights.
    """
    opt = config["optimizer"].lower()
    if opt == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    elif opt == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    elif opt == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")


def compute_spike_rate(model, x: torch.Tensor) -> float:
    """
    Measures average spike rate across the hidden layer.

    Spike rate = fraction of neurons that fired at each timestep,
    averaged over all timesteps and all samples in the batch.

    Example:
      256 hidden neurons, T=16 timesteps, batch=64
      If on average 51 neurons fire per timestep → spike rate = 20%

    Why it matters:
      Too low  → network is too silent, not learning
      Too high → network is firing too much, losing temporal selectivity
      ~10-20%  → healthy sparse coding, typical for SNNs
    """
    model.eval()
    T  = config_from_model(model)["timesteps"]

    # run one batch and collect hidden spikes
    # we re-run the forward pass manually to intercept hidden layer spikes
    x = x.permute(1, 0, 2, 3, 4)
    T_actual, batch_size = x.shape[0], x.shape[1]
    x_flat = x.reshape(T_actual, batch_size, -1)

    hidden_states = [None] * len(model.lif_cells)
    all_hidden_spikes = []

    with torch.no_grad():
        for t in range(T_actual):
            z = x_flat[t]
            for i, lif_cell in enumerate(model.lif_cells):
                if model.neuron_type in ("LIF", "LIFAdEx"):
                    z = model.fc_layers[i](z)
                    z, hidden_states[i] = lif_cell(z, hidden_states[i])
                elif model.neuron_type == "LIFRecurrent":
                    z, hidden_states[i] = lif_cell(z, hidden_states[i])
            all_hidden_spikes.append(z)

    # (T, batch, hidden) → mean fraction of neurons that fired
    spikes = torch.stack(all_hidden_spikes, dim=0)
    return spikes.mean().item()   # value in [0, 1]


def config_from_model(model):
    return model.config


def train(model, train_loader, config: dict, results_dir: str, device: torch.device):
    """
    Full training loop.

    Per epoch tracks:
      - average loss
      - training accuracy
      - hidden layer spike rate
      - time taken

    Saves:
      results_dir/training_log.csv
    """
    optimizer = get_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_log.csv")

    # CSV header
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "avg_loss",
            "train_accuracy_%",
            "spike_rate_%",
            "epoch_time_sec",
        ])

    print(f"\nTraining for {config['epochs']} epochs on {device}")
    print(f"Log → {log_path}\n")

    best_acc  = 0.0
    log_rows  = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()

        total_loss    = 0.0
        total_correct = 0
        total_samples = 0
        spike_rates   = []
        epoch_start   = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)   # (batch, T, 2, 34, 34)
            labels = labels.to(device)   # (batch,)

            optimizer.zero_grad()

            # forward pass — output is (batch, 10) class scores
            output = model(images)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # ── Metrics this batch ────────────────────────────
            predicted      = output.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss    += loss.item()

            # spike rate: sample every 50 batches to avoid slowdown
            if batch_idx % 50 == 0:
                rate = compute_spike_rate(model, images)
                spike_rates.append(rate)

            # progress print every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch}/{config['epochs']} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        # ── End of epoch metrics ──────────────────────────────
        epoch_time  = time.time() - epoch_start
        avg_loss    = total_loss / len(train_loader)
        train_acc   = 100.0 * total_correct / total_samples
        avg_spkrate = 100.0 * (sum(spike_rates) / len(spike_rates))

        if train_acc > best_acc:
            best_acc = train_acc

        print(
            f"\nEpoch {epoch}/{config['epochs']} complete | "
            f"Loss: {avg_loss:.4f} | "
            f"Accuracy: {train_acc:.2f}% | "
            f"Spike rate: {avg_spkrate:.2f}% | "
            f"Time: {epoch_time:.1f}s\n"
        )

        # save row
        log_rows.append([
            epoch,
            round(avg_loss, 4),
            round(train_acc, 2),
            round(avg_spkrate, 2),
            round(epoch_time, 1),
        ])

    # write all rows to CSV
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)

    print(f"Training complete. Best train accuracy: {best_acc:.2f}%")
    print(f"Training log saved → {log_path}\n")

    return log_rows
