import os
import time
import csv
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional

def get_optimizer(model, config):
    # Support both dictionary config and Class-based Settings
    opt_name = getattr(config, "OPTIMIZER", "adam").lower()
    lr = getattr(config, "LEARNING_RATE", 1e-3)
    wd = getattr(config, "WEIGHT_DECAY", 0.0)

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)

def compute_spike_rate(model, x: torch.Tensor) -> float:
    """
    Measures average spike rate across all LIF layers in SpikingJelly.
    """
    model.eval()
    # SpikingJelly requires a reset before manual forward passes
    functional.reset_net(model)
    
    total_spikes = 0
    total_neurons = 0
    
    with torch.no_grad():
        # Step through time: x is [T, B, C, H, W]
        for t in range(x.size(0)):
            _ = model(x[t].unsqueeze(0)) # Forward one step
            
            # Look for all LIFNode layers in the model to count spikes
            for m in model.modules():
                if hasattr(m, 'spike'):
                    # m.spike stores the spikes from the last forward step
                    total_spikes += m.spike.sum().item()
                    total_neurons += m.spike.numel()

    return total_spikes / total_neurons if total_neurons > 0 else 0.0

def train(model, train_loader, config, results_dir: str, device: torch.device):
    optimizer = get_optimizer(model, config)
    # SpikingJelly + Classification = CrossEntropy
    criterion = nn.CrossEntropyLoss()

    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "train_accuracy_%", "spike_rate_%", "epoch_time_sec"])

    epochs = getattr(config, "EPOCHS", 10)
    print(f"\nStarting SpikingJelly Training: {epochs} epochs on {device}")

    best_acc = 0.0
    log_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        spike_rates = []
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            # 1. Move to device and FIX DATA TYPE to Long for CrossEntropy
            images = images.to(device) 
            labels = labels.to(device).long() # <--- CRITICAL FIX

            optimizer.zero_grad()

            # 2. Forward pass (returns summed spikes [B, 10])
            output = model(images)

            # 3. Compute loss and backprop
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # 4. Update metrics
            predicted = output.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                spike_rates.append(compute_spike_rate(model, images))

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * total_correct / total_samples
        avg_spkrate = 100.0 * (sum(spike_rates) / len(spike_rates))

        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}% | Spk: {avg_spkrate:.2f}% | {epoch_time:.1f}s")

        log_rows.append([epoch, round(avg_loss, 4), round(train_acc, 2), round(avg_spkrate, 2), round(epoch_time, 1)])

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)

    return log_rows