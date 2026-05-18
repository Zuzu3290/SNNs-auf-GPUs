from __future__ import annotations

import os
import csv
import time
import torch
from skeleton import Settings

class SNNTrainer:

    def __init__(self, model, train_loader, cfg: Settings, device: torch.device):
        self.model        = model
        self.train_loader = train_loader
        self.cfg          = cfg
        self.device       = device

        self.loss_hist  = []
        self.acc_hist   = []
        self.epoch_log  = []

    def save_checkpoint(self, path: str):
        # SpikingJelly models usually store the network in 'net'
        torch.save({
            "model_state_dict":     self.model.net.state_dict(),
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "loss_history":         self.loss_hist,
            "accuracy_history":     self.acc_hist,
        }, path)

    def write_csv(self, path: str):
        if not self.epoch_log:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = list(self.epoch_log[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.epoch_log)
        print(f"[INFO] Training log saved → {path}")

    def train(self,
              checkpoint_dir: str,
              checkpoint_name: str = "best_model.pt",
              csv_path: str = "./outputs/data/training_results.csv") -> dict:

        epochs    = self.cfg.EPOCHS
        num_iters = getattr(self.cfg, 'ITERA', 999999) # Default to high number if not set

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        else:
            checkpoint_path = None

        best_acc = 0.0

        for epoch in range(epochs):
            self.model.net.train()
            epoch_loss, epoch_acc, n = 0.0, 0.0, 0
            t0 = time.perf_counter()

            for i, (data, targets) in enumerate(self.train_loader):
                # Move to device and FIX: targets must be Long for CrossEntropy on GPU
                data    = data.to(self.device)
                targets = targets.to(self.device).long() 

                # Forward pass - output is now summed spikes [B, 10]
                spk_rec  = self.model(data)
                loss_val = self.model.loss_fn(spk_rec, targets)

                self.model.optimizer.zero_grad()
                loss_val.backward()
                self.model.optimizer.step()

                # Manual accuracy calculation (replacing snntorch.SF)
                predicted = spk_rec.argmax(dim=1)
                acc = (predicted == targets).float().mean().item()
                
                self.loss_hist.append(loss_val.item())
                self.acc_hist.append(acc)
                epoch_loss += loss_val.item()
                epoch_acc  += acc
                n += 1

                if i == num_iters:
                    break

            epoch_duration = time.perf_counter() - t0
            train_loss     = epoch_loss / n
            train_acc      = epoch_acc  / n

            self.epoch_log.append({
                "epoch":               epoch + 1,
                "train_loss":         round(train_loss,     4),
                "train_accuracy":     round(train_acc,      4),
                "epoch_duration_s":   round(epoch_duration, 2),
            })

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  • Train Loss     : {train_loss:.4f}")
            print(f"  • Train Accuracy : {train_acc * 100:.2f}%")
            print(f"  • Duration       : {epoch_duration:.2f}s")

            if checkpoint_path is not None and train_acc > best_acc:
                best_acc = train_acc
                self.save_checkpoint(checkpoint_path)
                print(f"  • Checkpoint saved (best: {best_acc * 100:.2f}%)")

        self.write_csv(csv_path)

        return {
            "loss_history":           self.loss_hist,
            "accuracy_history":       self.acc_hist,
            "epoch_log":               self.epoch_log,
        }