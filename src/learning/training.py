from __future__ import annotations

import os
import csv
import time
import logging
from contextlib import nullcontext

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from skeleton import Settings

logger = logging.getLogger(__name__)

cfg = Settings()
device = torch.device(cfg.DEVICE)


class SNNTrainer:

    def __init__(self, model, train_loader, cfg: Settings, device: torch.device):
        self.model        = model
        self.train_loader = train_loader
        self.cfg          = cfg
        self.device       = device

        self.loss_hist       = []
        self.acc_hist        = []
        self.spike_rate_hist = []
        self.last_spk_rec    = None
        self.epoch_log       = []

        use_amp = getattr(cfg, "USE_AMP", True) and device.type == "cuda"
        self.scaler           = torch.amp.GradScaler(enabled=use_amp)
        self.use_amp          = use_amp
        self.grad_accum_steps = max(1, getattr(cfg, "GRAD_ACCUM_STEPS", 1))

        lr_sched = getattr(cfg, "LR_SCHEDULER", "cosine")
        self.scheduler = (
            CosineAnnealingLR(self.model.optimizer, T_max=cfg.EPOCHS)
            if lr_sched == "cosine" else None
        )

    def save_checkpoint(self, path: str):
        ckpt = {
            "model_state_dict":     self.model.net.state_dict(),
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "scaler_state_dict":    self.scaler.state_dict(),
            "loss_history":         self.loss_hist,
            "accuracy_history":     self.acc_hist,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(ckpt, path)

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
        num_iters = self.cfg.ITERA
        accum     = self.grad_accum_steps

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp else nullcontext()
        )

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        else:
            checkpoint_path = None

        best_acc = 0.0

        for epoch in range(epochs):
            self.model.net.train()
            epoch_loss, epoch_acc, epoch_spike, n, step_count = 0.0, 0.0, 0.0, 0, 0
            t0 = time.perf_counter()

            self.model.optimizer.zero_grad(set_to_none=True)

            for i, (data, targets) in enumerate(self.train_loader):
                data    = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast_ctx:
                    spk_rec  = self.model(data)
                    loss_val = self.model.loss_fn(spk_rec, targets) / accum

                self.scaler.scale(loss_val).backward()
                step_count += 1

                if step_count % accum == 0:
                    self.scaler.step(self.model.optimizer)
                    self.scaler.update()
                    self.model.optimizer.zero_grad(set_to_none=True)

                raw_loss   = loss_val.item() * accum
                acc        = (spk_rec.detach().argmax(dim=1) == targets).float().mean().item()
                spike_rate = spk_rec.detach().float().mean().item()

                self.loss_hist.append(raw_loss)
                self.acc_hist.append(acc)
                self.spike_rate_hist.append(spike_rate)
                self.last_spk_rec = spk_rec.detach().cpu()

                epoch_loss  += raw_loss
                epoch_acc   += acc
                epoch_spike += spike_rate
                n += 1

                if i == num_iters:
                    break

            # flush any gradients accumulated in a partial final batch
            if step_count % accum != 0:
                self.scaler.step(self.model.optimizer)
                self.scaler.update()
                self.model.optimizer.zero_grad(set_to_none=True)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_duration = time.perf_counter() - t0
            train_loss     = epoch_loss / n
            train_acc      = epoch_acc  / n
            current_lr     = self.model.optimizer.param_groups[0]["lr"]

            train_spike = epoch_spike / n

            self.epoch_log.append({
                "epoch":            epoch + 1,
                "train_loss":       round(train_loss,     4),
                "train_accuracy":   round(train_acc,      4),
                "spike_rate":       round(train_spike,    4),
                "learning_rate":    round(current_lr,     6),
                "epoch_duration_s": round(epoch_duration, 2),
            })

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  • Train Loss     : {train_loss:.4f}")
            print(f"  • Train Accuracy : {train_acc * 100:.2f}%")
            print(f"  • Spike Rate     : {train_spike:.4f}")
            print(f"  • LR             : {current_lr:.6f}")
            print(f"  • Duration       : {epoch_duration:.2f}s")

            if checkpoint_path is not None and train_acc > best_acc:
                best_acc = train_acc
                self.save_checkpoint(checkpoint_path)
                print(f"  • Checkpoint saved (best: {best_acc * 100:.2f}%)")

        self.write_csv(csv_path)

        return {
            "loss_history":       self.loss_hist,
            "accuracy_history":   self.acc_hist,
            "spike_rate_history": self.spike_rate_hist,
            "epoch_log":          self.epoch_log,
        }

    def plot_training(self, save_dir: str = "./outputs/plots") -> None:
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle("SNN Training Metrics", fontsize=14)

        axes[0].plot(self.loss_hist, linewidth=0.8)
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.acc_hist, color="tab:green", linewidth=0.8)
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.spike_rate_hist, color="tab:orange", linewidth=0.8)
        axes[2].set_ylabel("Spike Rate")
        axes[2].set_xlabel("Batch")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, "training_metrics.png")
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[PLOT] Saved → {path}")

    def plot_raster(self, save_dir: str = "./outputs/plots") -> None:
        if self.last_spk_rec is None:
            print("[PLOT] No spike data — run train() first.")
            return
        import matplotlib.pyplot as plt
        from snntorch import spikeplot as splt
        os.makedirs(save_dir, exist_ok=True)

        spk_sample = self.last_spk_rec[:, 0, :]   # [T, num_classes] — first sample in last batch
        fig, ax = plt.subplots(figsize=(10, 3))
        splt.raster(spk_sample, ax, s=40, c="black")
        ax.set_title("Output Neuron Spike Raster  (last batch · sample 0)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron index")
        plt.tight_layout()
        path = os.path.join(save_dir, "spike_raster.png")
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[PLOT] Saved → {path}")
    
