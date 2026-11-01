from __future__ import annotations
import os
import csv
import time
import logging
from pathlib import Path
from contextlib import nullcontext
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from skeleton import Settings
import matplotlib.pyplot as plt
from event_data_workflow.system_monitor import PipelineMonitor
from learning.utilities import elapsed_ms, spike_count_loss, timed, warm_up
logger = logging.getLogger(__name__)


class SNNTrainer:

    def __init__(self, model, train_loader, cfg: Settings, device: torch.device):
        self.model        = model
        self.train_loader = train_loader
        self.cfg          = cfg
        self.device       = device

        # One value per iteration across the whole run, read back from the GPU once per epoch.
        self.loss_hist       = []
        self.acc_hist        = []
        self.spike_rate_hist = []
        self.epoch_log       = []

        self.pipeline_monitor = PipelineMonitor()

        self.use_amp          = cfg.USE_AMP and device.type == "cuda"
        self.scaler           = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_accum_steps = cfg.GRAD_ACCUM_STEPS
        self.optimizer        = getattr(torch.optim, cfg.OPTIMIZER)(model.parameters(), **cfg.OPTIMIZER_ARGS)  # a torch.optim class by name
        self.scheduler        = CosineAnnealingLR(self.optimizer, T_max=cfg.EPOCHS) if cfg.LR_SCHEDULER == "cosine" else None

    def write_batch_csv(self, path: str = "./outputs/data/batch_metrics.csv"):
        """One row per training iteration across the whole run."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss", "accuracy", "spike_rate"])
            writer.writerows(zip(range(len(self.loss_hist)), self.loss_hist, self.acc_hist, self.spike_rate_hist))
        logger.info(f"[INFO] Iteration metrics saved -> {path}")

    def train(self, results_dir: str = "./outputs/data") -> dict:
        """Training loop with deferred sync: per-batch metrics and timings stay on the GPU and are read back once per epoch."""
        self.pipeline_monitor.start()
        self.pipeline_monitor.measure_idle_baseline()

        epochs    = self.cfg.EPOCHS
        num_iters = self.cfg.ITERA
        accum     = self.grad_accum_steps
        logger.info(f"[TRAIN] {epochs} epoch(s) x {num_iters} iterations (batch {self.cfg.BATCH_SIZE})")

        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext())

        # Untimed warm-up, so kernel compilation is not charged to the first epoch.
        probe_data, _ = next(iter(self.train_loader))  # already on the device
        self.warmup = warm_up(self.model, probe_data, self.cfg.WARMUP_ITERATIONS)
        if self.warmup["iterations"]:
            logger.info(f"[WARM-UP] {self.warmup['iterations']} untimed forward+backward passes  "
                        f"(weights unchanged: {self.warmup['weights_unchanged']})")
            if not self.warmup["weights_unchanged"]:
                raise RuntimeError(
                    "[WARM-UP] the model's weights changed during warm-up. No optimizer "
                    "step is taken there, so this means something else is mutating them "
                    "-- the timed epochs would not start from the seeded weights."
                )

        for epoch in range(epochs):
            self.model.train()
            loss_dev, acc_dev, spike_dev, fwd_events, bwd_events = [], [], [], [], []
            t0 = time.perf_counter()
            self.pipeline_monitor.set_phase(f"epoch_{epoch}")
            self.pipeline_monitor.reset_epoch_memory()

            self.optimizer.zero_grad(set_to_none=True)

            for i, (data, targets) in enumerate(self.train_loader):
                if i >= num_iters:  # checked before the work, so an epoch runs exactly num_iters batches
                    break
                if i % 20 == 0:
                    logger.info(f"[epoch {epoch + 1}/{epochs}] iter {i}/{num_iters}  "
                                f"({time.perf_counter() - t0:.1f}s elapsed)")
                targets = targets.long()

                with autocast_ctx:
                    with timed(self.device, fwd_events):
                        spk_rec = self.model(data)
                    task_loss = spike_count_loss(spk_rec, targets, self.cfg.LOSS_FN)
                    if self.cfg.ACTIVITY_REG_ENABLED:
                        task_loss = task_loss + self.model.spike_recorder.penalty(
                            min_rate   = self.cfg.ACTIVITY_REG_MIN_RATE,
                            max_rate   = self.cfg.ACTIVITY_REG_MAX_RATE,
                            lambda_low = self.cfg.ACTIVITY_REG_LAMBDA_LOW,
                            lambda_high= self.cfg.ACTIVITY_REG_LAMBDA_HIGH,
                        )
                    loss_val = task_loss / accum

                do_step = ((i + 1) % accum == 0)
                with timed(self.device, bwd_events):
                    self.scaler.scale(loss_val).backward()  # a disabled scaler (no AMP) passes the loss through unchanged
                    if do_step:  # otherwise gradients keep accumulating
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                if do_step:
                    self.optimizer.zero_grad(set_to_none=True)

                spikes = spk_rec.detach().float()
                loss_dev.append(loss_val.detach() * accum)
                acc_dev.append((spikes.sum(dim=0).argmax(dim=1) == targets).float().mean())  # the spike count per class is its score
                spike_dev.append(spikes.mean())

            n = len(loss_dev)
            # A partial accumulation cycle at the end: step with the gradients already accumulated (no second backward).
            if n % accum != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            # The one sync per epoch: everything above is read back here.
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            losses = [t.item() for t in loss_dev]
            accs   = [t.item() for t in acc_dev]
            rates  = [t.item() for t in spike_dev]
            self.loss_hist.extend(losses)
            self.acc_hist.extend(accs)
            self.spike_rate_hist.extend(rates)
            fwd_ms = sum(elapsed_ms(p) for p in fwd_events) / n
            bwd_ms = sum(elapsed_ms(p) for p in bwd_events) / n

            if self.scheduler is not None:
                self.scheduler.step()

            duration = time.perf_counter() - t0
            energy   = self.pipeline_monitor.phase_energy_report(f"epoch_{epoch}", duration)  # None fields without NVML
            gpu      = energy["gpu"]
            energy_j = energy["gpu_energy_j"] or 0.0          # idle draw included
            dynamic_j = energy["gpu_dynamic_energy_j"] or 0.0  # above idle
            idle_w   = energy["idle_power_w"]

            entry = {
                "epoch":               epoch + 1,
                "train_loss":          round(sum(losses) / n, 4),
                "train_accuracy":      round(sum(accs) / n, 4),
                "spike_rate":          round(sum(rates) / n, 4),
                "epoch_duration_s":    round(duration, 2),
                "forward_latency_ms":  round(fwd_ms, 3),
                "backward_latency_ms": round(bwd_ms, 3),
                "energy_j_total":      round(energy_j, 2),
                "energy_j_dynamic":    round(dynamic_j, 2),
                "idle_power_w":        round(idle_w, 2) if idle_w is not None else None,
                **gpu,
            }
            self.epoch_log.append(entry)

            lines = [f"Epoch {epoch + 1}/{epochs}",
                     f"  • Train Loss      : {entry['train_loss']:.4f}",
                     f"  • Train Accuracy  : {entry['train_accuracy'] * 100:.2f}%",
                     f"  • Spike Rate      : {entry['spike_rate']:.4f}",
                     f"  • Fwd/Bwd Latency : {fwd_ms:.3f} ms / {bwd_ms:.3f} ms",
                     f"  • Wall time       : {duration:.2f}s",
                     f"  • Energy (total)  : {energy_j:.2f} J  ({energy['avg_power_w'] or 0.0:.1f} W avg, idle draw included)"]
            if idle_w is not None:
                lines.append(f"  • Energy (dynamic): {dynamic_j:.2f} J  "
                             f"({energy['dynamic_power_w'] or 0.0:.1f} W above a {idle_w:.1f} W idle baseline)")
            else:
                lines.append("  • Energy (dynamic): not available -- no idle baseline was measured")
            if gpu:
                lines.append(f"  • GPU Util        : avg {gpu['gpu_util_avg_pct']}%  peak {gpu['gpu_util_peak_pct']}%")
                if gpu.get("gpu_idle_episodes", 0) > 0:
                    lines.append(f"  • GPU Idle        : {gpu['gpu_idle_episodes']} episode(s), {gpu['gpu_idle_total_s']:.1f}s total "
                                 f"— CPU-side data prep couldn't keep up with the GPU this often")
            logger.info("\n".join(lines))

        if self.model.spike_recorder is not None:
            self.model.spike_recorder.clear()
        self.pipeline_monitor.stop()

        self.write_batch_csv(str(Path(results_dir) / "batch_metrics.csv"))

        return {
            "loss_history":       self.loss_hist,
            "accuracy_history":   self.acc_hist,
            "spike_rate_history": self.spike_rate_hist,
            "epoch_log":          self.epoch_log,
        }

    def plot_training(self, save_dir: str = "./outputs/plots") -> None:
        """Loss, accuracy and spike rate per iteration, across all epochs."""
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
        axes[2].set_xlabel("Iteration (cumulative across epochs)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, "training_metrics.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"[PLOT] Saved -> {path}")
