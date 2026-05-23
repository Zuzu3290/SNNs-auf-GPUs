from __future__ import annotations

import os
import csv
import time
from contextlib import nullcontext

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from snntorch import functional as SF
from skeleton import Settings

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
                acc        = SF.accuracy_rate(spk_rec, targets)
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
    


"""Multi-node distributed training entry point.

Launch with ``torchrun`` (or the helper ``scripts/launch.sh``).  All
hyperparameters are configurable via CLI flags backed by
:class:`config.TrainingConfig`.

Example (single-node, 2 GPUs)::

    torchrun --standalone --nproc_per_node=2 train.py --epochs 5

Example (2 nodes, 4 GPUs each)::

    # Node 0
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
             --master_addr=<IP> --master_port=12355 train.py

    # Node 1
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
             --master_addr=<IP> --master_port=12355 train.py
"""

# from __future__ import annotations

# import os
# import time
# from contextlib import nullcontext
# from typing import Tuple

# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.data import DataLoader

# from config import TrainingConfig
# from dataset import SyntheticImageDataset, create_distributed_dataloader
# from ddp_utils import (
#     DistributedContext,
#     cleanup_distributed,
#     is_main_process,
#     load_checkpoint,
#     save_checkpoint,
#     setup_distributed,
# )
# from model import create_model, wrap_ddp
# from utils import (
#     MetricTracker,
#     all_reduce_scalar,
#     compute_accuracy,
#     log_epoch_summary,
#     log_gpu_info,
#     log_training_step,
#     setup_logger,
# )


# # ── Single-epoch training ───────────────────────────────────────────

# def train_one_epoch(
#     model: nn.Module,
#     loader: DataLoader,
#     criterion: nn.Module,
#     optimizer: torch.optim.Optimizer,
#     scaler: torch.amp.GradScaler,
#     ctx: DistributedContext,
#     config: TrainingConfig,
#     epoch: int,
#     logger,
# ) -> MetricTracker:
#     """Run one full pass over the training set.

#     Supports mixed-precision and gradient accumulation.  Gradient
#     synchronisation is handled transparently by DDP.

#     Returns
#     -------
#     MetricTracker
#         Contains running ``loss`` and ``accuracy`` averages for the
#         epoch.
#     """
#     model.train()
#     tracker = MetricTracker()
#     total_steps = len(loader)

#     use_amp = config.use_amp and ctx.device.type == "cuda"
#     autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

#     optimizer.zero_grad(set_to_none=True)

#     step_start = time.perf_counter()

#     for step, (images, labels) in enumerate(loader):
#         images = images.to(ctx.device, non_blocking=True)
#         labels = labels.to(ctx.device, non_blocking=True)

#         with autocast_ctx:
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss = loss / config.grad_accum_steps

#         scaler.scale(loss).backward()

#         if (step + 1) % config.grad_accum_steps == 0:
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad(set_to_none=True)

#         raw_loss = loss.item() * config.grad_accum_steps
#         acc = compute_accuracy(outputs, labels)
#         batch_size = images.size(0)
#         tracker.update("loss", raw_loss, n=batch_size)
#         tracker.update("accuracy", acc, n=batch_size)

#         if is_main_process(ctx.rank) and (step + 1) % config.log_interval == 0:
#             elapsed = time.perf_counter() - step_start
#             throughput = batch_size * config.log_interval / elapsed
#             current_lr = optimizer.param_groups[0]["lr"]
#             log_training_step(
#                 logger,
#                 epoch=epoch,
#                 step=step + 1,
#                 total_steps=total_steps,
#                 loss=raw_loss,
#                 lr=current_lr,
#                 throughput=throughput,
#             )
#             step_start = time.perf_counter()

#     return tracker


# # ── Profiling context manager ────────────────────────────────────────

# def _profiler_context(config: TrainingConfig, epoch: int):
#     """Return a ``torch.profiler`` context if profiling is enabled."""
#     if not config.enable_profiling:
#         return nullcontext()

#     return torch.profiler.profile(
#         activities=[
#             torch.profiler.ProfilerActivity.CPU,
#             torch.profiler.ProfilerActivity.CUDA,
#         ],
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(
#             os.path.join(config.checkpoint_dir, "profiler", f"epoch_{epoch}"),
#         ),
#         record_shapes=True,
#         with_stack=True,
#     )


# # ── Main ─────────────────────────────────────────────────────────────

# def main() -> None:
#     """Parse config, set up distributed training, and run the loop."""
#     config = TrainingConfig.from_args()

#     ctx = setup_distributed(config)
#     logger = setup_logger(ctx.rank)

#     torch.manual_seed(config.seed + ctx.rank)
#     torch.cuda.manual_seed(config.seed + ctx.rank)

#     if is_main_process(ctx.rank):
#         logger.info("=" * 60)
#         logger.info("Multi-Node DDP Training Pipeline")
#         logger.info("=" * 60)
#         logger.info("Config: %s", config)
#         log_gpu_info(logger, ctx.device)
#         logger.info(
#             "World size: %d  |  Backend: %s", ctx.world_size, config.backend,
#         )

#     # ── Model ────────────────────────────────────────────────────
#     model = create_model(config, ctx.device)
#     model = wrap_ddp(model, ctx.local_rank)

#     if is_main_process(ctx.rank):
#         total_params = sum(p.numel() for p in model.parameters())
#         logger.info("Model parameters: %s", f"{total_params:,}")

#     # ── Optimizer / Scheduler / Scaler ───────────────────────────
#     optimizer = torch.optim.SGD(
#         model.parameters(),
#         lr=config.lr,
#         momentum=config.momentum,
#         weight_decay=config.weight_decay,
#     )
#     scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
#     scaler = torch.amp.GradScaler(enabled=config.use_amp)

#     # ── Resume from checkpoint ───────────────────────────────────
#     start_epoch = 1
#     if config.resume_from is not None:
#         start_epoch = load_checkpoint(
#             config.resume_from,
#             model.module,
#             optimizer,
#             scaler,
#             device=ctx.device,
#         ) + 1
#         for _ in range(1, start_epoch):
#             scheduler.step()
#         if is_main_process(ctx.rank):
#             logger.info("Resuming from epoch %d", start_epoch)

#     # ── Dataset / DataLoader ─────────────────────────────────────
#     dataset = SyntheticImageDataset(
#         size=config.dataset_size,
#         image_size=config.image_size,
#         in_channels=config.in_channels,
#         num_classes=config.num_classes,
#     )
#     loader, sampler = create_distributed_dataloader(dataset, config, ctx)

#     criterion = nn.CrossEntropyLoss()

#     # ── Training loop ────────────────────────────────────────────
#     if is_main_process(ctx.rank):
#         logger.info("Starting training  |  epochs=%d  batch_size=%d/gpu",
#                      config.epochs, config.batch_size)

#     try:
#         for epoch in range(start_epoch, config.epochs + 1):
#             epoch_start = time.perf_counter()
#             sampler.set_epoch(epoch)

#             with _profiler_context(config, epoch):
#                 tracker = train_one_epoch(
#                     model=model,
#                     loader=loader,
#                     criterion=criterion,
#                     optimizer=optimizer,
#                     scaler=scaler,
#                     ctx=ctx,
#                     config=config,
#                     epoch=epoch,
#                     logger=logger,
#                 )

#             scheduler.step()

#             avg_loss = all_reduce_scalar(
#                 tracker.average("loss"), ctx.world_size, ctx.device,
#             )
#             avg_acc = all_reduce_scalar(
#                 tracker.average("accuracy"), ctx.world_size, ctx.device,
#             )

#             epoch_time = time.perf_counter() - epoch_start

#             if is_main_process(ctx.rank):
#                 log_epoch_summary(logger, epoch, {
#                     "loss": avg_loss,
#                     "accuracy": avg_acc,
#                     "lr": optimizer.param_groups[0]["lr"],
#                     "epoch_time_s": epoch_time,
#                 })

#                 if epoch % config.save_every == 0:
#                     ckpt_path = os.path.join(
#                         config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt",
#                     )
#                     save_checkpoint(
#                         path=ckpt_path,
#                         epoch=epoch,
#                         model=model,
#                         optimizer=optimizer,
#                         scaler=scaler,
#                         rank=ctx.rank,
#                     )

#         if is_main_process(ctx.rank):
#             logger.info("Training complete.")

#     finally:
#         cleanup_distributed()


# if __name__ == "__main__":
#     main()


# def train_one_epoch(model, loader, criterion, optimizer, scaler, ctx, config, epoch, logger):
#     """
#     Gradient Accumulation
#     Sometimes you need a larger effective batch size than your GPU memory allows. 
#     Gradient accumulation simulates this by running multiple forward-backward passes before stepping the optimizer. 
#     The key is to divide the loss by grad_accum_steps before backward(), so the accumulated gradient is correctly averaged.
#     """
#     model.train()
#     tracker = MetricTracker()
#     total_steps = len(loader)


#     use_amp = config.use_amp and ctx.device.type == "cuda"
#     autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()


#     optimizer.zero_grad(set_to_none=True)


#     for step, (images, labels) in enumerate(loader):
#         images = images.to(ctx.device, non_blocking=True)
#         labels = labels.to(ctx.device, non_blocking=True)


#         with autocast_ctx:
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss = loss / config.grad_accum_steps  # scale for accumulation


#         scaler.scale(loss).backward()


#         if (step + 1) % config.grad_accum_steps == 0:
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad(set_to_none=True)  # memory-efficient reset


#         # Track raw (unscaled) loss for logging
#         raw_loss = loss.item() * config.grad_accum_steps
#         acc = compute_accuracy(outputs, labels)
#         tracker.update("loss", raw_loss, n=images.size(0))
#         tracker.update("accuracy", acc, n=images.size(0))


#         if is_main_process(ctx.rank) and (step + 1) % config.log_interval == 0:
#             log_training_step(logger, epoch, step + 1, total_steps,
#                               raw_loss, optimizer.param_groups[0]["lr"])


#     return tracker