from __future__ import annotations
import os
import csv
import time
import logging
from contextlib import contextmanager, nullcontext
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from skeleton import Settings
from event_data_workflow.system_monitor import PipelineMonitor, monitor
from learning.utilities import measure_dense_macs, compute_cv_isi
logger = logging.getLogger(__name__)

# Neuromorphic SynOps energy constant — see SNN_GPU_Evaluation_Metrics.md §2.4/§4.4
SYNOPS_ENERGY_PJ_PER_MAC = 4.6


def generate_trades_adversarial(model: torch.nn.Module, data: torch.Tensor, clean_prob: torch.Tensor, epsilon: float, steps: int) -> torch.Tensor:
    """Find the worst-case input within the epsilon-ball by maximising KL divergence
    from the clean prediction.

    Using torch.autograd.grad so model parameter gradients are never accumulated,
    keeping gradient accumulation in the outer training loop intact.
    clean_prob must be a detached softmax probability tensor [B, C].
    """
    alpha = 2.0 * epsilon / steps
    adv = torch.clamp(
        data + 0.001 * torch.randn_like(data),
        data - epsilon,
        data + epsilon,
    ).detach()

    model.activity.pause()
    try:
        for _ in range(steps):
            adv = adv.requires_grad_(True)
            adv_logits = aggregate_spike_output(model(adv).float())
            kl   = F.kl_div(F.log_softmax(adv_logits, dim=1), clean_prob, reduction="batchmean")
            grad = torch.autograd.grad(kl, adv)[0]
            adv  = torch.clamp((adv + alpha * grad.sign()).detach(), data - epsilon, data + epsilon)
    finally:
        model.activity.resume()

    return adv


def aggregate_spike_output(spk_rec: torch.Tensor) -> torch.Tensor:
    """Reduce any spike recording shape to [B, C] class logits.

    Supported layouts:
      [B, C]       — already aggregated (e.g. SpikingJelly pre-summed output)
      [T, B, C]    — timestep-first stack (e.g. Norse, SNNTorch, Brian2)
    """
    if spk_rec.dim() == 2:
        return spk_rec
    if spk_rec.dim() == 3:
        return spk_rec.sum(dim=0)
    raise ValueError(
        f"aggregate_spike_output: unsupported spike recording shape {tuple(spk_rec.shape)}. "
        "Expected [B, C] or [T, B, C]."
    )


def buffer_report(activity_snapshot: dict) -> str:
    """Sparse-vs-dense spike buffer memory diagnostic (AER memory savings),
    computed from a stored ActivityMonitor.recordings() snapshot — this is
    only ever invoked from the deferred end-of-training report, using
    tensors already captured GPU-side during the run, so the one `.item()`
    sync it costs happens there, not mid-loop."""
    tensors = {k: v for k, v in activity_snapshot.items() if v is not None}
    if not tensors:
        return "no hooks"
    sparse_kb = sum(t.element_size() * t.numel() for t in tensors.values()) / 1024
    avg_rate  = sum(t.float().mean().item() for t in tensors.values()) / len(tensors)
    dense_kb  = sparse_kb / avg_rate if avg_rate > 0 else 0.0
    return f"sparse={sparse_kb:.1f}KB  dense_equiv={dense_kb:.1f}KB  rate={avg_rate * 100:.1f}%"


class SNNTrainer:

    def __init__(self, model, train_loader, cfg: Settings, device: torch.device):
        self.model        = model
        self.train_loader = train_loader
        self.cfg          = cfg
        self.device       = device

        # Final, host-side histories — populated once in bulk by
        # finalize_epoch_reports() after the whole run completes. Kept
        # under these exact names since plot_training()/the train() return
        # dict already depend on them.
        self.loss_hist       = []
        self.acc_hist        = []
        self.spike_rate_hist = []
        self.last_spk_rec    = None
        self.epoch_log       = []

        # GPU-resident accumulation through the run — see train()'s docstring
        # note on the deferred-sync design. Nothing here is read back to host
        # memory until training finishes.
        self.loss_hist_gpu       = []
        self.acc_hist_gpu        = []
        self.spike_rate_hist_gpu = []
        self.fwd_events          = []  # (start, end) CUDA event pairs, or (t0, t1) perf_counter pairs on CPU
        self.bwd_events          = []
        self.dense_macs_per_layer = {}

        self.pipeline_monitor = PipelineMonitor(enabled=getattr(cfg, "ENABLE_PIPELINE_MONITOR", True))

        use_amp = getattr(cfg, "USE_AMP", True) and device.type == "cuda"
        self.scaler           = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]
        self.use_amp          = use_amp
        self.grad_accum_steps = max(1, getattr(cfg, "GRAD_ACCUM_STEPS", 1))

        lr_sched = getattr(cfg, "LR_SCHEDULER", "cosine")
        opt = getattr(self.model, "optimizer", None)
        self.scheduler = (CosineAnnealingLR(opt, T_max=cfg.EPOCHS) if opt is not None and lr_sched == "cosine" else None)

    def forward_pass(self, data: torch.Tensor) -> torch.Tensor:
        """Single forward pass, adapting tensor layout to what the model expects.

        Shared by every call site in train() (clean pass, TRADES adversarial pass)
        so the tensor_format() transpose isn't repeated at each one.
        """
        if self.model.tensor_format() == "BT":
            data = data.permute(1, 0, 2, 3, 4).contiguous()
        return self.model(data)

    @contextmanager
    def timed(self, event_list: list):
        """Brackets one operation with a CUDA event pair (CPU: a perf_counter
        pair) and appends it to `event_list`. `record()` queues on the stream
        without waiting for it, so this never stalls the training loop the
        way a `torch.cuda.synchronize()`-bracketed timer would — the
        blocking half (`elapsed_time()` / the CPU delta) only happens once,
        later, in `finalize_epoch_reports`."""
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            event_list.append((start, end))
        else:
            t0 = time.perf_counter()
            yield
            event_list.append((t0, time.perf_counter()))

    @staticmethod
    def elapsed_ms(pair) -> float:
        a, b = pair
        if isinstance(a, torch.cuda.Event):
            return a.elapsed_time(b)
        return (b - a) * 1000.0

    def write_csv(self, path: str):
        if not self.epoch_log:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = list(self.epoch_log[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.epoch_log)
        print(f"[INFO] Training log saved -> {path}")

    def train(self, csv_path: str = "./outputs/data/training_results.csv") -> dict:
        """Deferred-sync training loop: every per-batch metric (loss,
        accuracy, spike rate, forward/backward latency, SynOps energy) is
        accumulated as a GPU-resident tensor or an un-synced CUDA event —
        no `.item()`/`.cpu()` call happens inside the batch loop. Read back
        to host memory once per epoch, at that epoch's own boundary (not
        once for the whole run -- holding every epoch's events/tensors
        until the very end exhausted CUDA resources on a real multi-epoch
        run). See SNN_GPU_Evaluation_Metrics.md for why per-batch `.item()`
        calls are avoided (forces a CUDA stream sync, stalling the GPU every
        iteration)."""
        monitor.enter_phase("training")
        self.pipeline_monitor.start()
        self.pipeline_monitor.measure_idle_baseline()

        epochs    = self.cfg.EPOCHS
        num_iters = self.cfg.ITERA
        accum     = self.grad_accum_steps
        window_s  = getattr(self.cfg, 'TEMPORAL_SLICE_DURATION', 15000) / 1e6

        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext())

        # Dense-MAC measurement for the SynOps estimate — SNN_GPU_Evaluation_Metrics.md §2.4/§4.4.
        # self.train_loader is a PrefetchedLoader — probe_data is already device-resident.
        probe_data, _ = next(iter(self.train_loader))
        timesteps = probe_data.shape[0]  # loader yields [T, B, C, H, W] — the real BPTT unroll length, not a config value
        if self.model.tensor_format() == "BT":
            probe_data = probe_data.permute(1, 0, 2, 3, 4).contiguous()
        self.dense_macs_per_layer = measure_dense_macs(self.model, probe_data)

        raw_epoch_records: list[dict] = []

        for epoch in range(epochs):
            self.model.train_mode()
            epoch_loss_sum   = torch.zeros((), device=self.device)
            epoch_acc_sum    = torch.zeros((), device=self.device)
            epoch_spike_sum  = torch.zeros((), device=self.device)
            epoch_synops_sum = torch.zeros((), device=self.device)
            n, step_count = 0, 0
            t0 = time.perf_counter()
            self.pipeline_monitor.set_phase(f"epoch_{epoch}")
            self.pipeline_monitor.reset_epoch_memory()

            self.model.zero_grad()

            # self.train_loader is a PrefetchedLoader (event_data_workflow.data_pipeline) —
            # batches arrive already device-resident, no .to(device) needed below.
            for i, (data, targets) in enumerate(self.train_loader):
                if i % 20 == 0:
                    print(f"  [epoch {epoch + 1}/{epochs}] iter {i}/{num_iters}  "
                          f"({time.perf_counter() - t0:.1f}s elapsed)", flush=True)
                targets = targets.long()

                if self.cfg.TRADES_ENABLED:
                    reset = getattr(self.model, "reset_state", None)
                    with torch.no_grad():
                        if reset is not None:
                            reset()
                        clean_prob = F.softmax(
                            aggregate_spike_output(self.forward_pass(data).float()), dim=1
                        )
                    adv_data = generate_trades_adversarial(self.model, data, clean_prob, self.cfg.TRADES_EPSILON, self.cfg.TRADES_STEPS
                                                           )
                    with autocast_ctx:
                        if reset is not None:
                            reset()
                        with self.timed(self.fwd_events):
                            spk_rec = self.forward_pass(data)
                        if reset is not None:
                            reset()
                        # Adversarial forward pass isn't separately timed — the
                        # "forward latency" metric tracks one T-timestep clean
                        # inference, matching the doc's definition; TRADES'
                        # extra forward passes are an internal training cost.
                        spk_rec_adv  = self.forward_pass(adv_data)
                        clean_logits = aggregate_spike_output(spk_rec.float())
                        adv_logits   = aggregate_spike_output(spk_rec_adv.float())
                        ce_loss  = F.cross_entropy(clean_logits, targets)
                        kl_loss  = F.kl_div(
                            F.log_softmax(adv_logits,           dim=1),
                            F.softmax(clean_logits.detach(),    dim=1),
                            reduction="batchmean",
                        )
                        act_penalty = torch.zeros(1, device=self.device)
                        if self.cfg.ACTIVITY_REG_ENABLED:
                            act_penalty = self.model.activity.regularization_loss(
                                min_rate   = self.cfg.ACTIVITY_REG_MIN_RATE,
                                max_rate   = self.cfg.ACTIVITY_REG_MAX_RATE,
                                lambda_low = self.cfg.ACTIVITY_REG_LAMBDA_LOW,
                                lambda_high= self.cfg.ACTIVITY_REG_LAMBDA_HIGH,
                            )
                        loss_val = (ce_loss + self.cfg.TRADES_LAMBDA * kl_loss + act_penalty) / accum
                else:
                    with autocast_ctx:
                        with self.timed(self.fwd_events):
                            spk_rec   = self.forward_pass(data)
                        task_loss = self.model.loss_fn(spk_rec, targets)
                        if self.cfg.ACTIVITY_REG_ENABLED:
                            task_loss = task_loss + self.model.activity.regularization_loss(
                                min_rate   = self.cfg.ACTIVITY_REG_MIN_RATE,
                                max_rate   = self.cfg.ACTIVITY_REG_MAX_RATE,
                                lambda_low = self.cfg.ACTIVITY_REG_LAMBDA_LOW,
                                lambda_high= self.cfg.ACTIVITY_REG_LAMBDA_HIGH,
                            )
                        loss_val = task_loss / accum

                do_step = ((step_count + 1) % accum == 0)
                with self.timed(self.bwd_events):
                    self.model.backward_pass(loss_val, scaler=self.scaler, do_step=do_step)
                if do_step:
                    self.model.zero_grad()
                step_count += 1

                per_batch_loss = loss_val.detach() * accum
                logits = clean_logits.detach() if self.cfg.TRADES_ENABLED else aggregate_spike_output(spk_rec.detach().float())
                per_batch_acc = (logits.argmax(dim=1) == targets).float().mean()
                per_batch_spike = spk_rec.detach().float().mean()

                self.loss_hist_gpu.append(per_batch_loss)
                self.acc_hist_gpu.append(per_batch_acc)
                self.spike_rate_hist_gpu.append(per_batch_spike)
                self.last_spk_rec = spk_rec.detach()  # stays GPU-resident — see plot_raster()

                per_batch_synops = torch.zeros((), device=self.device)
                for name, macs in self.dense_macs_per_layer.items():
                    buf = self.model.activity.buffers.get(name)
                    rate_t = buf.firing_rate_tensor() if buf is not None else None
                    if rate_t is not None:
                        per_batch_synops = per_batch_synops + rate_t * macs * timesteps

                epoch_loss_sum   = epoch_loss_sum   + per_batch_loss
                epoch_acc_sum    = epoch_acc_sum    + per_batch_acc
                epoch_spike_sum  = epoch_spike_sum  + per_batch_spike
                epoch_synops_sum = epoch_synops_sum + per_batch_synops
                n += 1

                if i == num_iters:
                    break

            # Flush any gradients accumulated in a partial final batch (fires
            # when the loop exits mid-accumulation-cycle, e.g. ITERA not a
            # multiple of accum). Must NOT call backward_pass() again here --
            # that re-invokes loss.backward() on loss_val's graph, which the
            # in-loop call above (line ~300) already consumed, and PyTorch
            # frees a graph after backward() by default. The gradients from
            # every in-loop call are already accumulated in .grad; this only
            # needs to step the optimizer with them, mirroring backward_pass's
            # own scaler-branch logic without repeating the backward() call.
            if step_count % accum != 0:
                if self.use_amp:
                    self.scaler.step(self.model.optimizer)
                    self.scaler.update()
                else:
                    self.model.optimizer.step()
                self.model.zero_grad()

            # Read back and discard this epoch's CUDA events now, rather than
            # holding all epochs' events (a torch.cuda.Event pair per batch)
            # for the whole run -- on a real multi-epoch run that's thousands
            # of live event handles never freed until the very end, which
            # exhausted a CUDA resource and crashed a real 5-epoch/1175-batch
            # run late in training. elapsed_time() does NOT block until both
            # events complete -- it raises if they haven't -- so this sync is
            # required, not optional; it's one sync per epoch boundary, not
            # the per-batch sync the deferred-accumulation design avoids.
            torch.cuda.synchronize()
            fwd_latencies_ms = [self.elapsed_ms(p) for p in self.fwd_events]
            bwd_latencies_ms = [self.elapsed_ms(p) for p in self.bwd_events]
            self.fwd_events, self.bwd_events = [], []

            # Same reasoning, same fix, for the per-batch loss/accuracy/spike-rate
            # scalars: read back and clear each epoch's GPU-resident list here
            # instead of holding all 5 epochs' worth (1175 tensors each) until
            # the very end.
            self.loss_hist.extend(t.item() for t in self.loss_hist_gpu)
            self.acc_hist.extend(t.item() for t in self.acc_hist_gpu)
            self.spike_rate_hist.extend(t.item() for t in self.spike_rate_hist_gpu)
            self.loss_hist_gpu, self.acc_hist_gpu, self.spike_rate_hist_gpu = [], [], []

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_duration = time.perf_counter() - t0
            epoch_phase    = f"epoch_{epoch}"
            energy_report  = self.pipeline_monitor.phase_energy_report(epoch_phase, epoch_duration)  # None fields on CPU-only or without NVML
            gpu             = energy_report["gpu"]
            gpu_diag        = energy_report["gpu_diag"]
            energy_j        = energy_report["gpu_energy_j"] or 0.0
            avg_power_w     = energy_report["avg_power_w"] or 0.0
            dynamic_power_w = energy_report["dynamic_power_w"] or 0.0
            gpu_active_s    = epoch_duration * gpu.get("gpu_util_avg_pct", 0.0) / 100.0

            raw_epoch_records.append({
                "epoch":              epoch + 1,
                "n":                  n,
                "fwd_latencies_ms":   fwd_latencies_ms,
                "bwd_latencies_ms":   bwd_latencies_ms,
                "epoch_loss_sum":     epoch_loss_sum,
                "epoch_acc_sum":      epoch_acc_sum,
                "epoch_spike_sum":    epoch_spike_sum,
                "epoch_synops_sum":   epoch_synops_sum,
                "activity_snapshot":  self.model.activity.recordings(),
                "epoch_duration":     epoch_duration,
                "gpu":                gpu,
                "gpu_diag":           gpu_diag,
                "energy_j":           energy_j,
                "avg_power_w":        avg_power_w,
                "dynamic_power_w":    dynamic_power_w,
                "gpu_active_s":       gpu_active_s,
                "current_lr":         self.model.get_lr(),
            })

        # ---- Bulk transfer: the single sync point for everything
        # accumulated above, now that every epoch has finished training. ----
        self.finalize_epoch_reports(raw_epoch_records, epochs, timesteps, window_s)

        self.pipeline_monitor.stop()
        overall = self.pipeline_monitor.summary()
        if overall.get("avg_gpu_util_pct") is not None:
            print("\nGPU Training Summary")
            print(f"  • Avg utilization  : {overall['avg_gpu_util_pct']}%")
            print(f"  • Peak utilization : {overall['max_gpu_util_pct']}%")
            print(f"  • Peak VRAM used   : {overall['overall_peak_mem_gb']} GB / {overall['total_vram_gb']} GB  ({overall['overall_peak_mem_pct']}%)")

        trend = self.pipeline_monitor.memory_trend()
        if "ram_trend" in trend:
            print(f"  • RAM trend        : {trend['ram_trend']}  ({trend['ram_start_gb']}GB -> {trend['ram_end_gb']}GB)")

        self.write_csv(csv_path)

        return {
            "loss_history":       self.loss_hist,
            "accuracy_history":   self.acc_hist,
            "spike_rate_history": self.spike_rate_hist,
            "epoch_log":          self.epoch_log,
        }

    def finalize_epoch_reports(self, raw_epoch_records: list, epochs: int, timesteps: int, window_s: float) -> None:
        """Prints the full per-epoch report (deferred rather than streamed
        live) from data train() already synced back per-epoch --
        self.loss_hist/acc_hist/spike_rate_hist and each record's
        fwd/bwd_latencies_ms are already plain Python values by this point,
        read back at each epoch's own boundary rather than held GPU-resident
        for the whole run."""
        credit_assignment = self.model.credit_assignment()

        best_acc_so_far = 0.0
        for record in raw_epoch_records:
            n = record["n"]
            epoch_fwd = record["fwd_latencies_ms"]
            epoch_bwd = record["bwd_latencies_ms"]

            train_loss  = (record["epoch_loss_sum"]  / n).item() if n else 0.0
            train_acc   = (record["epoch_acc_sum"]   / n).item() if n else 0.0
            train_spike = (record["epoch_spike_sum"] / n).item() if n else 0.0
            synops_pj   = record["epoch_synops_sum"].item() * SYNOPS_ENERGY_PJ_PER_MAC if n else 0.0

            firing_rate_hz = train_spike * timesteps / window_s if window_s > 0 else 0.0

            cv_isi      = compute_cv_isi(record["activity_snapshot"])
            cv_isi_mean = cv_isi.get("network_wide", 0.0)
            buf_report  = buffer_report(record["activity_snapshot"])

            avg_fwd_ms = sum(epoch_fwd) / len(epoch_fwd) if epoch_fwd else 0.0
            avg_bwd_ms = sum(epoch_bwd) / len(epoch_bwd) if epoch_bwd else 0.0

            gpu      = record["gpu"]
            gpu_diag = record["gpu_diag"]

            is_best_epoch = train_acc > best_acc_so_far
            if is_best_epoch:
                best_acc_so_far = train_acc

            self.epoch_log.append({
                "epoch":               record["epoch"],
                "train_loss":          round(train_loss, 4),
                "train_accuracy":      round(train_acc, 4),
                "best_accuracy_so_far": round(best_acc_so_far, 4),
                "is_best_epoch":       is_best_epoch,
                "spike_rate":          round(train_spike, 4),
                "firing_rate_hz":      round(firing_rate_hz, 2),
                "learning_rate":       round(record["current_lr"], 6),
                "epoch_duration_s":    round(record["epoch_duration"], 2),
                "gpu_active_s":        round(record["gpu_active_s"], 2),
                "energy_j":            round(record["energy_j"], 2),
                "avg_power_w":         round(record["avg_power_w"], 2),
                "dynamic_power_w":     round(record["dynamic_power_w"], 2),
                "forward_latency_ms":  round(avg_fwd_ms, 3),
                "backward_latency_ms": round(avg_bwd_ms, 3),
                "cv_isi_mean":         round(cv_isi_mean, 4),
                "credit_assignment":   credit_assignment,
                "synops_energy_pj":    round(synops_pj, 2),
                **{k: v for k, v in gpu_diag.items()},
                **{k: v for k, v in gpu.items()},
            })

            print(f"\nEpoch {record['epoch']}/{epochs}")
            print(f"  • Train Loss     : {train_loss:.4f}")
            print(f"  • Train Accuracy : {train_acc * 100:.2f}%" + ("  (best so far)" if is_best_epoch else ""))
            print(f"  • Spike Rate     : {train_spike:.4f}")
            print(f"  • Firing Rate    : {firing_rate_hz:.2f} Hz")
            print(f"  • Spike Buffer   : {buf_report}")
            print(f"  • Fwd/Bwd Latency: {avg_fwd_ms:.3f} ms / {avg_bwd_ms:.3f} ms  ({credit_assignment})")
            print(f"  • CV_ISI (net)   : {cv_isi_mean:.3f}")
            print(f"  • SynOps Energy  : {synops_pj:.2f} pJ (estimated, per epoch)")
            print(f"  • LR             : {record['current_lr']:.6f}")
            print(f"  • Wall time      : {record['epoch_duration']:.2f}s")
            print(f"  • GPU active     : {record['gpu_active_s']:.2f}s  ({gpu.get('gpu_util_avg_pct', 0.0):.1f}% of wall time)")
            print(f"  • Energy         : {record['energy_j']:.2f} J  ({record['avg_power_w']:.1f} W avg)")
            if self.pipeline_monitor.idle_power_w is not None:
                print(f"  • Dynamic Power  : {record['dynamic_power_w']:.1f} W  (idle baseline {self.pipeline_monitor.idle_power_w:.1f} W subtracted)")
            if gpu:
                print(f"  • GPU Util       : avg {gpu['gpu_util_avg_pct']}%  peak {gpu['gpu_util_peak_pct']}%")
                print(f"  • GPU Memory     : {gpu['gpu_mem_peak_gb']} GB / {self.pipeline_monitor.total_memory_gb:.2f} GB  ({gpu['gpu_mem_peak_pct']}% peak)")
                if gpu.get("gpu_idle_episodes", 0) > 0:
                    print(f"  • GPU Idle       : {gpu['gpu_idle_episodes']} episode(s), {gpu['gpu_idle_total_s']:.1f}s total "
                          f"— CPU-side data prep couldn't keep up with the GPU this often")
            print(f"  • Max Mem Reserved: {gpu_diag.get('max_memory_reserved_gb', 0.0):.2f} GB")
            if "gpu_temp_c" in gpu_diag:
                print(f"  • GPU Temp/Clock : {gpu_diag['gpu_temp_c']}°C   SM {gpu_diag.get('sm_clock_mhz')} MHz   Mem {gpu_diag.get('mem_clock_mhz')} MHz")

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
        print(f"[PLOT] Saved -> {path}")

    def plot_raster(self, save_dir: str = "./outputs/plots") -> None:
        if self.last_spk_rec is None:
            print("[PLOT] No spike data — run train() first.")
            return
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        # Normalise to [T, C]: for [T, B, C] take sample 0; for [B, C] treat each batch row as a timestep
        # last_spk_rec is GPU-resident (deferred-sync design) — .cpu() happens
        # here, once, on explicit user request to plot, not inside the loop.
        spk = self.last_spk_rec.cpu()
        spk_sample = spk[:, 0, :] if spk.dim() == 3 else spk
        timesteps, neurons = spk_sample.numpy().nonzero()
        _, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(timesteps, neurons, s=2, c="black", marker="|")
        ax.set_title("Output Neuron Spike Raster  (last batch · sample 0)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron index")
        plt.tight_layout()
        path = os.path.join(save_dir, "spike_raster.png")
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[PLOT] Saved -> {path}")
