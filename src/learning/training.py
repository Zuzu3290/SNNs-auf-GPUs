from __future__ import annotations

import os
import csv
import time
import logging
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from skeleton import Settings, ReliabilityTracker
from event_data_workflow.gpu_stats import GPUStats
from learning.frameworks.activity_reg import get_hidden_spike_recordings, activity_regularization, stdp_regularization, pause_hooks, resume_hooks
from runtime import PhaseManager, SpikeRateBus, Phase

logger = logging.getLogger(__name__)


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

    pause_hooks(model)
    try:
        for a in range(steps):
            adv = adv.requires_grad_(True)
            adv_logits = aggregate_spike_output(model(adv).float())
            kl   = F.kl_div(F.log_softmax(adv_logits, dim=1), clean_prob, reduction="batchmean")
            grad = torch.autograd.grad(kl, adv)[0]
            adv  = torch.clamp((adv + alpha * grad.sign()).detach(), data - epsilon, data + epsilon)
    finally:
        resume_hooks(model)

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


class SNNTrainer:

    def __init__(self, model, train_loader, cfg: Settings, device: torch.device):
        self.model        = model
        self.train_loader = train_loader
        self.cfg          = cfg
        self.device       = device
        self.kernel_module = None
        self.use_custom_kernel = False
        self.kernel_mode = cfg.KERNEL
        if cfg.KERNEL == "ON":
            try:
                import sys as _sys
                # Reuse already-loaded module if present — avoids reimporting on trainer reinit
                km = _sys.modules.get("snn_forward") or __import__("snn_forward")
                self.kernel_module = km
                self.use_custom_kernel = True
                self._report_kernel_compatibility()
            except ImportError:
                print("[kernel] snn_forward not built — run: python src/learning/setup.py build_ext --inplace")
                self.reliability.record_interruption("kernel import failed — falling back to framework forward")
        self._voltage_buf: torch.Tensor | None = None

        self.loss_hist       = []
        self.acc_hist        = []
        self.spike_rate_hist = []
        self.last_spk_rec    = None
        self.epoch_log       = []

        self.phase_mgr  = PhaseManager.get()
        self.rate_bus   = SpikeRateBus.get(alpha=cfg.SPIKE_RATE_EWMA_ALPHA)
        self.reliability = ReliabilityTracker()

        # Push YAML runtime config into MemoryArbiter before any subsystem calls get()
        try:
            from runtime import MemoryArbiter, Zone
            MemoryArbiter.configure(
                dataset_ratio   = cfg.VRAM_DATASET_CACHE_RATIO,
                model_ratio     = cfg.VRAM_MODEL_PARAMS_RATIO,
                workspace_ratio = cfg.VRAM_KERNEL_WORKSPACE_RATIO,
                emergency_ratio = cfg.VRAM_EMERGENCY_RATIO,
                overhead_mb     = cfg.MEMORY_ALLOCATOR_OVERHEAD_MB,
            )
        except Exception:
            pass

        device_idx = (device.index or 0) if device.type == "cuda" else 0
        self.gpu_stats = GPUStats(device_idx=device_idx)

        use_amp = getattr(cfg, "USE_AMP", True) and device.type == "cuda"
        self.scaler           = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore[attr-defined]
        self.use_amp          = use_amp
        self.grad_accum_steps = max(1, getattr(cfg, "GRAD_ACCUM_STEPS", 1))

        lr_sched = getattr(cfg, "LR_SCHEDULER", "cosine")
        opt = getattr(self.model, "optimizer", None)
        self.scheduler = (CosineAnnealingLR(opt, T_max=cfg.EPOCHS) if opt is not None and lr_sched == "cosine" else None)

    def _report_kernel_compatibility(self) -> None:
        """Inspect the loaded architecture against the kernel's contract and print a status report.

        Kernel contract:
          - Input must be reshapeable to [B, N, T] float32 on CUDA
          - T = cfg.TIMESTEPS > 0
          - Model must produce spike-like outputs (SNN framework with binary activations)
          - N = flattened spatial input dimension

        Does NOT disable the kernel — reports only. The user decides based on the output.
        """
        model_name = type(self.model).__name__
        N = self.cfg.INPUT_SIZE
        T = self.cfg.TIMESTEPS
        B = self.cfg.BATCH_SIZE

        checks = []
        compatible = True

        # Timesteps
        if T > 0:
            checks.append(("T = timesteps", f"{T}", True))
        else:
            checks.append(("T = timesteps", f"{T} — kernel requires T > 0", False))
            compatible = False

        # Spatial input dimension.
        # N here is the flattened input to the kernel layer, not the total neuron count.
        # The full network structure (input → hidden layers → output) is owned by the SNN
        # framework above the kernel. The kernel processes one LIF layer at a time — the
        # input tensor reshaped to [B, N, T] where N = flattened spatial dimension of the data.
        # Hidden and output layers remain with Norse / SpikingJelly / snn_torch.
        if N > 0:
            checks.append((
                "N = kernel layer width",
                f"{N} (flattened input)  →  [{B}, {N}, {T}]  "
                f"|  full network: {self.cfg.network_structure}",
                True
            ))
        else:
            checks.append(("N = kernel layer width", f"{N} — cannot form [B, N, T]", False))
            compatible = False

        # CUDA
        if torch.cuda.is_available():
            checks.append(("CUDA", torch.cuda.get_device_name(0), True))
        else:
            checks.append(("CUDA", "not available — kernel requires GPU", False))
            compatible = False

        # SNN framework detection — looks at class name for known SNN frameworks.
        # If you add a new model class, ensure its name contains one of the markers below
        # or the report will flag it as unverified (kernel still runs — this is advisory only).
        SNN_MARKERS = ("snn", "lif", "norse", "spiking", "spike", "leaky", "neuron")
        name_lower = model_name.lower()
        is_snn = any(m in name_lower for m in SNN_MARKERS)
        if is_snn:
            checks.append(("Model framework", f"{model_name} — SNN type confirmed", True))
        else:
            checks.append((
                "Model framework",
                f"{model_name} — not recognised as SNN. "
                "Kernel expects binary spike output [B, N, T]. "
                "CNN / RNN / Transformer outputs are not compatible unless adapted to spike format.",
                False
            ))
            compatible = False

        # Kernel mode availability
        available_modes = [x for x in dir(self.kernel_module) if not x.startswith("_")]
        mode_fn_map = {
            "warp_oriented": "warp_oriented_forward",
            "temporal":      "temporal_forward",
            "basic":         "forward",
        }
        required_fn = mode_fn_map.get(self.kernel_mode, "forward")
        if required_fn in available_modes:
            checks.append(("Kernel mode", f"{self.kernel_mode}  →  {required_fn}()", True))
        else:
            checks.append(("Kernel mode", f"{self.kernel_mode}  →  {required_fn}() NOT FOUND in built module", False))
            compatible = False

        # Print report
        status = "COMPATIBLE" if compatible else "INCOMPATIBLE"
        print(f"\n[kernel] Architecture compatibility — {status}")
        print(f"[kernel]   Network : {self.cfg.network_structure}")
        for label, detail, ok in checks:
            mark = "ok" if ok else "!!"
            print(f"[kernel]   [{mark}] {label:<22}  {detail}")
        if not compatible:
            print("[kernel]   Kernel is ON but architecture does not fully meet the contract above.")
            print("[kernel]   Review the flagged items — kernel will attempt to run as configured.")
        print()

    def forward_pass(self, data: torch.Tensor) -> torch.Tensor:
        """Single forward pass. Routes through the custom CRSC CUDA kernel when
        kernel: ON is set in SNN_module.yaml, otherwise uses the framework model."""
        if not self.use_custom_kernel:
            result = self.model(data)
            self.rate_bus.push_dense(result.detach())
            return result
        if self.model.tensor_format() == "BT":
            data = data.permute(1, 0, 2, 3, 4).contiguous()
            return self.model(data)

        # Custom kernel expects [B, N, T]; typical neuromorphic data is [T, B, C, H, W]
        if data.dim() == 5:
            T, B, C, H, W = data.shape
            inp = data.view(T, B, C * H * W).permute(1, 2, 0).contiguous()  # [B, N, T]
        elif data.dim() == 4:
            T, B, C, N = data.shape
            inp = data.view(T, B, C * N).permute(1, 2, 0).contiguous()      # [B, N, T]
        else:
            result = self.model(data)
            self.rate_bus.push_dense(result.detach())
            return result

        B_sz, N_sz = inp.size(0), inp.size(1)
        if self._voltage_buf is None or self._voltage_buf.shape != (B_sz, N_sz):
            self._voltage_buf = torch.zeros(B_sz, N_sz, device=self.device)

        kernel    = self.kernel_module
        assert kernel is not None
        tau_inv = 1.0 - float(self.cfg.BETA)
        v_th    = float(self.cfg.THRESHOLD)

        if self.kernel_mode == "warp_oriented":
            spikes, _, _ = kernel.warp_oriented_forward(
                inp, self._voltage_buf, v_th, tau_inv,
                self.cfg.TARGET_BLOCKS_PER_SM,
            )
        elif self.kernel_mode == "temporal":
            spikes = kernel.temporal_forward(inp, self._voltage_buf, v_th, tau_inv)
        else:
            spikes = kernel.forward(inp, self._voltage_buf, v_th, tau_inv)

        self.rate_bus.push_dense(spikes.detach())
        fw_cfg    = self.cfg.active_fw_cfg
        fw        = self.cfg.FRAMEWORK
        if fw == "torch":
            tau_inv = 1.0 - float(fw_cfg["beta"])
        elif fw == "norse":
            tau_inv = float(fw_cfg["tau_mem_inv"])
        elif fw == "sj":
            tau_inv = 1.0 / float(fw_cfg["tau"])
        else:
            raise ValueError(f"Custom kernel: no tau_inv mapping for framework '{fw}'. Add it here.")
        threshold = float(fw_cfg["threshold"])
        spikes    = kernel.forward(
            inp, self._voltage_buf,
            threshold, tau_inv,
        )                                              # [B, N, T]
        return spikes.permute(2, 0, 1).contiguous()   # [T, B, N]

    def save_checkpoint(self, path: str):
        ckpt = {
            **self.model.get_state(),
            "scaler_state_dict": self.scaler.state_dict(),
            "loss_history":      self.loss_hist,
            "accuracy_history":  self.acc_hist,
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

    def train(self, checkpoint_dir: str, checkpoint_name: str = "best_model.pt", csv_path: str = "./outputs/data/training_results.csv") -> dict:

        self.reliability.start_session()

        epochs    = self.cfg.EPOCHS
        num_iters = self.cfg.ITERA
        accum     = self.grad_accum_steps

        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext())

        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        best_acc = 0.0

        for epoch in range(epochs):
            self.phase_mgr.enter(Phase.TRAIN)
            self.rate_bus.reset()
            self.model.train_mode()
            epoch_loss, epoch_acc, epoch_spike, n, step_count = 0.0, 0.0, 0.0, 0, 0
            t0 = time.perf_counter()
            self.gpu_stats.start_epoch()

            self.model.zero_grad()

            for i, (data, targets) in enumerate(self.train_loader):
                data    = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).long()

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
                        spk_rec      = self.forward_pass(data)
                        if reset is not None:
                            reset()
                        spk_rec_adv  = self.forward_pass(adv_data)
                        clean_logits = aggregate_spike_output(spk_rec.float())
                        adv_logits   = aggregate_spike_output(spk_rec_adv.float())
                        ce_loss  = F.cross_entropy(clean_logits, targets)
                        kl_loss  = F.kl_div(
                            F.log_softmax(adv_logits,           dim=1),
                            F.softmax(clean_logits.detach(),    dim=1),
                            reduction="batchmean",
                        )
                        hidden      = get_hidden_spike_recordings(self.model)
                        act_penalty = torch.zeros(1, device=self.device)
                        stdp_penalty = torch.zeros(1, device=self.device)
                        if self.cfg.ACTIVITY_REG_ENABLED:
                            act_penalty = activity_regularization(hidden,
                                min_rate   = self.cfg.ACTIVITY_REG_MIN_RATE,
                                max_rate   = self.cfg.ACTIVITY_REG_MAX_RATE,
                                lambda_low = self.cfg.ACTIVITY_REG_LAMBDA_LOW,
                                lambda_high= self.cfg.ACTIVITY_REG_LAMBDA_HIGH,
                            )
                        if self.cfg.STDP_ENABLED:
                            stdp_penalty = stdp_regularization(hidden,
                                output_spikes = spk_rec,
                                tau    = self.cfg.STDP_TAU,
                                A_plus = self.cfg.STDP_A_PLUS,
                                A_minus= self.cfg.STDP_A_MINUS,
                            )
                        loss_val = (ce_loss + self.cfg.TRADES_LAMBDA * kl_loss + act_penalty + stdp_penalty) / accum
                else:
                    with autocast_ctx:
                        spk_rec   = self.forward_pass(data)
                        task_loss = self.model.loss_fn(spk_rec, targets)
                        hidden    = get_hidden_spike_recordings(self.model)
                        if self.cfg.ACTIVITY_REG_ENABLED:
                            task_loss = task_loss + activity_regularization(hidden,
                                min_rate   = self.cfg.ACTIVITY_REG_MIN_RATE,
                                max_rate   = self.cfg.ACTIVITY_REG_MAX_RATE,
                                lambda_low = self.cfg.ACTIVITY_REG_LAMBDA_LOW,
                                lambda_high= self.cfg.ACTIVITY_REG_LAMBDA_HIGH,
                            )
                        if self.cfg.STDP_ENABLED:
                            task_loss = task_loss + stdp_regularization(hidden,
                                output_spikes = spk_rec,
                                tau    = self.cfg.STDP_TAU,
                                A_plus = self.cfg.STDP_A_PLUS,
                                A_minus= self.cfg.STDP_A_MINUS,
                            )
                        loss_val = task_loss / accum

                do_step = ((step_count + 1) % accum == 0)
                self.phase_mgr.enter(Phase.BACKWARD)
                self.model.backward_pass(loss_val, scaler=self.scaler, do_step=do_step)
                self.phase_mgr.enter(Phase.TRAIN)
                if do_step:
                    self.model.zero_grad()
                step_count += 1

                raw_loss   = loss_val.item() * accum
                logits     = clean_logits.detach() if self.cfg.TRADES_ENABLED else aggregate_spike_output(spk_rec.detach().float())
                acc        = (logits.argmax(dim=1) == targets).float().mean().item()
                spike_rate = self.rate_bus.rate  # EWMA from forward_pass — no extra .item() sync

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
                self.model.backward_pass(loss_val, scaler=self.scaler, do_step=True)
                self.model.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_duration = time.perf_counter() - t0
            gpu            = self.gpu_stats.end_epoch()

            # Record GPU pressure interruption if peak memory exceeded 90% of total
            if gpu.get("gpu_mem_peak_pct", 0) > 90:
                self.reliability.record_interruption(
                    f"GPU memory pressure — epoch {epoch+1} peak {gpu['gpu_mem_peak_pct']}% of VRAM"
                )
            train_loss     = epoch_loss / n
            train_acc      = epoch_acc  / n
            current_lr     = self.model.get_lr()

            train_spike = epoch_spike / n

            # Convert mean spike fraction to Hz: (spikes/timestep) / (seconds/timestep)
            window_s        = getattr(self.cfg, 'TEMPORAL_SLICE_DURATION_US', 15000) / 1e6
            timesteps       = getattr(self.cfg, 'TIMESTEPS', 25)
            firing_rate_hz  = train_spike * timesteps / window_s

            energy_j     = self.gpu_stats.gpu_energy_j(epoch_duration)
            avg_power_w  = energy_j / epoch_duration
            gpu_active_s = epoch_duration * gpu.get("gpu_util_avg_pct", 0.0) / 100.0

            self.epoch_log.append({
                "epoch":              epoch + 1,
                "train_loss":         round(train_loss,     4),
                "train_accuracy":     round(train_acc,      4),
                "spike_rate":         round(train_spike,    4),
                "firing_rate_hz":     round(firing_rate_hz, 2),
                "learning_rate":      round(current_lr,     6),
                "epoch_duration_s":   round(epoch_duration, 2),
                "gpu_active_s":       round(gpu_active_s,   2),
                "energy_j":           round(energy_j,       2),
                "avg_power_w":        round(avg_power_w,    2),
                **{k: v for k, v in gpu.items()},
            })

            # Sparse buffer diagnostics — shows AER memory savings vs dense equivalent
            spk_buf = getattr(self.model, "hidden_spk_buf", {})
            if spk_buf:
                sparse_kb  = sum(b.memory_bytes for b in spk_buf.values()) / 1024
                avg_rate   = sum(b.firing_rate   for b in spk_buf.values()) / len(spk_buf)
                dense_kb   = sparse_kb / avg_rate if avg_rate > 0 else 0.0
                buf_report = (f"sparse={sparse_kb:.1f}KB  dense_equiv={dense_kb:.1f}KB  "
                              f"rate={avg_rate * 100:.1f}%")
            else:
                buf_report = "no hooks"

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  • Train Loss     : {train_loss:.4f}")
            print(f"  • Train Accuracy : {train_acc * 100:.2f}%")
            print(f"  • Spike Rate     : {train_spike:.4f}")
            print(f"  • Firing Rate    : {firing_rate_hz:.2f} Hz")
            print(f"  • Spike Buffer   : {buf_report}")
            print(f"  • LR             : {current_lr:.6f}")
            print(f"  • Wall time      : {epoch_duration:.2f}s")
            print(f"  • GPU active     : {gpu_active_s:.2f}s  ({gpu.get('gpu_util_avg_pct', 0.0):.1f}% of wall time)")
            print(f"  • Energy         : {energy_j:.2f} J  ({avg_power_w:.1f} W avg)")
            if gpu:
                print(f"  • GPU Util       : avg {gpu['gpu_util_avg_pct']}%  peak {gpu['gpu_util_peak_pct']}%")
                print(f"  • GPU Memory     : {gpu['gpu_mem_peak_gb']} GB / {self.gpu_stats.total_memory_gb:.2f} GB  ({gpu['gpu_mem_peak_pct']}% peak)")

            if checkpoint_path is not None and train_acc > best_acc:
                best_acc = train_acc
                self.save_checkpoint(checkpoint_path)
                print(f"  • Checkpoint saved (best: {best_acc * 100:.2f}%)")

        overall = self.gpu_stats.summary()
        if overall:
            print("\nGPU Training Summary")
            print(f"  • Avg utilization  : {overall['overall_avg_util_pct']}%")
            print(f"  • Peak utilization : {overall['overall_peak_util_pct']}%")
            print(f"  • Peak VRAM used   : {overall['overall_peak_mem_gb']} GB / {overall['total_vram_gb']} GB  ({overall['overall_peak_mem_pct']}%)")

        self.write_csv(csv_path)

        self.reliability.end_session(outcome="success")
        self.reliability.print_report()

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
        os.makedirs(save_dir, exist_ok=True)

        # Normalise to [T, C]: for [T, B, C] take sample 0; for [B, C] treat each batch row as a timestep
        spk = self.last_spk_rec
        spk_sample = spk[:, 0, :] if spk.dim() == 3 else spk
        timesteps, neurons = spk_sample.numpy().nonzero()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(timesteps, neurons, s=2, c="black", marker="|")
        ax.set_title("Output Neuron Spike Raster  (last batch · sample 0)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron index")
        plt.tight_layout()
        path = os.path.join(save_dir, "spike_raster.png")
        plt.savefig(path, dpi=150)
        plt.show()
        print(f"[PLOT] Saved → {path}")
    
