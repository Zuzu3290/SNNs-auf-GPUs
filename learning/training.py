from __future__ import annotations
import os
import csv
import time
import logging
import psutil
from pathlib import Path
from contextlib import contextmanager, nullcontext
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from skeleton import Settings
import matplotlib.pyplot as plt
from event_data_workflow.system_monitor import PipelineMonitor, monitor
from learning.utilities import (
    compute_cv_isi, firing_window_seconds, measure_dense_macs,
    spikes_per_neuron_per_inference, warm_up,
)
from skeleton import WorkflowSettings
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
    tensors already moved to CPU at their epoch's own boundary (see
    SNNTrainer.train()), so no CUDA sync happens here."""
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
        self.vram_current_hist = []  # per-iteration current allocated VRAM (GB) -- host-side counter, appended every iteration, no sync
        self.last_spk_rec    = None
        self.last_activity_snapshot: dict = {}
        self.epoch_log       = []
        self.timesteps: int | None = None  # set once in train(), reused to derive per-iteration firing rate at plot time
        self.window_s: float | None = None

        # GPU-resident accumulation through the run — see train()'s docstring
        # note on the deferred-sync design. Nothing here is read back to host
        # memory until training finishes.
        self.loss_hist_gpu       = []
        self.acc_hist_gpu        = []
        self.spike_rate_hist_gpu = []
        self.fwd_events          = []  # (start, end) CUDA event pairs, or (t0, t1) perf_counter pairs on CPU
        self.bwd_events          = []
        self.dense_macs_per_layer = {}
        # Gradient-norm tracking (opt-in, see cfg.COMPUTE_CAPACITY_METRICS): GPU-resident
        # running sums, synced ONCE at the end of train() -- never per-iteration, per
        # this file's deferred-sync rule (see train()'s own docstring).
        self.compute_capacity_metrics = getattr(cfg, "COMPUTE_CAPACITY_METRICS", False)
        self.grad_norm_sums: dict[str, torch.Tensor] = {}
        self.grad_norm_steps: int = 0
        self.grad_norm_means: dict[str, float] = {}

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

    def record_gradient_norms(self) -> None:
        """Accumulate this optimizer step's per-layer gradient norm as a GPU-resident
        running sum. Called only when self.compute_capacity_metrics is on, and only
        when do_step is True (a full effective batch's gradient is complete, not a
        partial accumulation step). No .item() here -- see finalize_gradient_norms()
        for the one sync, at the end of the whole run.
        """
        net = self.model.net
        for name in net.named_lif_layers():
            layer = net.dense_before(name)
            if layer is None or layer.weight.grad is None:
                continue
            norm = layer.weight.grad.detach().norm(2)
            if name in self.grad_norm_sums:
                self.grad_norm_sums[name] = self.grad_norm_sums[name] + norm
            else:
                self.grad_norm_sums[name] = norm
        self.grad_norm_steps += 1

    def finalize_gradient_norms(self) -> None:
        """The one sync point for gradient norms, called once after the whole training
        run finishes -- mirrors how loss_hist_gpu/acc_hist_gpu are read back once per
        epoch rather than per iteration, just at run granularity since layers.csv wants
        one mean per layer per run, not one per epoch."""
        if self.grad_norm_steps == 0:
            return
        self.grad_norm_means = {
            name: float((total / self.grad_norm_steps).item())
            for name, total in self.grad_norm_sums.items()
        }

    def measure_activity(self, probe_data: torch.Tensor, timesteps: int) -> dict:
        """One UNTIMED forward pass with spike recording ON, for the epoch's
        activity metrics: SynOps, CV_ISI and the sparse-vs-dense buffer report.

        Separated from the timed loop deliberately. ActivityMonitor's hooks fire on
        every LIF call, and while they force no CUDA sync they do detach, take a
        threading lock and — the part that actually matters — RETAIN one spike tensor
        per layer per timestep for the whole forward. At T=16 with two hooked layers
        that is 32 tensors held live inside the region being timed, which is real
        allocator pressure and can move a timing in ways that do not reproduce.

        So recording stays paused throughout training (see train()) and happens here
        instead, once per epoch, outside any timer. The cost is one extra forward pass
        per epoch; what is bought is a timed number that measures the network and
        nothing else.

        WHAT THIS CHANGES IN THE OUTPUT: SynOps and CV_ISI are now sampled once per
        epoch on one batch, rather than accumulated per iteration. The per-iteration
        SynOps series is therefore a broadcast of the epoch's value, exactly as GPU
        energy already was -- see iteration_series().
        """
        self.model.activity.resume()
        self.model.activity.clear()
        try:
            with torch.no_grad():
                self.forward_pass(probe_data)
            recordings = self.model.activity.recordings()

            synops = torch.zeros((), device=self.device)
            for name, macs in self.dense_macs_per_layer.items():
                buf = self.model.activity.buffers.get(name)
                rate_t = buf.firing_rate_tensor() if buf is not None else None
                if rate_t is not None:
                    synops = synops + rate_t * macs * timesteps
            snapshot = {k: (v.cpu() if v is not None else None) for k, v in recordings.items()}
        finally:
            # Straight back off before the next epoch's timed loop starts.
            self.model.activity.pause()
            self.model.activity.clear()
        return {"synops": float(synops.item()), "snapshot": snapshot}

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

    def iteration_series(self) -> dict:
        """Learning rate and GPU energy, broadcast to one value per iteration from each
        epoch's own already-recorded value (record["n"] iterations of that epoch share
        it) -- the scheduler only steps once per epoch, so there's no separate per-
        iteration LR to measure, and NVML power sampling isn't synced to iteration
        boundaries, so per-iteration energy is that epoch's total split evenly across
        its iterations, not a real per-iteration reading. Firing rate is a direct per-
        iteration derivation from self.spike_rate_hist, no broadcasting involved."""
        firing_rate_hz, learning_rate, gpu_energy_j, synops_pj = [], [], [], []
        for r in self.epoch_log:
            n = r["n"]
            learning_rate.extend([r["learning_rate"]] * n)
            gpu_energy_j.extend([r["energy_j_total"] / n if n else 0.0] * n)
            # Broadcast, like energy: SynOps is now sampled ONCE per epoch by the
            # untimed activity pass, because recording it per iteration meant the
            # ActivityMonitor's hooks ran inside the timed loop. So this is that
            # epoch's single measurement repeated, not a per-iteration reading.
            synops_pj.extend([r["synops_energy_pj"]] * n)

        # Spikes per neuron per inference (rate x T). Time-unit free, so always
        # available and always right -- this is the headline spike figure.
        spikes_per_inference = []
        if self.timesteps is not None:
            spikes_per_inference = [spikes_per_neuron_per_inference(s, self.timesteps)
                                    for s in self.spike_rate_hist]
        # Hz needs a real per-sample duration. window_s is None when that is not
        # knowable, and then the column is left empty rather than filled with a figure
        # computed against a guessed window.
        if self.window_s:
            firing_rate_hz = [v / self.window_s for v in spikes_per_inference]
        else:
            firing_rate_hz = [None] * len(spikes_per_inference)

        return {"firing_rate_hz": firing_rate_hz,
                "spikes_per_neuron_per_inference": spikes_per_inference,
                "learning_rate": learning_rate, "gpu_energy_j": gpu_energy_j,
                "synops_energy_pj": synops_pj}

    def write_batch_csv(self, path: str = "./outputs/data/batch_metrics.csv"):
        """One row per training iteration across the whole run -- every series plot_iteration_metrics()/plot_training() draw."""
        if not self.loss_hist:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        derived = self.iteration_series()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss", "accuracy", "spike_rate",
                              "spikes_per_neuron_per_inference", "firing_rate_hz",
                              "learning_rate", "vram_current_gb", "gpu_energy_j", "synops_energy_pj"])
            writer.writerows(zip(
                range(len(self.loss_hist)), self.loss_hist, self.acc_hist, self.spike_rate_hist,
                derived["spikes_per_neuron_per_inference"],
                derived["firing_rate_hz"], derived["learning_rate"], self.vram_current_hist,
                derived["gpu_energy_j"], derived["synops_energy_pj"],
            ))
        print(f"[INFO] Iteration metrics log saved -> {path}")

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
        # None means "not resolved yet" (a caller that skipped resolve_iterations).
        # Settle it here rather than letting `i >= None` raise mid-epoch.
        num_iters = self.cfg.ITERA
        if num_iters is None:
            num_iters = self.cfg.resolve_iterations(self.train_loader)
        print(f"  [TRAIN] {epochs} epoch(s) x {num_iters} iterations "
              f"(batch {self.cfg.BATCH_SIZE})")
        accum     = self.grad_accum_steps
        # None when the real per-sample duration is not knowable -- Hz is then reported
        # as unavailable rather than computed against a guessed window. See
        # utilities.firing_window_seconds for why the old fixed 15 ms was ~20x wrong.
        window_s  = firing_window_seconds(self.cfg, WorkflowSettings(config=self.cfg.config))

        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext())

        # Dense-MAC measurement for the SynOps estimate — SNN_GPU_Evaluation_Metrics.md §2.4/§4.4.
        # self.train_loader is a PrefetchedLoader — probe_data is already device-resident.
        probe_data, _ = next(iter(self.train_loader))
        timesteps = probe_data.shape[0]  # loader yields [T, B, C, H, W] — the real BPTT unroll length, not a config value
        self.timesteps, self.window_s = timesteps, window_s
        if self.model.tensor_format() == "BT":
            probe_data = probe_data.permute(1, 0, 2, 3, 4).contiguous()
        self.dense_macs_per_layer = measure_dense_macs(self.model, probe_data)

        # ---- warm-up: untimed, weights untouched -----------------------------------
        # Before the timed loop, so CUDA kernel compilation and allocator growth are not
        # charged to training -- and, in a multi-framework run, not charged to whichever
        # framework happens to run first.
        self.warmup = warm_up(self.model, probe_data, getattr(self.cfg, "WARMUP_ITERATIONS", 5))

        # Spike recording OFF for every timed region from here on. The hooks retain a
        # tensor per layer per timestep, which is allocator pressure inside the window
        # being timed. measure_activity() turns them back on once per epoch, untimed.
        self.model.activity.pause()
        if self.warmup["iterations"]:
            print(f"  [WARM-UP] {self.warmup['iterations']} untimed forward+backward passes  "
                  f"(weights unchanged: {self.warmup['weights_unchanged']})")
            if not self.warmup["weights_unchanged"]:
                raise RuntimeError(
                    "[WARM-UP] the model's weights changed during warm-up. No optimizer "
                    "step is taken there, so this means something else is mutating them "
                    "-- the timed epochs would not start from the seeded weights."
                )

        raw_epoch_records: list[dict] = []

        for epoch in range(epochs):
            self.model.train_mode()
            epoch_loss_sum   = torch.zeros((), device=self.device)
            epoch_acc_sum    = torch.zeros((), device=self.device)
            epoch_spike_sum  = torch.zeros((), device=self.device)
            n, step_count = 0, 0
            t0 = time.perf_counter()
            self.pipeline_monitor.set_phase(f"epoch_{epoch}")
            self.pipeline_monitor.reset_epoch_memory()

            self.model.zero_grad()
            mem_breakdown: dict = {}

            # self.train_loader is a PrefetchedLoader (event_data_workflow.data_pipeline) —
            # batches arrive already device-resident, no .to(device) needed below.
            for i, (data, targets) in enumerate(self.train_loader):
                # Checked BEFORE the work, not after. Previously this sat at the end of
                # the body, so the batch at i == num_iters had already been trained on
                # and an epoch ran num_iters + 1 iterations. Invisible on a calibrated
                # run (ITERA == len(loader), so the loader exhausts first and the check
                # never fires) but a 33% overrun when iterations_per_epoch is set small
                # for a diagnostic -- exactly when the timings are being read.
                if i >= num_iters:
                    break
                if i % 20 == 0:
                    # One .item() sync here, same cadence as this print already used --
                    # not a new per-iteration cost. Shows the last up-to-20 iterations'
                    # accuracy (whatever has accumulated so far this epoch, or trailing
                    # from the previous epoch at i==0), not this iteration's own -- its
                    # accuracy hasn't been computed yet at this point in the loop.
                    recent_n = min(20, len(self.acc_hist_gpu))
                    acc_display = (f"{torch.stack(self.acc_hist_gpu[-recent_n:]).mean().item() * 100:.2f}%"
                                   if recent_n else "n/a")
                    # available_gb is psutil's estimate of memory a new process could
                    # get without swapping -- it already treats reclaimable OS disk
                    # cache as usable, unlike a raw "how much is used" figure. Printed
                    # here so a live Colab run shows the one number that actually
                    # distinguishes "healthy, just caching files" from "genuinely
                    # running out" without needing a second cell (which Colab won't run
                    # concurrently with this one anyway).
                    ram_gb = psutil.virtual_memory().available / (1024 ** 3)
                    print(f"  [epoch {epoch + 1}/{epochs}] iter {i}/{num_iters}  "
                          f"acc (last {recent_n} iter): {acc_display}  "
                          f"RAM available: {ram_gb:.2f}GB  "
                          f"({time.perf_counter() - t0:.1f}s elapsed)", flush=True)
                targets = targets.long()
                measure_mem = i == 0 and self.device.type == "cuda"  # reset/read are host-side counters, not a stream sync -- cheap, but sampled once/epoch anyway
                if measure_mem:
                    torch.cuda.reset_peak_memory_stats(self.device)

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

                if measure_mem:
                    mem_breakdown["forward_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    torch.cuda.reset_peak_memory_stats(self.device)

                do_step = ((step_count + 1) % accum == 0)
                with self.timed(self.bwd_events):
                    self.model.backward_pass(loss_val, scaler=self.scaler, do_step=do_step)
                if measure_mem:
                    mem_breakdown["backward_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    mem_breakdown["weights_gb"] = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 3)
                    mem_breakdown["gradients_gb"] = sum(p.grad.numel() * p.grad.element_size() for p in self.model.parameters() if p.grad is not None) / (1024 ** 3)
                if do_step and self.compute_capacity_metrics:
                    self.record_gradient_norms()
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

                # SynOps is NOT computed here any more. It needs the ActivityMonitor's
                # per-timestep recordings, and those are paused throughout this timed
                # loop so the hooks' tensor retention cannot affect the timing. It is
                # measured once per epoch instead, in measure_activity(), outside any
                # timer -- see that method for the full reasoning and what it costs.
                #
                # Host-side allocator counter, not a tensor -- no .item()/sync needed, and it's the
                # quantity that actually varies iteration to iteration (activation size tracks each
                # batch's real, unpadded sample lengths); weights/gradients/optimizer state don't
                # change shape once training starts, so they're measured once, not every iteration.
                self.vram_current_hist.append(torch.cuda.memory_allocated(self.device) / (1024 ** 3) if self.device.type == "cuda" else 0.0)

                epoch_loss_sum   = epoch_loss_sum   + per_batch_loss
                epoch_acc_sum    = epoch_acc_sum    + per_batch_acc
                epoch_spike_sum  = epoch_spike_sum  + per_batch_spike
                n += 1


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
            # Guarded: on a CPU-only torch build this raises "Torch not compiled with
            # CUDA enabled" and the whole run dies at the first epoch boundary, which
            # made a laptop run impossible. On CUDA the condition is true, so the
            # behaviour there is unchanged.
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            fwd_latencies_ms = [self.elapsed_ms(p) for p in self.fwd_events]
            bwd_latencies_ms = [self.elapsed_ms(p) for p in self.bwd_events]
            self.fwd_events, self.bwd_events = [], []

            if self.device.type == "cuda":
                opt_state = getattr(self.model.optimizer, "state", {})
                mem_breakdown["optimizer_state_gb"] = sum(
                    t.numel() * t.element_size() for s in opt_state.values() for t in s.values() if torch.is_tensor(t)
                ) / (1024 ** 3)

            # Same reasoning, same fix, for the per-batch loss/accuracy/spike-rate
            # scalars: read back and clear each epoch's GPU-resident list here
            # instead of holding all 5 epochs' worth (1175 tensors each) until
            # the very end.
            self.loss_hist.extend(t.item() for t in self.loss_hist_gpu)
            self.acc_hist.extend(t.item() for t in self.acc_hist_gpu)
            self.spike_rate_hist.extend(t.item() for t in self.spike_rate_hist_gpu)
            self.loss_hist_gpu, self.acc_hist_gpu, self.spike_rate_hist_gpu = [], [], []

            # ---- untimed activity pass ------------------------------------------
            # After the epoch's timer has stopped and after the energy window has
            # closed, so nothing measured above is affected by the recording hooks.
            activity = self.measure_activity(probe_data, timesteps)

            if self.scheduler is not None:
                self.scheduler.step()

            epoch_duration = time.perf_counter() - t0
            epoch_phase    = f"epoch_{epoch}"
            energy_report  = self.pipeline_monitor.phase_energy_report(epoch_phase, epoch_duration)  # None fields on CPU-only or without NVML
            gpu             = energy_report["gpu"]
            gpu_diag        = energy_report["gpu_diag"]
            # TOTAL energy: idle draw INCLUDED. Scales with duration, so a slower
            # framework reports more of it even at identical power draw.
            energy_j         = energy_report["gpu_energy_j"] or 0.0
            # ABOVE idle: what the computation itself cost. Falls back to the total
            # when no idle baseline was measured, in which case idle_power_w is None.
            dynamic_energy_j = energy_report["gpu_dynamic_energy_j"] or 0.0
            avg_power_w      = energy_report["avg_power_w"] or 0.0
            dynamic_power_w  = energy_report["dynamic_power_w"] or 0.0
            idle_power_w     = energy_report["idle_power_w"]
            gpu_active_s    = epoch_duration * gpu.get("gpu_util_avg_pct", 0.0) / 100.0

            # .item()/.cpu() here (sync already paid above for fwd/bwd_latencies_ms), not held
            # GPU-resident until the whole run finishes -- same class of fix as fwd/bwd_events below.
            raw_epoch_records.append({
                "epoch":              epoch + 1,
                "n":                  n,
                "fwd_latencies_ms":   fwd_latencies_ms,
                "bwd_latencies_ms":   bwd_latencies_ms,
                "epoch_loss_sum":     epoch_loss_sum.item(),
                "epoch_acc_sum":      epoch_acc_sum.item(),
                "epoch_spike_sum":    epoch_spike_sum.item(),
                # Sampled once per epoch by the untimed pass above, not accumulated
                # per iteration -- see measure_activity().
                "epoch_synops":       activity["synops"],
                "activity_snapshot":  activity["snapshot"],
                "epoch_duration":     epoch_duration,
                "gpu":                gpu,
                "gpu_diag":           gpu_diag,
                "energy_j":           energy_j,
                "dynamic_energy_j":   dynamic_energy_j,
                "avg_power_w":        avg_power_w,
                "dynamic_power_w":    dynamic_power_w,
                "idle_power_w":       idle_power_w,
                "gpu_active_s":       gpu_active_s,
                "current_lr":         self.model.get_lr(),
                "mem_breakdown":      mem_breakdown,
            })

        # ---- Bulk transfer: the single sync point for everything
        # accumulated above, now that every epoch has finished training. ----
        self.finalize_epoch_reports(raw_epoch_records, epochs, timesteps, window_s)
        self.finalize_gradient_norms()

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
        # Beside training_results.csv, wherever that went -- NOT the module default.
        # Called bare, it wrote to ./outputs/data/batch_metrics.csv even on a run whose
        # --results-root pointed at mounted Drive, so on Colab that one file stayed on
        # the runtime and vanished with it while every other artefact was saved.
        self.write_batch_csv(str(Path(csv_path).parent / "batch_metrics.csv"))

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

            train_loss  = (record["epoch_loss_sum"]  / n) if n else 0.0
            train_acc   = (record["epoch_acc_sum"]   / n) if n else 0.0
            train_spike = (record["epoch_spike_sum"] / n) if n else 0.0
            # One batch's SynOps, sampled by the untimed pass -- a PER-BATCH figure,
            # not an epoch total. It used to be summed over every iteration, which
            # made it scale with iteration count and so not comparable between runs
            # of different length.
            synops_pj   = record["epoch_synops"] * SYNOPS_ENERGY_PJ_PER_MAC

            # Time-unit free, so always valid. This is the headline spike figure and
            # the one directly comparable with published SNN numbers.
            spikes_per_inference = spikes_per_neuron_per_inference(train_spike, timesteps)
            # None when the real per-sample duration is not knowable -- reported as
            # unavailable rather than computed against a guessed window.
            firing_rate_hz = (spikes_per_inference / window_s) if window_s else None

            # Kept on the trainer so a caller can build per-layer rows (layers.csv)
            # from the measurement this epoch already took, instead of paying for an
            # extra pass. Hooked layers only -- lif_out is not hooked.
            self.last_activity_snapshot = record["activity_snapshot"]

            cv_isi      = compute_cv_isi(record["activity_snapshot"])
            cv_isi_mean = cv_isi.get("network_wide", 0.0)
            buf_report  = buffer_report(record["activity_snapshot"])

            avg_fwd_ms = sum(epoch_fwd) / len(epoch_fwd) if epoch_fwd else 0.0
            avg_bwd_ms = sum(epoch_bwd) / len(epoch_bwd) if epoch_bwd else 0.0

            gpu      = record["gpu"]
            gpu_diag = record["gpu_diag"]
            mem      = record["mem_breakdown"]

            is_best_epoch = train_acc > best_acc_so_far
            if is_best_epoch:
                best_acc_so_far = train_acc

            self.epoch_log.append({
                "epoch":               record["epoch"],
                "n":                   n,
                "train_loss":          round(train_loss, 4),
                "train_accuracy":      round(train_acc, 4),
                "best_accuracy_so_far": round(best_acc_so_far, 4),
                "is_best_epoch":       is_best_epoch,
                "spike_rate":          round(train_spike, 4),
                "spikes_per_neuron_per_inference": round(spikes_per_inference, 4),
                "firing_rate_hz":      (round(firing_rate_hz, 2)
                                        if firing_rate_hz is not None else None),
                # Recorded because calibrate_batch_size sizes this from live VRAM, so
                # it legitimately differs between machines and between runs. Batch size
                # is the largest single lever on wall-clock time, so two rows can only
                # be compared on speed once this column is known to match.
                "batch_size":          getattr(self.cfg, "BATCH_SIZE", None),
                "batch_size_calibrated": getattr(self.cfg, "CALIBRATE_BATCH_SIZE", None),
                "learning_rate":       round(record["current_lr"], 6),
                "epoch_duration_s":    round(record["epoch_duration"], 2),
                "gpu_active_s":        round(record["gpu_active_s"], 2),
                # Named so a CSV reader cannot mistake one basis for the other.
                "energy_j_total":      round(record["energy_j"], 2),
                "energy_j_dynamic":    round(record["dynamic_energy_j"], 2),
                "avg_power_w":         round(record["avg_power_w"], 2),
                "dynamic_power_w":     round(record["dynamic_power_w"], 2),
                # The baseline both dynamic figures rest on. None = never measured,
                # which is also when dynamic equals total.
                "idle_power_w":        (round(record["idle_power_w"], 2)
                                        if record["idle_power_w"] is not None else None),
                "forward_latency_ms":  round(avg_fwd_ms, 3),
                "backward_latency_ms": round(avg_bwd_ms, 3),
                "cv_isi_mean":         round(cv_isi_mean, 4),
                "credit_assignment":   credit_assignment,
                "synops_energy_pj":    round(synops_pj, 2),
                **{k: v for k, v in gpu_diag.items()},
                **{k: v for k, v in gpu.items()},
                **{f"model_{k}": round(v, 4) for k, v in mem.items()},
            })

            print(f"\nEpoch {record['epoch']}/{epochs}")
            print(f"  • Train Loss     : {train_loss:.4f}")
            print(f"  • Train Accuracy : {train_acc * 100:.2f}%" + ("  (best so far)" if is_best_epoch else ""))
            print(f"  • Spike Rate     : {train_spike:.4f}")
            print(f"  • Spikes/neuron  : {spikes_per_inference:.4f} per inference  (rate x T)")
            if firing_rate_hz is not None:
                print(f"  • Firing Rate    : {firing_rate_hz:.2f} Hz  (over a {window_s * 1000:.1f} ms sample)")
            else:
                print("  • Firing Rate    : n/a -- the real per-sample duration is not known. "
                      "Set framing.sample_duration_us to get Hz.")
            print(f"  • Spike Buffer   : {buf_report}")
            print(f"  • Fwd/Bwd Latency: {avg_fwd_ms:.3f} ms / {avg_bwd_ms:.3f} ms  ({credit_assignment})")
            print(f"  • CV_ISI (net)   : {cv_isi_mean:.3f}")
            print(f"  • SynOps Energy  : {synops_pj:.2f} pJ (estimated, per epoch)")
            print(f"  • LR             : {record['current_lr']:.6f}")
            print(f"  • Wall time      : {record['epoch_duration']:.2f}s")
            print(f"  • GPU active     : {record['gpu_active_s']:.2f}s  ({gpu.get('gpu_util_avg_pct', 0.0):.1f}% of wall time)")
            print(f"  • Energy (total) : {record['energy_j']:.2f} J  ({record['avg_power_w']:.1f} W avg, idle draw INCLUDED)")
            if record["idle_power_w"] is not None:
                print(f"  • Energy (dynamic): {record['dynamic_energy_j']:.2f} J  "
                      f"({record['dynamic_power_w']:.1f} W above a {record['idle_power_w']:.1f} W idle baseline)")
                print("                      ^ total scales with runtime; dynamic isolates the work")
            else:
                print("  • Energy (dynamic): not available -- no idle baseline was measured, "
                      "so the total above is all there is")
            if gpu:
                print(f"  • GPU Util       : avg {gpu['gpu_util_avg_pct']}%  peak {gpu['gpu_util_peak_pct']}%")
                print(f"  • GPU Memory     : {gpu['gpu_mem_peak_gb']} GB / {self.pipeline_monitor.total_memory_gb:.2f} GB  ({gpu['gpu_mem_peak_pct']}% peak)")
                if gpu.get("gpu_idle_episodes", 0) > 0:
                    print(f"  • GPU Idle       : {gpu['gpu_idle_episodes']} episode(s), {gpu['gpu_idle_total_s']:.1f}s total "
                          f"— CPU-side data prep couldn't keep up with the GPU this often")
            print(f"  • Max Mem Reserved: {gpu_diag.get('max_memory_reserved_gb', 0.0):.2f} GB")
            if mem:
                weights_mb, grad_mb, optim_mb = (mem.get(k, 0.0) * 1024 for k in ("weights_gb", "gradients_gb", "optimizer_state_gb"))
                print(f"  • Model VRAM     : weights {weights_mb:.2f}MB + grad {grad_mb:.2f}MB + "
                      f"optim {optim_mb:.2f}MB = {weights_mb + grad_mb + optim_mb:.2f}MB static, held between iterations "
                      f"(excludes the input batch/prefetch queue)")
                print(f"  • Fwd/Bwd Peak   : {mem.get('forward_peak_gb', 0.0):.3f}GB / {mem.get('backward_peak_gb', 0.0):.3f}GB  "
                      f"(cumulative allocator high-water mark during each phase -- already includes the static figure above, not additive with it)")
            if "gpu_temp_c" in gpu_diag:
                print(f"  • GPU Temp/Clock : {gpu_diag['gpu_temp_c']}°C   SM {gpu_diag.get('sm_clock_mhz')} MHz   Mem {gpu_diag.get('mem_clock_mhz')} MHz")

    def plot_training(self, save_dir: str = "./outputs/plots") -> None:
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
        print(f"[PLOT] Saved -> {path}")

    def plot_iteration_metrics(self, save_dir: str = "./outputs/plots") -> None:
        """VRAM, firing rate, learning rate, and both energy readings, one point per
        iteration -- a single index counting up across every epoch back to back (epoch 2's
        iteration 0 continues from epoch 1's last iteration, not a reset).

        VRAM comes from self.vram_current_hist, appended every iteration inside train() at
        effectively no cost (a host-side counter read, no CUDA sync). Firing rate,
        learning rate, GPU energy and SynOps come from self.iteration_series() -- see
        that method for which are exact per-iteration readings and which are one
        per-epoch measurement broadcast across that epoch. Each metric gets its own
        figure and its own file."""
        if not self.epoch_log or not self.loss_hist:
            print("[PLOT] No iteration data — run train() first.")
            return
        os.makedirs(save_dir, exist_ok=True)
        iters = list(range(len(self.loss_hist)))
        derived = self.iteration_series()

        def save(fig, filename: str) -> None:
            plt.tight_layout()
            path = os.path.join(save_dir, filename)
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[PLOT] Saved -> {path}")

        # Current allocated VRAM genuinely fluctuates iteration to iteration (each batch's
        # real, unpadded event count differs) -- plotted as a real series, not a flat
        # per-epoch peak. Weights/gradients/optimizer state don't change shape once
        # training starts, so they're reference lines from one measurement, not redrawn
        # as if remeasured every iteration.
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(iters, self.vram_current_hist, linewidth=0.6, color="tab:blue", label="current allocated")
        ref = self.epoch_log[0]
        ax.axhline(ref.get("model_weights_gb", 0.0), color="tab:green", linestyle="--",
                   label=f"weights ({ref.get('model_weights_gb', 0.0):.3f} GB, static)")
        ax.axhline(ref.get("model_gradients_gb", 0.0), color="tab:red", linestyle="--",
                   label=f"gradients ({ref.get('model_gradients_gb', 0.0):.3f} GB, static)")
        ax.axhline(ref.get("model_optimizer_state_gb", 0.0), color="tab:purple", linestyle="--",
                   label=f"optimizer state ({ref.get('model_optimizer_state_gb', 0.0):.3f} GB, static)")
        ax.set_title("VRAM per Iteration")
        ax.set_xlabel("Iteration (cumulative across epochs)")
        ax.set_ylabel("VRAM (GB)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        save(fig, "vram_breakdown.png")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(iters, derived["firing_rate_hz"], linewidth=0.6, color="tab:orange")
        ax.set_title("Firing Rate per Iteration")
        ax.set_xlabel("Iteration (cumulative across epochs)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.grid(True, alpha=0.3)
        save(fig, "firing_rate.png")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(iters, derived["learning_rate"], linewidth=0.8, color="tab:red")
        ax.set_title("Learning Rate per Iteration")
        ax.set_xlabel("Iteration (cumulative across epochs)")
        ax.set_ylabel("Learning Rate")
        ax.grid(True, alpha=0.3)
        save(fig, "learning_rate.png")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(iters, derived["gpu_energy_j"], linewidth=0.8, color="tab:purple")
        ax.set_title("GPU Energy per Iteration (epoch total ÷ iteration count)")
        ax.set_xlabel("Iteration (cumulative across epochs)")
        ax.set_ylabel("GPU Energy (J)")
        ax.grid(True, alpha=0.3)
        save(fig, "gpu_energy.png")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(iters, self.iteration_series()["synops_energy_pj"], linewidth=0.6, color="tab:brown")
        ax.set_title("SynOps Energy per Iteration")
        ax.set_xlabel("Iteration (cumulative across epochs)")
        ax.set_ylabel("SynOps Energy (pJ)")
        ax.grid(True, alpha=0.3)
        save(fig, "synops_energy.png")

    def plot_raster(self, save_dir: str = "./outputs/plots") -> None:
        if self.last_spk_rec is None:
            print("[PLOT] No spike data — run train() first.")
            return
        os.makedirs(save_dir, exist_ok=True)

        # Normalise to [T, C]: for [T, B, C] take sample 0; for [B, C] treat each batch row as a timestep
        # last_spk_rec is GPU-resident (deferred-sync design) — .cpu() happens
        # here, once, on explicit user request to plot, not inside the loop.
        spk = self.last_spk_rec.cpu()
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
        plt.close(fig)
        print(f"[PLOT] Saved -> {path}")
