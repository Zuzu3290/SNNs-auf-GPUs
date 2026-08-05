from __future__ import annotations
 
import os
import csv
import time
import torch
import numpy as np
from skeleton import Settings
from learning.training import aggregate_spike_output
from event_data_workflow.gpu_stats import GPUStats
from event_data_workflow.prefetch import AsyncGPUPrefetcher

# Energy per synaptic op — adjust for your target neuromorphic platform
ENERGY_PER_SPIKE_PJ = 3.5
 
class SNNTester:

    def __init__(self, model, test_loader, cfg: Settings, device: torch.device, visualize: bool = False):
        self.model       = model
        self.test_loader = test_loader
        self.cfg         = cfg
        self.device      = device
        self.num_classes = cfg.NUM_CLASSES
        self.batch_log   = []
        self.visualize   = visualize
        self._viz        = None  # lazily built on first use — see _show_frame()
        device_idx = (device.index or 0) if device.type == "cuda" else 0
        self.gpu_stats = GPUStats(device_idx=device_idx)

    def forward_pass(self, data: torch.Tensor) -> torch.Tensor:
        """Single forward pass, adapting tensor layout to what the model expects."""
        if self.model.tensor_format() == "BT":
            data = data.permute(1, 0, 2, 3, 4).contiguous()
        return self.model(data)

    def _show_frame(self, data: torch.Tensor, preds: torch.Tensor, tgts: torch.Tensor, batch_idx: int) -> None:
        """Live view of one sample from the batch: the input event-frame (summed over
        time and polarity) plus predicted vs ground-truth label. Opens an interactive
        matplotlib window on first call; updates it in place afterward (no new windows
        per batch). Classification datasets only — there's no meaningful single-frame
        view for the regression datasets yet (see docs/Haseeb-open-items.md)."""
        import matplotlib.pyplot as plt

        # data: [T, B, C, H, W] (or [B, T, ...] already normalized to T-first by
        # forward_pass's caller — here it's the raw batch, still whatever tensor_format
        # the DataLoader produced, which is always time-first per data_pipeline.py).
        frame = data[:, 0].sum(dim=(0, 1)).detach().cpu().numpy()  # sum T and C -> [H, W]

        if self._viz is None:
            plt.ion()
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(frame, cmap="hot")
            ax.set_axis_off()
            self._viz = (fig, ax, im)
        else:
            fig, ax, im = self._viz
            im.set_data(frame)
            im.set_clim(frame.min(), frame.max())

        fig, ax, im = self._viz
        ax.set_title(f"Batch {batch_idx} | Pred: {int(preds[0])}  GT: {int(tgts[0])}")
        fig.canvas.draw_idle()
        plt.pause(0.001)

    def close_visualization(self) -> None:
        if self._viz is not None:
            import matplotlib.pyplot as plt
            plt.close(self._viz[0])
            self._viz = None

    def class_metrics(self, cm: np.ndarray) -> list[dict]:
        total = cm.sum()
        rows  = []
        for c in range(self.num_classes):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            tn = int(total - tp - fp - fn)
 
            precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1          = (2 * precision * recall / (precision + recall)
                           if (precision + recall) > 0 else 0.0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            class_acc   = (tp + tn) / total if total > 0 else 0.0
 
            rows.append({
                "class":       c,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "accuracy":    round(class_acc,   4),
                "precision":   round(precision,   4),
                "recall":      round(recall,       4),
                "f1":          round(f1,           4),
                "specificity": round(specificity,  4),
            })
        return rows

    def write_csv(self, path: str):
        if not self.batch_log:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = list(self.batch_log[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.batch_log)
        print(f"[INFO] Test log saved -> {path}")

    def run(self, csv_path: str = "./outputs/data/test.csv") -> dict:
        self.model.eval_mode()

        # Duck-typed: only GPURecordingCache (gpu_memory cache tier) exposes
        # set_phase(). Widens its VRAM budget for eval (no gradients/optimizer
        # state active) — no-op for every other cache tier.
        set_phase = getattr(self.test_loader.dataset, "set_phase", None)
        if set_phase is not None:
            set_phase("eval")

        all_preds, all_targets = [], []
        total_spikes       = 0
        total_input_spikes = 0  # "framework ratio" — see run() docstring note below
        total_latency_ms   = 0.0
        total_samples      = 0
        total_energy_pj    = 0.0
        per_sample_latencies_ms: list[float] = []
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

        print("\n[TEST RUN]")

        self.gpu_stats.measure_idle_baseline()
        self.gpu_stats.start_epoch()
        t_run_start = time.perf_counter()

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(AsyncGPUPrefetcher(self.test_loader)):
                data    = data.to(self.device)
                targets = targets.to(self.device)
                B = targets.size(0)
                T = data.size(0)
 
                t0         = time.perf_counter()
                spk_rec    = self.forward_pass(data)
                latency_ms = (time.perf_counter() - t0) * 1000
 
                batch_spikes    = int(spk_rec.sum().item())
                # "framework ratio": how much event activity a framework's encoding/
                # neuron dynamics compress the raw input down to at the output. data
                # is event-count frames (not binary spikes) — this is total input
                # activity, the same-shaped quantity output-side batch_spikes measures.
                batch_input_spikes = float(data.sum().item())
                possible_spikes = T * B * self.num_classes
                spike_rate      = batch_spikes / possible_spikes
                energy_pj       = batch_spikes * ENERGY_PER_SPIKE_PJ

                window_s       = getattr(self.cfg, 'TEMPORAL_SLICE_DURATION_US', 15000) / 1e6
                firing_rate_hz = spike_rate * T / window_s
 
                logits = aggregate_spike_output(spk_rec.float())
                preds  = logits.argmax(dim=1).cpu()
                tgts   = targets.cpu()
                acc    = (preds == tgts).float().mean().item()
 
                np.add.at(cm, (tgts.numpy(), preds.numpy()), 1)
                all_preds.append(preds)
                all_targets.append(tgts)

                if self.visualize:
                    self._show_frame(data, preds, tgts, batch_idx)

                total_spikes       += batch_spikes
                total_input_spikes += batch_input_spikes
                total_latency_ms   += latency_ms
                total_samples      += B
                total_energy_pj    += energy_pj
                # Per-sample latency isn't individually timed — only per-batch is —
                # so this repeats the batch's per-sample average once per sample.
                # Percentiles below are an approximation at batch-timing granularity,
                # not true per-sample measurement.
                per_sample_latencies_ms.extend([latency_ms / B] * B)

                self.batch_log.append({
                    "batch":                 batch_idx,
                    "samples":               B,
                    "timesteps":             T,
                    "accuracy":              round(acc, 4),
                    "spikes_activated":      batch_spikes,
                    "input_spikes":          round(batch_input_spikes, 1),
                    "framework_ratio":       round(batch_input_spikes / batch_spikes, 4) if batch_spikes > 0 else None,
                    "possible_spikes":       possible_spikes,
                    "spike_rate":            round(spike_rate, 4),
                    "firing_rate_hz":        round(firing_rate_hz, 2),
                    "latency_ms":            round(latency_ms, 3),
                    "latency_per_sample_ms": round(latency_ms / B, 3),
                    "energy_pJ":             round(energy_pj, 2),
                })

                print(f"  Batch {batch_idx:>3} | "
                      f"Acc: {acc * 100:.2f}% | "
                      f"Spikes: {batch_spikes:>6} | "
                      f"Rate: {spike_rate:.3f} ({firing_rate_hz:.1f} Hz) | "
                      f"Latency: {latency_ms:.1f}ms | "
                      f"Energy: {energy_pj:.1f}pJ")
 
        t_run_elapsed = time.perf_counter() - t_run_start
        gpu           = self.gpu_stats.end_epoch()
        gpu_energy_j  = self.gpu_stats.gpu_energy_j(t_run_elapsed)
        avg_power_w     = gpu_energy_j / t_run_elapsed if gpu_energy_j is not None else None
        dynamic_power_w = self.gpu_stats.dynamic_power_w(avg_power_w) if avg_power_w is not None else None

        all_preds   = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        overall_acc            = (all_preds == all_targets).float().mean().item()
        avg_latency_ms         = total_latency_ms / len(self.batch_log)
        avg_latency_per_sample = total_latency_ms / total_samples
        median_latency_per_sample_ms = float(np.percentile(per_sample_latencies_ms, 50)) if per_sample_latencies_ms else 0.0
        p90_latency_per_sample_ms    = float(np.percentile(per_sample_latencies_ms, 90)) if per_sample_latencies_ms else 0.0
        # Tail latency, not just median/p90 — a real-time deadline is missed by
        # the slow outliers, not the typical case. See RealTimeLatencyEvaluator.
        p99_latency_per_sample_ms    = float(np.percentile(per_sample_latencies_ms, 99)) if per_sample_latencies_ms else 0.0
        throughput_samples_per_s     = total_samples / t_run_elapsed if t_run_elapsed > 0 else 0.0
        avg_spikes_per_sample  = total_spikes / total_samples
        avg_input_spikes_per_sample = total_input_spikes / total_samples
        # "framework ratio": input activity per output spike. Higher = the framework's
        # encoding/neuron dynamics compress more raw input activity into each output
        # spike; lower = the network stays closer to 1:1 with what it was shown.
        framework_ratio        = total_input_spikes / total_spikes if total_spikes > 0 else None
        avg_spike_rate         = total_spikes / (len(self.batch_log) * self.cfg.TIMESTEPS * self.num_classes) if self.batch_log else 0.0
        window_s               = getattr(self.cfg, 'TEMPORAL_SLICE_DURATION_US', 15000) / 1e6
        avg_firing_rate_hz     = avg_spike_rate * self.cfg.TIMESTEPS / window_s
        class_metrics          = self.class_metrics(cm)
        gt_dist                = {c: int((all_targets == c).sum()) for c in range(self.num_classes)}
        pred_dist              = {c: int((all_preds   == c).sum()) for c in range(self.num_classes)}

        print("\n[TEST SUMMARY]")
        print(f"  • Framework               : {self.cfg.FRAMEWORK}")
        print(f"  • Overall Accuracy        : {overall_acc * 100:.2f}%")
        print(f"  • Total Samples           : {total_samples}")
        print(f"  • Total Spikes Activated  : {total_spikes:,}")
        print(f"  • Avg Spikes / Sample     : {avg_spikes_per_sample:.2f}")
        print(f"  • Total Input Activity    : {total_input_spikes:,.0f}")
        print(f"  • Avg Input / Sample      : {avg_input_spikes_per_sample:.2f}")
        print(f"  • Framework Ratio (in/out): {framework_ratio:.3f}" if framework_ratio is not None else "  • Framework Ratio (in/out): N/A (zero output spikes)")
        print(f"  • Avg Firing Rate         : {avg_firing_rate_hz:.2f} Hz")
        print(f"  • Avg Batch Latency       : {avg_latency_ms:.2f} ms")
        print(f"  • Avg Latency / Sample    : {avg_latency_per_sample:.3f} ms")
        print(f"  • Median Latency / Sample : {median_latency_per_sample_ms:.3f} ms  (p50)")
        print(f"  • p90 Latency / Sample    : {p90_latency_per_sample_ms:.3f} ms")
        print(f"  • p99 Latency / Sample    : {p99_latency_per_sample_ms:.3f} ms")
        print(f"  • Throughput              : {throughput_samples_per_s:.1f} samples/s")
        print(f"  • Total Energy Estimate   : {total_energy_pj:.1f} pJ  (neuromorphic model)")
        print(f"  • Energy / Sample         : {total_energy_pj / total_samples:.2f} pJ")

        if gpu_energy_j is not None:
            neuromorphic_j = total_energy_pj * 1e-12
            gap = gpu_energy_j / neuromorphic_j if neuromorphic_j > 0 else float("inf")
            print(f"  • GPU Energy (actual)     : {gpu_energy_j * 1e3:.2f} mJ")
            print(f"  • Mean GPU Power (actual) : {avg_power_w:.1f} W")
            if self.gpu_stats.idle_power_w is not None:
                print(f"  • Mean Dynamic Power      : {dynamic_power_w:.1f} W  (idle baseline {self.gpu_stats.idle_power_w:.1f} W subtracted)")
            else:
                print(f"  • Mean Dynamic Power      : N/A  (idle baseline not measured — NVML unavailable)")
            print(f"  • Hardware Efficiency Gap : {gap:.2e}x  (GPU vs ideal neuromorphic silicon)")
        else:
            print(f"  • GPU Energy (actual)     : N/A  (install nvidia-ml-py for real power readings)")

        if gpu:
            print(f"  • Peak GPU Memory        : {gpu['gpu_mem_peak_gb']} GB / {self.gpu_stats.total_memory_gb:.2f} GB  ({gpu['gpu_mem_peak_pct']}% peak)")
            print(f"  • GPU Utilization         : avg {gpu['gpu_util_avg_pct']}%  peak {gpu['gpu_util_peak_pct']}%")

        print("\n  Per-class metrics:")
        for row in class_metrics:
            print(f"    Class {row['class']:>2} | "
                  f"Acc: {row['accuracy']:.3f} | "
                  f"P: {row['precision']:.3f}  "
                  f"R: {row['recall']:.3f}  "
                  f"F1: {row['f1']:.3f}  "
                  f"Spec: {row['specificity']:.3f}")
 
        print("\n  Label distribution (ground truth vs predicted):")
        for c in range(self.num_classes):
            print(f"    Class {c:>2} | GT: {gt_dist[c]:>5}  Pred: {pred_dist[c]:>5}")
 
        self.write_csv(csv_path)
        self.close_visualization()

        return {
            "framework":                 self.cfg.FRAMEWORK,
            "overall_accuracy":          overall_acc,
            "total_spikes":              total_spikes,
            "total_input_spikes":        total_input_spikes,
            "avg_input_spikes_per_sample": avg_input_spikes_per_sample,
            "framework_ratio":           framework_ratio,
            "avg_spikes_per_sample":     avg_spikes_per_sample,
            "avg_firing_rate_hz":        avg_firing_rate_hz,
            "avg_latency_ms":            avg_latency_ms,
            "avg_latency_per_sample_ms": avg_latency_per_sample,
            "median_latency_per_sample_ms": median_latency_per_sample_ms,
            "p90_latency_per_sample_ms":    p90_latency_per_sample_ms,
            "p99_latency_per_sample_ms":    p99_latency_per_sample_ms,
            "throughput_samples_per_s":     throughput_samples_per_s,
            "total_energy_pj":           total_energy_pj,
            "energy_per_sample_pj":      total_energy_pj / total_samples,
            "gpu_energy_j":              gpu_energy_j,
            "avg_power_w":               avg_power_w,
            "dynamic_power_w":           dynamic_power_w,
            "gpu_mem_peak_gb":           gpu.get("gpu_mem_peak_gb") if gpu else None,
            "gpu_util_avg_pct":          gpu.get("gpu_util_avg_pct") if gpu else None,
            "class_metrics":             class_metrics,
            "gt_distribution":           gt_dist,
            "pred_distribution":         pred_dist,
            "confusion_matrix":          cm,
        }