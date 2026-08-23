from __future__ import annotations

import os
import csv
import time
from contextlib import contextmanager
import torch
import numpy as np
from skeleton import Settings
from learning.training import aggregate_spike_output
from learning.utilities import measure_dense_macs, compute_cv_isi
from event_data_workflow.system_monitor import PipelineMonitor, monitor
import matplotlib.pyplot as plt


# Neuromorphic energy-estimate constants — see SNN_GPU_Evaluation_Metrics.md
ENERGY_PER_SPIKE_PJ = 3.5
SYNOPS_ENERGY_PJ_PER_MAC = 4.6


class SNNTester:

    def __init__(self, model, test_loader, cfg: Settings, device: torch.device, visualize: bool = False):
        self.model       = model
        self.test_loader = test_loader
        self.cfg         = cfg
        self.device      = device
        self.num_classes = cfg.NUM_CLASSES
        self.batch_log   = []
        self.visualize   = visualize
        self.viz_window  = None  # lazily built on first use — see show_frame()
        self.pipeline_monitor = PipelineMonitor(enabled=getattr(cfg, "ENABLE_PIPELINE_MONITOR", True))

        # Deferred-sync accumulation — see run()'s docstring. Nothing here is
        # read back to host memory until the whole test pass finishes.
        self.fwd_events = []  # (start, end) CUDA event pairs
        self.dense_macs_per_layer = {}

    def forward_pass(self, data: torch.Tensor) -> torch.Tensor:
        """Single forward pass, adapting tensor layout to what the model expects."""
        if self.model.tensor_format() == "BT":
            data = data.permute(1, 0, 2, 3, 4).contiguous()
        return self.model(data)

    @contextmanager
    def timed(self, event_list: list):
        """Brackets one operation with a CUDA event pair, queued on the stream
        without blocking; the blocking readout (`elapsed_time()`) only happens
        once, in run()'s post-loop bulk sync."""
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        event_list.append((start, end))

    @staticmethod
    def elapsed_ms(pair) -> float:
        start, end = pair
        return start.elapsed_time(end)

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
        """Deferred-sync test pass: every per-batch quantity (spike counts,
        accuracy, predictions/targets, forward latency) is accumulated as a
        GPU-resident tensor or an un-synced CUDA event — no `.item()`/
        `.cpu()` call happens inside the batch loop. Everything, including
        every batch's printed line and the confusion matrix, is read back to
        host memory in ONE bulk transfer after the loop finishes. See
        SNN_GPU_Evaluation_Metrics.md and SNNTrainer.train()'s matching
        docstring for why."""
        monitor.enter_phase("testing")
        self.model.eval_mode()

        window_s = getattr(self.cfg, 'TEMPORAL_SLICE_DURATION_US', 15000) / 1e6
        timesteps_cfg = getattr(self.cfg, 'TIMESTEPS', 25)

        # Dense-MAC measurement for the SynOps estimate — SNN_GPU_Evaluation_Metrics.md §2.4/§4.4
        probe_data, _ = next(iter(self.test_loader))
        probe_data = probe_data.to(self.device)
        if self.model.tensor_format() == "BT":
            probe_data = probe_data.permute(1, 0, 2, 3, 4).contiguous()
        self.dense_macs_per_layer = measure_dense_macs(self.model, probe_data)

        all_preds_gpu, all_targets_gpu = [], []
        raw_batch_records: list[dict] = []
        last_activity_snapshot = {}
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

        print("\n[TEST RUN]")

        self.pipeline_monitor.start()
        self.pipeline_monitor.measure_idle_baseline()
        self.pipeline_monitor.set_phase("test_run")
        self.pipeline_monitor.reset_epoch_memory()
        t_run_start = time.perf_counter()

        # self.test_loader is a PrefetchedLoader (event_data_workflow.data_pipeline) —
        # batches arrive already device-resident.
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.test_loader):
                B = targets.size(0)
                T = data.size(0)

                with self.timed(self.fwd_events):
                    spk_rec = self.forward_pass(data)

                batch_spikes_gpu       = spk_rec.sum()
                batch_input_spikes_gpu = data.sum()

                logits    = aggregate_spike_output(spk_rec.float())
                preds_gpu = logits.argmax(dim=1)
                acc_gpu   = (preds_gpu == targets).float().mean()

                synops_gpu = torch.zeros((), device=self.device)
                for name, macs in self.dense_macs_per_layer.items():
                    buf = self.model.activity.buffers.get(name)
                    rate_t = buf.firing_rate_tensor() if buf is not None else None
                    if rate_t is not None:
                        synops_gpu = synops_gpu + rate_t * macs * T

                all_preds_gpu.append(preds_gpu)
                all_targets_gpu.append(targets)
                last_activity_snapshot = self.model.activity.recordings()

                raw_batch_records.append({
                    "batch_idx":           batch_idx,
                    "B": B, "T": T,
                    "batch_spikes":        batch_spikes_gpu,
                    "batch_input_spikes":  batch_input_spikes_gpu,
                    "acc":                 acc_gpu,
                    "synops":              synops_gpu,
                })

        t_run_elapsed = time.perf_counter() - t_run_start
        self.pipeline_monitor.stop()
        energy_report   = self.pipeline_monitor.phase_energy_report("test_run", t_run_elapsed)
        gpu             = energy_report["gpu"]
        gpu_energy_j    = energy_report["gpu_energy_j"]
        avg_power_w     = energy_report["avg_power_w"]
        dynamic_power_w = energy_report["dynamic_power_w"]
        gpu_diag        = energy_report["gpu_diag"]
        credit_assignment = self.model.credit_assignment()

        # ---- Bulk transfer: the single sync point for everything
        # accumulated above, now that the whole test pass has finished. ----
        all_preds   = torch.cat(all_preds_gpu).cpu()   if all_preds_gpu   else torch.empty(0, dtype=torch.long)
        all_targets = torch.cat(all_targets_gpu).cpu() if all_targets_gpu else torch.empty(0, dtype=torch.long)
        np.add.at(cm, (all_targets.numpy(), all_preds.numpy()), 1)

        fwd_latencies_ms = [self.elapsed_ms(p) for p in self.fwd_events]
        cv_isi            = compute_cv_isi(last_activity_snapshot)
        cv_isi_mean       = cv_isi.get("network_wide", 0.0)

        total_spikes             = 0
        total_input_spikes       = 0.0
        total_latency_ms         = 0.0
        total_samples            = 0
        total_energy_pj          = 0.0
        total_synops_pj          = 0.0
        per_sample_latencies_ms: list[float] = []

        for rec, latency_ms in zip(raw_batch_records, fwd_latencies_ms):
            B, T           = rec["B"], rec["T"]
            batch_spikes       = int(rec["batch_spikes"].item())
            batch_input_spikes = float(rec["batch_input_spikes"].item())
            acc                = rec["acc"].item()
            batch_synops_pj    = rec["synops"].item() * SYNOPS_ENERGY_PJ_PER_MAC

            possible_spikes = T * B * self.num_classes
            spike_rate      = batch_spikes / possible_spikes if possible_spikes > 0 else 0.0
            energy_pj       = batch_spikes * ENERGY_PER_SPIKE_PJ
            firing_rate_hz  = spike_rate * T / window_s if window_s > 0 else 0.0

            total_spikes       += batch_spikes
            total_input_spikes += batch_input_spikes
            total_latency_ms   += latency_ms
            total_samples      += B
            total_energy_pj    += energy_pj
            total_synops_pj    += batch_synops_pj
            # Per-sample latency isn't individually timed — only per-batch is —
            # so this repeats the batch's per-sample average once per sample.
            # Percentiles below are an approximation at batch-timing granularity,
            # not true per-sample measurement.
            per_sample_latencies_ms.extend([latency_ms / B] * B)

            self.batch_log.append({
                "batch":                 rec["batch_idx"],
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
                "synops_energy_pJ":      round(batch_synops_pj, 2),
                "credit_assignment":     credit_assignment,
            })

            print(f"  Batch {rec['batch_idx']:>3} | "
                  f"Acc: {acc * 100:.2f}% | "
                  f"Spikes: {batch_spikes:>6} | "
                  f"Rate: {spike_rate:.3f} ({firing_rate_hz:.1f} Hz) | "
                  f"Latency: {latency_ms:.1f}ms | "
                  f"Energy: {energy_pj:.1f}pJ")

        overall_acc            = (all_preds == all_targets).float().mean().item() if total_samples > 0 else 0.0
        avg_latency_ms         = total_latency_ms / len(self.batch_log) if self.batch_log else 0.0
        avg_latency_per_sample = total_latency_ms / total_samples if total_samples > 0 else 0.0
        median_latency_per_sample_ms = float(np.percentile(per_sample_latencies_ms, 50)) if per_sample_latencies_ms else 0.0
        # Tail latency, not just median/p90 — a real-time deadline is missed by
        # the slow outliers, not the typical case. See RealTimeLatencyEvaluator.
        p90_latency_per_sample_ms    = float(np.percentile(per_sample_latencies_ms, 90)) if per_sample_latencies_ms else 0.0
        p99_latency_per_sample_ms    = float(np.percentile(per_sample_latencies_ms, 99)) if per_sample_latencies_ms else 0.0
        throughput_samples_per_s     = total_samples / t_run_elapsed if t_run_elapsed > 0 else 0.0
        avg_spikes_per_sample  = total_spikes / total_samples if total_samples > 0 else 0.0
        avg_input_spikes_per_sample = total_input_spikes / total_samples if total_samples > 0 else 0.0
        # "framework ratio": input activity per output spike. Higher = the framework's
        # encoding/neuron dynamics compress more raw input activity into each output
        # spike; lower = the network stays closer to 1:1 with what it was shown.
        framework_ratio        = total_input_spikes / total_spikes if total_spikes > 0 else None
        avg_spike_rate         = total_spikes / (len(self.batch_log) * timesteps_cfg * self.num_classes) if self.batch_log else 0.0
        avg_firing_rate_hz     = avg_spike_rate * timesteps_cfg / window_s if window_s > 0 else 0.0
        energy_per_sample_pj   = total_energy_pj / total_samples if total_samples > 0 else 0.0
        synops_energy_per_sample_pj = total_synops_pj / total_samples if total_samples > 0 else 0.0
        class_metrics          = self.class_metrics(cm)
        gt_dist                = {c: int((all_targets == c).sum()) for c in range(self.num_classes)}
        pred_dist               = {c: int((all_preds   == c).sum()) for c in range(self.num_classes)}

        print("\n[TEST SUMMARY]")
        print(f"  • Framework               : {self.cfg.FRAMEWORK}")
        print(f"  • Credit Assignment       : {credit_assignment}")
        print(f"  • Overall Accuracy        : {overall_acc * 100:.2f}%")
        print(f"  • Total Samples           : {total_samples}")
        print(f"  • Total Spikes Activated  : {total_spikes:,}")
        print(f"  • Avg Spikes / Sample     : {avg_spikes_per_sample:.2f}")
        print(f"  • Total Input Activity    : {total_input_spikes:,.0f}")
        print(f"  • Avg Input / Sample      : {avg_input_spikes_per_sample:.2f}")
        print(f"  • Framework Ratio (in/out): {framework_ratio:.3f}" if framework_ratio is not None else "  • Framework Ratio (in/out): N/A (zero output spikes)")
        print(f"  • Avg Firing Rate         : {avg_firing_rate_hz:.2f} Hz")
        print(f"  • CV_ISI (network-wide)   : {cv_isi_mean:.3f}  (last batch)")
        print(f"  • Avg Batch Latency       : {avg_latency_ms:.2f} ms")
        print(f"  • Avg Latency / Sample    : {avg_latency_per_sample:.3f} ms")
        print(f"  • Median Latency / Sample : {median_latency_per_sample_ms:.3f} ms  (p50)")
        print(f"  • p90 Latency / Sample    : {p90_latency_per_sample_ms:.3f} ms")
        print(f"  • p99 Latency / Sample    : {p99_latency_per_sample_ms:.3f} ms")
        print(f"  • Throughput              : {throughput_samples_per_s:.1f} samples/s")
        print(f"  • Total Energy Estimate   : {total_energy_pj:.1f} pJ  (flat per-spike model)")
        print(f"  • Energy / Sample         : {energy_per_sample_pj:.2f} pJ")
        print(f"  • SynOps Energy Estimate  : {total_synops_pj:.1f} pJ  (per-layer, dense-MAC-weighted)")
        print(f"  • SynOps Energy / Sample  : {synops_energy_per_sample_pj:.2f} pJ")
        print(f"  • GPU Energy (actual)     : {gpu_energy_j * 1e3:.2f} mJ")
        print(f"  • Mean GPU Power (actual) : {avg_power_w:.1f} W")
        print(f"  • Mean Dynamic Power      : {dynamic_power_w:.1f} W  (idle baseline {self.pipeline_monitor.idle_power_w:.1f} W subtracted)")
        print(f"  • Peak GPU Memory        : {gpu['gpu_mem_peak_gb']} GB / {self.pipeline_monitor.total_memory_gb:.2f} GB  ({gpu['gpu_mem_peak_pct']}% peak)")
        print(f"  • GPU Utilization         : avg {gpu['gpu_util_avg_pct']}%  peak {gpu['gpu_util_peak_pct']}%")
        if gpu.get("gpu_idle_episodes", 0) > 0:
            print(f"  • GPU Idle               : {gpu['gpu_idle_episodes']} episode(s), {gpu['gpu_idle_total_s']:.1f}s total")
        print(f"  • Max Mem Reserved        : {gpu_diag.get('max_memory_reserved_gb', 0.0):.2f} GB")
        print(f"  • GPU Temp/Clock          : {gpu_diag['gpu_temp_c']}°C   SM {gpu_diag.get('sm_clock_mhz')} MHz   Mem {gpu_diag.get('mem_clock_mhz')} MHz")

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

        return {
            "framework":                 self.cfg.FRAMEWORK,
            "credit_assignment":         credit_assignment,
            "overall_accuracy":          overall_acc,
            "total_spikes":              total_spikes,
            "total_input_spikes":        total_input_spikes,
            "avg_input_spikes_per_sample": avg_input_spikes_per_sample,
            "framework_ratio":           framework_ratio,
            "avg_spikes_per_sample":     avg_spikes_per_sample,
            "avg_firing_rate_hz":        avg_firing_rate_hz,
            "cv_isi_mean":               cv_isi_mean,
            "avg_latency_ms":            avg_latency_ms,
            "avg_latency_per_sample_ms": avg_latency_per_sample,
            "median_latency_per_sample_ms": median_latency_per_sample_ms,
            "p90_latency_per_sample_ms":    p90_latency_per_sample_ms,
            "p99_latency_per_sample_ms":    p99_latency_per_sample_ms,
            "throughput_samples_per_s":     throughput_samples_per_s,
            "total_energy_pj":           total_energy_pj,
            "energy_per_sample_pj":      energy_per_sample_pj,
            "total_synops_energy_pj":    total_synops_pj,
            "synops_energy_per_sample_pj": synops_energy_per_sample_pj,
            "gpu_energy_j":              gpu_energy_j,
            "avg_power_w":               avg_power_w,
            "dynamic_power_w":           dynamic_power_w,
            "gpu_mem_peak_gb":           gpu.get("gpu_mem_peak_gb") if gpu else None,
            "gpu_util_avg_pct":          gpu.get("gpu_util_avg_pct") if gpu else None,
            "max_memory_reserved_gb":    gpu_diag.get("max_memory_reserved_gb"),
            "gpu_temp_c":                gpu_diag.get("gpu_temp_c"),
            "sm_clock_mhz":              gpu_diag.get("sm_clock_mhz"),
            "mem_clock_mhz":             gpu_diag.get("mem_clock_mhz"),
            "class_metrics":             class_metrics,
            "gt_distribution":           gt_dist,
            "pred_distribution":         pred_dist,
            "confusion_matrix":          cm,
        }
