from __future__ import annotations

import os
import csv
import time
from contextlib import contextmanager
import torch
import numpy as np
from skeleton import Settings
from learning.training import aggregate_spike_output
from learning.utilities import measure_dense_macs
from event_data_workflow.system_monitor import PipelineMonitor, monitor


# A synaptic operation is one addition, 0.9 pJ at 45 nm (Horowitz, ISSCC 2014)
SYNOPS_ENERGY_PJ = 0.9


class SNNTester:

    def __init__(self, model, test_loader, cfg: Settings, device: torch.device):
        self.model       = model
        self.test_loader = test_loader
        self.cfg         = cfg
        self.device      = device
        self.num_classes = cfg.NUM_CLASSES
        self.pipeline_monitor = PipelineMonitor()

        # Deferred-sync accumulation — see run()'s docstring. Nothing here is
        # read back to host memory until the whole test pass finishes.
        self.fwd_events = []  # (start, end) CUDA event pairs
        self.dense_macs_per_layer = {}

    def forward_pass(self, data: torch.Tensor) -> torch.Tensor:
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
                "ground_truth_count": int(cm[c, :].sum()),
                "predicted_count":    int(cm[:, c].sum()),
            })
        return rows

    def write_csv(self, path: str, rows: list[dict]):
        """Per-class test metrics, one row per class."""
        if not rows:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Per-class test metrics saved -> {path}")

    def run(self, csv_path: str = "./outputs/data/test.csv") -> dict:
        """Deferred-sync test pass: every per-batch quantity (spike counts,
        accuracy, predictions/targets, forward latency) is accumulated as a
        GPU-resident tensor or an un-synced CUDA event — no `.item()`/
        `.cpu()` call happens inside the batch loop. Everything, including
        every batch's printed line and the confusion matrix, is read back to
        host memory in ONE bulk transfer after the loop finishes."""
        monitor.enter_phase("testing")
        self.model.eval_mode()

        # Recording back ON. SNNTrainer pauses it before its timed regions and leaves it
        # paused (its measure_activity() re-pauses in a finally), so a tester handed a
        # just-trained model inherits hooks that return immediately. Every buffer then
        # stays empty, firing_rate_tensor() returns None, and the SynOps estimate
        # comes out as a clean, plausible 0.0 for every batch of every run --
        # which is exactly what they did until this line existed.
        self.model.activity.resume()
        self.model.activity.clear()

        # Dense-MAC measurement for the SynOps estimate.
        probe_data, _ = next(iter(self.test_loader))
        probe_data = probe_data.to(self.device)
        self.dense_macs_per_layer = measure_dense_macs(self.model, probe_data)

        all_preds_gpu, all_targets_gpu = [], []
        raw_batch_records: list[dict] = []
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

        print("\n[TEST RUN]")

        self.pipeline_monitor.start()
        self.pipeline_monitor.measure_idle_baseline()
        self.pipeline_monitor.set_phase("test_run")
        self.pipeline_monitor.reset_epoch_memory()
        t_run_start = time.perf_counter()

        # self.test_loader is a DeviceLoader (event_data_workflow.data_pipeline) —
        # batches arrive already device-resident.
        with torch.no_grad():
            n_batches = len(self.test_loader)
            for batch_idx, (data, targets) in enumerate(self.test_loader):
                if batch_idx % 20 == 0:
                    print(f"  [test] batch {batch_idx}/{n_batches}  "
                          f"({time.perf_counter() - t_run_start:.1f}s elapsed)", flush=True)
                B = targets.size(0)
                T = data.size(0)

                with self.timed(self.fwd_events):
                    spk_rec = self.forward_pass(data)

                batch_spikes_gpu = spk_rec.sum()

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

                raw_batch_records.append({
                    "batch_idx":    batch_idx,
                    "B": B, "T": T,
                    "batch_spikes": batch_spikes_gpu,
                    "acc":          acc_gpu,
                    "synops":       synops_gpu,
                })

        t_run_elapsed = time.perf_counter() - t_run_start
        self.pipeline_monitor.stop()
        energy_report   = self.pipeline_monitor.phase_energy_report("test_run", t_run_elapsed)
        gpu             = energy_report["gpu"]
        # TOTAL includes idle draw and scales with runtime; DYNAMIC is above idle and
        # isolates the work. See PipelineMonitor.phase_energy_report for which answers
        # which question -- they are not interchangeable.
        gpu_energy_j         = energy_report["gpu_energy_j"]
        gpu_dynamic_energy_j = energy_report["gpu_dynamic_energy_j"]
        avg_power_w          = energy_report["avg_power_w"]
        dynamic_power_w      = energy_report["dynamic_power_w"]
        idle_power_w         = energy_report["idle_power_w"]

        # ---- Bulk transfer: the single sync point for everything
        # accumulated above, now that the whole test pass has finished. ----
        all_preds   = torch.cat(all_preds_gpu).cpu()   if all_preds_gpu   else torch.empty(0, dtype=torch.long)
        all_targets = torch.cat(all_targets_gpu).cpu() if all_targets_gpu else torch.empty(0, dtype=torch.long)
        np.add.at(cm, (all_targets.numpy(), all_preds.numpy()), 1)

        fwd_latencies_ms = [self.elapsed_ms(p) for p in self.fwd_events]

        total_spikes          = 0
        total_samples         = 0
        total_synops_pj       = 0.0
        total_possible_spikes = 0
        timesteps_seen: list[int] = []

        for rec, latency_ms in zip(raw_batch_records, fwd_latencies_ms):
            B, T            = rec["B"], rec["T"]
            batch_spikes    = int(rec["batch_spikes"].item())
            acc             = rec["acc"].item()
            batch_synops_pj = rec["synops"].item() * SYNOPS_ENERGY_PJ

            possible_spikes = T * B * self.num_classes
            spike_rate      = batch_spikes / possible_spikes if possible_spikes > 0 else 0.0

            total_spikes          += batch_spikes
            total_samples         += B
            total_synops_pj       += batch_synops_pj
            total_possible_spikes += possible_spikes
            timesteps_seen.append(T)

            print(f"  Batch {rec['batch_idx']:>3} | "
                  f"Acc: {acc * 100:.2f}% | "
                  f"Rate: {spike_rate:.3f} | "
                  f"Latency: {latency_ms:.1f}ms")

        overall_acc              = (all_preds == all_targets).float().mean().item() if total_samples > 0 else 0.0
        throughput_samples_per_s = total_samples / t_run_elapsed if t_run_elapsed > 0 else 0.0
        output_spike_rate        = (total_spikes / total_possible_spikes) if total_possible_spikes else 0.0
        # MEASURED from the data, per batch -- not framing.n_time_bins, so runs.csv records the T that ran.
        mean_timesteps           = sum(timesteps_seen) / len(timesteps_seen) if timesteps_seen else 0
        synops_energy_per_sample_pj = total_synops_pj / total_samples if total_samples > 0 else 0.0
        class_metrics            = self.class_metrics(cm)

        print("\n[TEST SUMMARY]")
        print(f"  • Framework               : {self.cfg.FRAMEWORK}")
        print(f"  • Overall Accuracy        : {overall_acc * 100:.2f}%")
        print(f"  • Total Samples           : {total_samples}")
        print(f"  • Output Spike Rate       : {output_spike_rate:.4f}")
        print(f"  • Throughput              : {throughput_samples_per_s:.1f} samples/s")
        print(f"  • SynOps Energy Estimate  : {total_synops_pj:.1f} pJ  (per-layer, dense-MAC-weighted)")
        print(f"  • SynOps Energy / Sample  : {synops_energy_per_sample_pj:.2f} pJ")
        if gpu_energy_j is None:
            print("  • GPU Energy              : not measured -- test pass shorter than one power sample")
        else:
            print(f"  • GPU Energy (total)      : {gpu_energy_j * 1e3:.2f} mJ  (idle draw INCLUDED)")
            print(f"  • Mean GPU Power          : {avg_power_w:.1f} W")
            if idle_power_w is not None:
                print(f"  • GPU Energy (dynamic)    : {gpu_dynamic_energy_j * 1e3:.2f} mJ  "
                      f"({dynamic_power_w:.1f} W above a {idle_power_w:.1f} W idle baseline)")
            else:
                print("  • GPU Energy (dynamic)    : not available -- no idle baseline measured")
        print(f"  • Peak GPU Memory         : {gpu['gpu_mem_peak_gb']} GB / {self.pipeline_monitor.total_memory_gb:.2f} GB  ({gpu['gpu_mem_peak_pct']}% peak)")
        print(f"  • GPU Utilization         : avg {gpu['gpu_util_avg_pct']}%  peak {gpu['gpu_util_peak_pct']}%")
        if gpu.get("gpu_idle_episodes", 0) > 0:
            print(f"  • GPU Idle                : {gpu['gpu_idle_episodes']} episode(s), {gpu['gpu_idle_total_s']:.1f}s total")

        self.write_csv(csv_path, class_metrics)

        return {
            "framework":                 self.cfg.FRAMEWORK,
            "overall_accuracy":          overall_acc,
            "output_spike_rate":         output_spike_rate,
            "dense_macs_per_layer":      dict(self.dense_macs_per_layer),
            "throughput_samples_per_s":  throughput_samples_per_s,
            "total_synops_energy_pj":    total_synops_pj,
            "synops_energy_per_sample_pj": synops_energy_per_sample_pj,
            # Named so a CSV reader cannot mistake one basis for the other.
            "gpu_energy_j_total":        gpu_energy_j,
            "gpu_energy_j_dynamic":      gpu_dynamic_energy_j,
            "avg_power_w":               avg_power_w,
            "dynamic_power_w":           dynamic_power_w,
            "idle_power_w":              idle_power_w,
            "gpu_mem_peak_gb":           gpu.get("gpu_mem_peak_gb") if gpu else None,
            "gpu_util_avg_pct":          gpu.get("gpu_util_avg_pct") if gpu else None,
            "timesteps":                 mean_timesteps or None,
            "class_metrics":             class_metrics,
            "confusion_matrix":          cm,
        }
