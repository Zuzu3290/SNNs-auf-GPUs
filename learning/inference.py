from __future__ import annotations
import logging
import os
import time
import torch
import matplotlib.pyplot as plt
from skeleton import Settings
from learning.utilities import SYNOPS_ENERGY_PJ, elapsed_ms, measure_dense_macs, synops_estimate, timed
from event_data_workflow.system_monitor import PipelineMonitor

logger = logging.getLogger(__name__)


class SNNTester:

    def __init__(self, model, test_loader, cfg: Settings, device: torch.device):
        self.model       = model
        self.test_loader = test_loader
        self.cfg         = cfg
        self.device      = device
        self.num_classes = cfg.NUM_CLASSES
        self.pipeline_monitor = PipelineMonitor()
        self.fwd_events = []
        self.dense_macs_per_layer = {}

    def forward_pass(self, data: torch.Tensor) -> torch.Tensor:
        return self.model(data)

    def run(self) -> dict:
        """Test pass with deferred sync: per-batch values stay on the device and are read back once after the loop."""
        self.model.eval()

        probe_data, _ = next(iter(self.test_loader))
        self.dense_macs_per_layer = measure_dense_macs(self.model, probe_data)

        all_preds_dev, all_targets_dev = [], []
        raw_batch_records: list[dict] = []

        logger.info(f"[TEST RUN]  on {self.device.type}")

        self.pipeline_monitor.start()
        self.pipeline_monitor.measure_idle_baseline()
        self.pipeline_monitor.set_phase("test_run")
        self.pipeline_monitor.reset_epoch_memory()
        t_run_start = time.perf_counter()

        with torch.no_grad():
            n_batches = len(self.test_loader)
            for batch_idx, (data, targets) in enumerate(self.test_loader):
                if batch_idx % 20 == 0:
                    logger.info(f"[test] batch {batch_idx}/{n_batches}  "
                                f"({time.perf_counter() - t_run_start:.1f}s elapsed)")
                T = data.size(0)

                self.model.set_spike_counting(True)  # fresh per-layer counters for this batch's SynOps
                with timed(self.device, self.fwd_events):
                    spk_rec = self.forward_pass(data)

                preds = spk_rec.float().sum(dim=0).argmax(dim=1)  # the spike count per class is its score
                all_preds_dev.append(preds)
                all_targets_dev.append(targets)
                self.last_spk_rec, self.last_targets = spk_rec, targets  # for plot_raster; references only, no sync
                raw_batch_records.append({
                    "B": targets.size(0), "T": T,
                    "batch_spikes": spk_rec.sum(),
                    "synops": synops_estimate(self.model, self.dense_macs_per_layer, T, self.device),
                })
        self.model.set_spike_counting(False)

        t_run_elapsed = time.perf_counter() - t_run_start
        self.pipeline_monitor.stop()
        energy_report        = self.pipeline_monitor.phase_energy_report("test_run", t_run_elapsed)
        gpu                  = energy_report["gpu"]
        gpu_energy_j         = energy_report["gpu_energy_j"]          # idle draw included, scales with runtime
        gpu_dynamic_energy_j = energy_report["gpu_dynamic_energy_j"]  # above idle, isolates the work
        avg_power_w          = energy_report["avg_power_w"]
        dynamic_power_w      = energy_report["dynamic_power_w"]
        idle_power_w         = energy_report["idle_power_w"]

        # The single sync point for everything accumulated above.
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        all_preds   = torch.cat(all_preds_dev).cpu()   if all_preds_dev   else torch.empty(0, dtype=torch.long)
        all_targets = torch.cat(all_targets_dev).cpu() if all_targets_dev else torch.empty(0, dtype=torch.long)

        total_spikes          = 0
        total_samples         = 0
        total_synops_pj       = 0.0
        total_possible_spikes = 0
        timesteps_seen: list[int] = []

        for rec in raw_batch_records:
            B, T = rec["B"], rec["T"]
            total_spikes          += int(rec["batch_spikes"].item())
            total_samples         += B
            total_synops_pj       += rec["synops"].item() * SYNOPS_ENERGY_PJ
            total_possible_spikes += T * B * self.num_classes
            timesteps_seen.append(T)

        overall_acc              = (all_preds == all_targets).float().mean().item() if total_samples > 0 else 0.0
        throughput_samples_per_s = total_samples / t_run_elapsed if t_run_elapsed > 0 else 0.0
        output_spike_rate        = (total_spikes / total_possible_spikes) if total_possible_spikes else 0.0
        mean_timesteps           = sum(timesteps_seen) / len(timesteps_seen) if timesteps_seen else 0  # the T that ran
        synops_energy_per_sample_pj = total_synops_pj / total_samples if total_samples > 0 else 0.0
        fwd_latencies_ms         = [elapsed_ms(p) for p in self.fwd_events]
        mean_batch_latency_ms    = sum(fwd_latencies_ms) / len(fwd_latencies_ms) if fwd_latencies_ms else 0.0

        lines = []
        lines.append("[TEST SUMMARY]")
        lines.append(f"  • Framework               : {self.cfg.FRAMEWORK}")
        lines.append(f"  • Device                  : {self.device.type}")
        lines.append(f"  • Overall Accuracy        : {overall_acc * 100:.2f}%")
        lines.append(f"  • Total Samples           : {total_samples}")
        lines.append(f"  • Output Spike Rate       : {output_spike_rate:.4f}")
        lines.append(f"  • Throughput              : {throughput_samples_per_s:.1f} samples/s")
        lines.append(f"  • Mean Batch Forward      : {mean_batch_latency_ms:.1f} ms")
        lines.append(f"  • SynOps Energy Estimate  : {total_synops_pj:.1f} pJ  (per-layer, dense-MAC-weighted)")
        lines.append(f"  • SynOps Energy / Sample  : {synops_energy_per_sample_pj:.2f} pJ")
        if gpu_energy_j is None:
            lines.append("  • GPU Energy              : not measured -- test pass shorter than one power sample")
        else:
            lines.append(f"  • GPU Energy (total)      : {gpu_energy_j * 1e3:.2f} mJ  (idle draw INCLUDED)")
            lines.append(f"  • Mean GPU Power          : {avg_power_w:.1f} W")
            if idle_power_w is not None:
                lines.append(f"  • GPU Energy (dynamic)    : {gpu_dynamic_energy_j * 1e3:.2f} mJ  "
                             f"({dynamic_power_w:.1f} W above a {idle_power_w:.1f} W idle baseline)")
            else:
                lines.append("  • GPU Energy (dynamic)    : not available -- no idle baseline measured")
        lines.append(f"  • Peak GPU Memory         : {gpu['gpu_mem_peak_gb']} GB / {self.pipeline_monitor.total_memory_gb:.2f} GB  ({gpu['gpu_mem_peak_pct']}% peak)")
        lines.append(f"  • GPU Utilization         : avg {gpu['gpu_util_avg_pct']}%  peak {gpu['gpu_util_peak_pct']}%")
        if gpu.get("gpu_idle_episodes", 0) > 0:
            lines.append(f"  • GPU Idle                : {gpu['gpu_idle_episodes']} episode(s), {gpu['gpu_idle_total_s']:.1f}s total")
        logger.info("\n".join(lines))

        return {
            "framework":                 self.cfg.FRAMEWORK,
            "inference_device":          self.device.type,
            "overall_accuracy":          overall_acc,
            "output_spike_rate":         output_spike_rate,
            "dense_macs_per_layer":      dict(self.dense_macs_per_layer),
            "throughput_samples_per_s":  throughput_samples_per_s,
            "total_synops_energy_pj":    total_synops_pj,
            "synops_energy_per_sample_pj": synops_energy_per_sample_pj,
            "gpu_energy_j_total":        gpu_energy_j,
            "gpu_energy_j_dynamic":      gpu_dynamic_energy_j,
            "avg_power_w":               avg_power_w,
            "dynamic_power_w":           dynamic_power_w,
            "idle_power_w":              idle_power_w,
            "gpu_mem_peak_gb":           gpu.get("gpu_mem_peak_gb") if gpu else None,
            "gpu_util_avg_pct":          gpu.get("gpu_util_avg_pct") if gpu else None,
            "timesteps":                 mean_timesteps or None,
        }

    def plot_raster(self, save_dir: str = "./outputs/plots") -> None:
        """Output spikes of one test sample over time: one row per class neuron, the true class shaded."""
        os.makedirs(save_dir, exist_ok=True)
        spk = self.last_spk_rec[:, 0, :].float().cpu()  # [T, B, classes] -> first sample of the last test batch
        timesteps, classes = spk.shape
        true_class = int(self.last_targets[0])
        predicted = int(spk.sum(dim=0).argmax())
        steps, neurons = spk.numpy().nonzero()

        fig, ax = plt.subplots(figsize=(10, min(12, max(3, classes * 0.3))))
        ax.axhspan(true_class - 0.5, true_class + 0.5, color="tab:green", alpha=0.2, label=f"true class ({true_class})")
        ax.scatter(steps, neurons, s=80, c="black", marker="|")
        ax.set_xlim(-0.5, timesteps - 0.5)
        ax.set_ylim(-0.5, classes - 0.5)
        if classes <= 20:
            ax.set_yticks(range(classes))
        ax.set_title(f"Output spikes of one test sample: true class {true_class}, predicted {predicted} "
                     f"({'correct' if predicted == true_class else 'wrong'})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Output neuron (class)")
        ax.legend(loc="upper right")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, "spike_raster.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"[PLOT] Saved -> {path}")
