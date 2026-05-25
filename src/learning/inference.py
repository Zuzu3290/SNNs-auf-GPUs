from __future__ import annotations
 
import os
import csv
import time
import torch
import numpy as np
from skeleton import Settings
from learning.training import aggregate_spike_output

# Energy per synaptic op — adjust for your target neuromorphic platform
ENERGY_PER_SPIKE_PJ = 3.5
 
class SNNTester:
 
    def __init__(self, model, test_loader, cfg: Settings, device: torch.device):
        self.model       = model
        self.test_loader = test_loader
        self.cfg         = cfg
        self.device      = device
        self.num_classes = cfg.NUM_CLASSES
        self.batch_log   = []

    def _class_metrics(self, cm: np.ndarray) -> list[dict]:
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
            class_acc   = (tp + tn) / total
 
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

    def _write_csv(self, path: str):
        if not self.batch_log:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = list(self.batch_log[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.batch_log)
        print(f"[INFO] Test log saved → {path}")

    def run(self, csv_path: str = "./outputs/data/test.csv") -> dict:
        self.model.eval()
 
        all_preds, all_targets = [], []
        total_spikes     = 0
        total_latency_ms = 0.0
        total_samples    = 0
        total_energy_pj  = 0.0
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
 
        print("\n[TEST RUN]")
 
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.test_loader):
                data    = data.to(self.device)
                targets = targets.to(self.device)
                B = targets.size(0)
                T = data.size(0)
 
                t0         = time.perf_counter()
                spk_rec    = self.model(data)               # [T, B, num_classes]
                latency_ms = (time.perf_counter() - t0) * 1000
 
                batch_spikes    = int(spk_rec.sum().item())
                possible_spikes = T * B * self.num_classes
                spike_rate      = batch_spikes / possible_spikes
                energy_pj       = batch_spikes * ENERGY_PER_SPIKE_PJ
 
                logits = aggregate_spike_output(spk_rec.float())
                preds  = logits.argmax(dim=1).cpu()
                tgts   = targets.cpu()
                acc    = (preds == tgts).float().mean().item()
 
                np.add.at(cm, (tgts.numpy(), preds.numpy()), 1)
                all_preds.append(preds)
                all_targets.append(tgts)
 
                total_spikes     += batch_spikes
                total_latency_ms += latency_ms
                total_samples    += B
                total_energy_pj  += energy_pj
 
                self.batch_log.append({
                    "batch":                 batch_idx,
                    "samples":               B,
                    "timesteps":             T,
                    "accuracy":              round(acc, 4),
                    "spikes_activated":      batch_spikes,
                    "possible_spikes":       possible_spikes,
                    "spike_rate":            round(spike_rate, 4),
                    "latency_ms":            round(latency_ms, 3),
                    "latency_per_sample_ms": round(latency_ms / B, 3),
                    "energy_pJ":             round(energy_pj, 2),
                })
 
                print(f"  Batch {batch_idx:>3} | "
                      f"Acc: {acc * 100:.2f}% | "
                      f"Spikes: {batch_spikes:>6} | "
                      f"Rate: {spike_rate:.3f} | "
                      f"Latency: {latency_ms:.1f}ms | "
                      f"Energy: {energy_pj:.1f}pJ")
 
        all_preds   = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
 
        overall_acc            = (all_preds == all_targets).float().mean().item()
        avg_latency_ms         = total_latency_ms / len(self.batch_log)
        avg_latency_per_sample = total_latency_ms / total_samples
        avg_spikes_per_sample  = total_spikes / total_samples
        class_metrics          = self._class_metrics(cm)
        gt_dist                = {c: int((all_targets == c).sum()) for c in range(self.num_classes)}
        pred_dist              = {c: int((all_preds   == c).sum()) for c in range(self.num_classes)}
 
        print("\n[TEST SUMMARY]")
        print(f"  • Overall Accuracy        : {overall_acc * 100:.2f}%")
        print(f"  • Total Samples           : {total_samples}")
        print(f"  • Total Spikes Activated  : {total_spikes:,}")
        print(f"  • Avg Spikes / Sample     : {avg_spikes_per_sample:.2f}")
        print(f"  • Avg Batch Latency       : {avg_latency_ms:.2f} ms")
        print(f"  • Avg Latency / Sample    : {avg_latency_per_sample:.3f} ms")
        print(f"  • Total Energy Estimate   : {total_energy_pj:.1f} pJ")
        print(f"  • Energy / Sample         : {total_energy_pj / total_samples:.2f} pJ")
 
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
 
        self._write_csv(csv_path)
 
        return {
            "overall_accuracy":          overall_acc,
            "total_spikes":              total_spikes,
            "avg_spikes_per_sample":     avg_spikes_per_sample,
            "avg_latency_ms":            avg_latency_ms,
            "avg_latency_per_sample_ms": avg_latency_per_sample,
            "total_energy_pj":           total_energy_pj,
            "energy_per_sample_pj":      total_energy_pj / total_samples,
            "class_metrics":             class_metrics,
            "gt_distribution":           gt_dist,
            "pred_distribution":         pred_dist,
            "confusion_matrix":          cm,
        }