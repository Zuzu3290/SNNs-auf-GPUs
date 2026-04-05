"""
Module 7: Output — Spike Decoding & Classification
====================================================
Decodes output spike trains from Module 6 into predictions,
confidence scores, latency metrics, and visualisation.

Decoders:
  1. RateDecoder       — sum spikes over time, argmax
  2. LatencyDecoder    — time-to-first-spike (TTFS)
  3. PopulationDecoder — winner-take-all over population code

Outputs:
  - Predicted class label + confidence
  - Per-class spike rate plot (ASCII or matplotlib)
  - Event frame + spike overlay
  - Pipeline latency breakdown

Usage:
    decoder = RateDecoder(n_classes=10, class_names=CIFAR10_NAMES)
    result  = decoder.decode(spk_rec)
    print(result)
    result.plot_rates()
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Prediction result container
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Holds a single decoded prediction and metadata."""
    class_idx:    int
    class_name:   str
    confidence:   float                    # normalised spike rate for winner
    rates:        np.ndarray               # (n_classes,) rate per class
    latencies_ms: Optional[np.ndarray]     # (n_classes,) TTFS in ms, or None
    spike_counts: np.ndarray               # (n_classes,) raw spike counts
    wall_time_ms: float = 0.0             # end-to-end pipeline latency

    def __str__(self):
        bar  = self._ascii_bar()
        lines = [
            f"Prediction:  {self.class_name} (class {self.class_idx})",
            f"Confidence:  {self.confidence*100:.1f}%",
            f"Pipeline:    {self.wall_time_ms:.1f} ms",
            "",
            "Per-class spike rates:",
            bar,
        ]
        return "\n".join(lines)

    def _ascii_bar(self, width: int = 30) -> str:
        lines = []
        max_r = max(self.rates.max(), 1e-9)
        for i, (r, c) in enumerate(zip(self.rates, self._class_names)):
            bar_w = int(r / max_r * width)
            marker = "◀" if i == self.class_idx else " "
            lines.append(f"  {c:>12s}  {'█' * bar_w:<{width}s}  {r:.3f} {marker}")
        return "\n".join(lines)

    def plot_rates(self, save_path: Optional[str] = None):
        """Bar chart of per-class spike rates (requires matplotlib)."""
        if not HAS_MPL:
            print(self._ascii_bar())
            return

        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#1a8cff" if i == self.class_idx else "#b0bec5"
                  for i in range(len(self.rates))]
        ax.bar(self._class_names, self.rates, color=colors, edgecolor="none")
        ax.set_title(f"SNN Output — Prediction: {self.class_name} "
                     f"({self.confidence*100:.1f}%)", fontsize=11)
        ax.set_ylabel("Spike rate")
        ax.set_xlabel("Class")
        plt.xticks(rotation=35, ha="right", fontsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Output] Saved rate plot → {save_path}")
        else:
            plt.show()
        plt.close()


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

class RateDecoder:
    """
    Rate coding: accumulate spikes over all timesteps,
    normalise by T, predict class with highest rate.
    """

    def __init__(self, n_classes: int, class_names: Optional[List[str]] = None):
        self.n_classes   = n_classes
        self.class_names = class_names or [f"class_{i}" for i in range(n_classes)]

    def decode(self, spk_rec: "torch.Tensor",
               wall_time_ms: float = 0.0) -> PredictionResult:
        """
        Args:
            spk_rec: (T, batch, n_classes) — from Module 6
                     or (T, n_classes) for single sample

        Returns:
            PredictionResult for the first sample in batch
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required")

        if spk_rec.dim() == 3:
            spk = spk_rec[:, 0, :]  # first sample: (T, n_classes)
        else:
            spk = spk_rec           # (T, n_classes)

        T = spk.shape[0]
        counts = spk.sum(dim=0).cpu().numpy()   # (n_classes,)
        rates  = counts / max(T, 1)

        winner     = int(np.argmax(rates))
        confidence = rates[winner] / max(rates.sum(), 1e-9)

        r = PredictionResult(
            class_idx    = winner,
            class_name   = self.class_names[winner],
            confidence   = float(confidence),
            rates        = rates,
            latencies_ms = None,
            spike_counts = counts,
            wall_time_ms = wall_time_ms,
        )
        r._class_names = self.class_names
        return r

    def decode_batch(self, spk_rec: "torch.Tensor") -> List[PredictionResult]:
        """Decode all samples in a batch."""
        T, B, C = spk_rec.shape
        return [self.decode(spk_rec[:, b:b+1, :]) for b in range(B)]


class LatencyDecoder:
    """
    Time-to-first-spike (TTFS) decoder.
    The class whose neuron fires first wins.
    Shorter latency = higher confidence.
    """

    def __init__(self, n_classes: int, class_names: Optional[List[str]] = None,
                 bin_duration_ms: float = 1.0):
        self.n_classes       = n_classes
        self.class_names     = class_names or [f"class_{i}" for i in range(n_classes)]
        self.bin_duration_ms = bin_duration_ms

    def decode(self, spk_rec: "torch.Tensor") -> PredictionResult:
        if spk_rec.dim() == 3:
            spk = spk_rec[:, 0, :].cpu().numpy()
        else:
            spk = spk_rec.cpu().numpy()

        T, C = spk.shape

        # Find first spike time per class
        latencies = np.full(C, np.inf)
        for c in range(C):
            fired = np.where(spk[:, c] > 0.5)[0]
            if len(fired) > 0:
                latencies[c] = fired[0] * self.bin_duration_ms

        # Winner = earliest fire
        winner = int(np.argmin(latencies))
        lat_ms = latencies.copy()
        lat_ms[lat_ms == np.inf] = T * self.bin_duration_ms + 1

        # Confidence = inverse latency rank
        rank       = lat_ms.argsort().argsort()
        confidence = 1.0 - rank[winner] / max(C - 1, 1)

        rates = spk.sum(0) / max(T, 1)

        r = PredictionResult(
            class_idx    = winner,
            class_name   = self.class_names[winner],
            confidence   = float(confidence),
            rates        = rates,
            latencies_ms = lat_ms,
            spike_counts = spk.sum(0),
        )
        r._class_names = self.class_names
        return r


# ---------------------------------------------------------------------------
# Pipeline — wires all modules together end-to-end
# ---------------------------------------------------------------------------

class EventSNNPipeline:
    """
    Wraps Modules 1–7 into a single streaming pipeline.
    Each call to run_step() processes one time window end-to-end.

    Args:
        model:       SpikingClassifier from Module 6
        gm:          GPUMemoryManager from Module 4
        kernel:      SpikeKernel from Module 5
        decoder:     RateDecoder or LatencyDecoder
        sensor_size: (W, H)
        n_bins:      temporal bins
        window_us:   event accumulation window in µs
    """

    def __init__(self, model, gm, kernel, decoder,
                 sensor_size=(346,260), n_bins=5, window_us=10_000):
        self.model       = model
        self.gm          = gm
        self.kernel      = kernel
        self.decoder     = decoder
        self.sensor_size = sensor_size
        self.n_bins      = n_bins
        self.window_us   = window_us

        self._latencies  = []   # rolling buffer for latency stats

    def run_step(self, voxel_np: np.ndarray) -> Tuple["PredictionResult", dict]:
        """
        Run one full pipeline step.

        Args:
            voxel_np: (2, n_bins, H, W) numpy voxel grid from Module 3

        Returns:
            (PredictionResult, timing_dict)
        """
        t_total = time.perf_counter()
        timing  = {}

        # Step 4: CPU → GPU
        t0 = time.perf_counter()
        voxel_gpu = self.gm.push(voxel_np)
        timing["h2d_ms"] = (time.perf_counter() - t0) * 1000

        # Step 5: CUDA kernel → spikes
        t0 = time.perf_counter()
        spikes = self.kernel.lif_spikes(
            voxel_gpu.unsqueeze(0),  # add batch dim
            v_thresh = 0.5,
            leak     = 0.9,
        )
        timing["kernel_ms"] = (time.perf_counter() - t0) * 1000

        # Step 6: SNN forward pass
        t0 = time.perf_counter()
        # Reshape for model: (batch=1, pol=2, T, H, W)
        if HAS_TORCH:
            voxel_in = voxel_gpu.unsqueeze(0)   # (1, 2, T, H, W)
            spk_rec, _ = self.model(voxel_in)
        timing["snn_ms"] = (time.perf_counter() - t0) * 1000

        # Step 7: Decode
        t0 = time.perf_counter()
        wall_ms = (time.perf_counter() - t_total) * 1000
        result  = self.decoder.decode(spk_rec, wall_time_ms=wall_ms)
        timing["decode_ms"]  = (time.perf_counter() - t0) * 1000
        timing["total_ms"]   = wall_ms

        self._latencies.append(wall_ms)
        if len(self._latencies) > 100:
            self._latencies.pop(0)

        return result, timing

    def latency_stats(self) -> dict:
        if not self._latencies:
            return {}
        a = np.array(self._latencies)
        return {
            "mean_ms":   float(a.mean()),
            "p50_ms":    float(np.percentile(a, 50)),
            "p95_ms":    float(np.percentile(a, 95)),
            "p99_ms":    float(np.percentile(a, 99)),
            "max_ms":    float(a.max()),
            "fps":       1000.0 / max(a.mean(), 0.1),
        }


# ---------------------------------------------------------------------------
# Spike raster visualiser
# ---------------------------------------------------------------------------

def plot_spike_raster(spk_rec: "torch.Tensor",
                      n_neurons: int = 50,
                      save_path: Optional[str] = None):
    """
    Raster plot of the first `n_neurons` output neurons.

    Args:
        spk_rec:   (T, batch, n_classes) or (T, n_classes)
        n_neurons: how many neurons to display
        save_path: file path to save, or None to show
    """
    if not HAS_MPL:
        print("[Output] matplotlib not available — skipping raster plot")
        return

    if spk_rec.dim() == 3:
        spk = spk_rec[:, 0, :].cpu().numpy()
    else:
        spk = spk_rec.cpu().numpy()

    T, C = spk.shape
    n    = min(n_neurons, C)

    fig, ax = plt.subplots(figsize=(10, 4))
    for neuron in range(n):
        times = np.where(spk[:, neuron] > 0.5)[0]
        ax.scatter(times, np.full_like(times, neuron),
                   marker="|", s=50, c="#1a8cff", linewidths=0.8)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Output neuron")
    ax.set_title("SNN Output Spike Raster")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Output] Saved raster → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

N_CLASSES    = 10
CLASS_NAMES  = ["airplane","automobile","bird","cat","deer",
                "dog","frog","horse","ship","truck"]

if __name__ == "__main__":
    print("=== Module 7: Output Decoding Demo ===\n")

    if not HAS_TORCH:
        print("PyTorch required.")
        exit(1)

    # --- Simulate spike output from Module 6 ---
    T, B, C = 5, 4, N_CLASSES
    # Make class 3 ("cat") fire the most
    spk = torch.zeros(T, B, C)
    spk[:, :, 3] = 0.8   # "cat" fires in 80% of bins
    spk[:, :, 7] = 0.3   # "horse" fires occasionally
    spk += torch.rand(T, B, C) * 0.1   # noise

    # --- Rate decoder ---
    rate_dec = RateDecoder(n_classes=C, class_names=CLASS_NAMES)
    result   = rate_dec.decode(spk, wall_time_ms=4.7)
    print(result)

    # --- Latency decoder ---
    lat_dec  = LatencyDecoder(n_classes=C, class_names=CLASS_NAMES, bin_duration_ms=2.0)
    lat_res  = lat_dec.decode(spk[:, 0, :])
    print(f"\nLatency decoder → {lat_res.class_name}  "
          f"(TTFS: {lat_res.latencies_ms[lat_res.class_idx]:.1f} ms)")

    # --- Batch decode ---
    batch_res = rate_dec.decode_batch(spk)
    print(f"\nBatch predictions: {[r.class_name for r in batch_res]}")

    # --- Raster + rate plot (saves to /tmp if matplotlib available) ---
    if HAS_MPL:
        plot_spike_raster(spk[:, 0, :], save_path="/tmp/raster.png")
        result.plot_rates(save_path="/tmp/rates.png")
        print("\nPlots saved to /tmp/raster.png and /tmp/rates.png")
    else:
        print("\n(matplotlib not installed — install with: pip install matplotlib)")

    # --- Simulate pipeline latency stats ---
    pipe_lats = np.random.normal(loc=4.2, scale=0.8, size=50)
    pipe_lats = np.clip(pipe_lats, 1.0, 20.0)
    print("\nSimulated pipeline latency stats:")
    print(f"  Mean:  {pipe_lats.mean():.2f} ms")
    print(f"  P95:   {np.percentile(pipe_lats,95):.2f} ms")
    print(f"  FPS:   {1000/pipe_lats.mean():.1f}")
