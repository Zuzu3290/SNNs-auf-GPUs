"""

Defines SNN architectures using snnTorch, with a Norse fallback.

Architectures:
  1. SpikingEncoder      — 2-layer conv SNN feature extractor
  2. SpikingClassifier   — full encoder + pooling + FC classifier
  3. SpikingRecurrent    — recurrent SNN with SLSTM cells (Norse)

Neuron models:
  - Leaky (standard LIF)
  - Synaptic (2nd-order dynamics)
  - Alpha  (alpha-function synapse)

Training:
  - BPTT (backprop through time) with surrogate gradients
  - Rate coding loss or van Rossum distance

Usage:
    model = SpikingClassifier(
        sensor_size=(346, 260), n_bins=5, n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # One training step:
    loss, acc = train_step(model, voxel_batch, labels, optimizer)

    # Inference:
    spikes, pred = model.infer(voxel_gpu)
"""

import numpy as np
import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import norse.torch as norse



def spike_rate(spikes: "torch.Tensor") -> float:
    """Mean firing rate across all neurons and timesteps."""
    return spikes.float().mean().item()


# ---------------------------------------------------------------------------
# Module 6a: SpikingEncoder (snnTorch)
# ---------------------------------------------------------------------------

class SpikingEncoder(nn.Module):
    """
    2-layer convolutional SNN encoder.

    Input:  (batch, 2*n_bins, H, W) — flattened voxel grid
    Output: spike trains at each layer, final membrane potentials

    Architecture:
        Conv2d(2*n_bins → 32, 3×3) → BatchNorm → LIF
        Conv2d(32 → 64, 3×3)        → BatchNorm → LIF
        MaxPool2d(2×2)
    """

    def __init__(self,
                 in_channels: int = 10,    # 2 polarities × 5 bins
                 hidden_ch1:  int = 32,
                 hidden_ch2:  int = 64,
                 beta:        float = 0.95,
                 threshold:   float = 1.0,
                 spike_grad         = None):
        super().__init__()

        sg = spike_grad or surrogate.fast_sigmoid(slope=25)

        # Convolutional feature layers
        self.conv1 = nn.Conv2d(in_channels, hidden_ch1, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_ch1)
        self.lif1  = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=sg, init_hidden=True)

        self.conv2 = nn.Conv2d(hidden_ch1, hidden_ch2, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(hidden_ch2)
        self.lif2  = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=sg, init_hidden=True)

        self.pool  = nn.MaxPool2d(2, 2)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor",
                                                    "torch.Tensor", "torch.Tensor"]:
        """
        Args:
            x: (batch, 2*n_bins, H, W)

        Returns:
            spk1: (batch, 32, H, W)   — layer 1 spikes
            mem1: (batch, 32, H, W)   — layer 1 membrane
            spk2: (batch, 64, H//2, W//2) — layer 2 spikes
            mem2: (batch, 64, H//2, W//2) — layer 2 membrane
        """
        # Layer 1
        cur1 = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(cur1)

        # Layer 2 + pool
        cur2 = self.bn2(self.conv2(spk1))
        spk2, mem2 = self.lif2(cur2)
        spk2 = self.pool(spk2)
        mem2 = self.pool(mem2)

        return spk1, mem1, spk2, mem2


# ---------------------------------------------------------------------------
# Module 6b: SpikingClassifier — full end-to-end model
# ---------------------------------------------------------------------------

class SpikingClassifier(nn.Module):
    """
    Full SNN classifier for event-camera input.

    Architecture (temporal unrolling over n_bins timesteps):
        [voxel grid input]
        → Conv-LIF encoder (timestep-by-timestep)
        → Global average pool
        → Leaky readout layer (accumulate spikes over time)
        → Rate-code softmax at final timestep

    Args:
        sensor_size:  (W, H)
        n_bins:       temporal bins from voxel grid
        n_classes:    number of output classes
        hidden_ch:    feature channels after encoder
        beta:         membrane decay factor
        threshold:    spike threshold
    """

    def __init__(self,
                 sensor_size: tuple = (346, 260),
                 n_bins:      int   = 5,
                 n_classes:   int   = 10,
                 hidden_ch:   int   = 64,
                 beta:        float = 0.9,
                 threshold:   float = 1.0):
        if not HAS_SNNTORCH:
            raise RuntimeError("snnTorch required")
        super().__init__()

        W, H             = sensor_size
        self.n_bins      = n_bins
        self.n_classes   = n_classes
        sg               = surrogate.fast_sigmoid(slope=25)

        # Encoder
        self.conv1 = nn.Conv2d(2, 32, 5, padding=2, bias=False)
        self.lif1  = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=sg, init_hidden=True)

        self.conv2 = nn.Conv2d(32, hidden_ch, 3, padding=1, bias=False)
        self.lif2  = snn.Leaky(beta=beta, threshold=threshold,
                                spike_grad=sg, init_hidden=True)

        self.pool  = nn.AdaptiveAvgPool2d((8, 8))  # fixed spatial size

        # Readout
        fc_in = hidden_ch * 8 * 8
        self.fc    = nn.Linear(fc_in, n_classes, bias=True)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold,
                                  spike_grad=sg, init_hidden=True)

    def reset(self):
        """Reset all hidden states between sequences."""
        utils.reset(self)

    def forward(self, voxel: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Temporal unrolling: process each time bin sequentially.

        Args:
            voxel: (batch, 2, n_bins, H, W) — from Module 4/5

        Returns:
            spk_rec:  (n_bins, batch, n_classes)  — output spikes
            mem_rec:  (n_bins, batch, n_classes)  — output membrane
        """
        batch, pol, T, H, W = voxel.shape
        self.reset()

        spk_rec = []
        mem_rec = []

        for t in range(T):
            # Input at this timestep: (batch, 2, H, W)
            x = voxel[:, :, t, :, :]

            # Encoder
            cur1 = self.conv1(x)
            spk1, _  = self.lif1(cur1)

            cur2 = self.conv2(spk1)
            spk2, _  = self.lif2(cur2)

            # Pool + flatten
            feat = self.pool(spk2)
            flat = feat.flatten(1)

            # Readout
            cur_out = self.fc(flat)
            spk_out, mem_out = self.lif_out(cur_out)

            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)

    def infer(self, voxel: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Single-pass inference.

        Returns:
            spikes: (n_bins, batch, n_classes)
            pred:   (batch,) class indices from rate-code vote
        """
        self.eval()
        with torch.no_grad():
            spk_rec, _ = self(voxel)
        # Rate code: sum spikes over time → argmax
        rate  = spk_rec.sum(dim=0)   # (batch, n_classes)
        pred  = rate.argmax(dim=1)
        return spk_rec, pred


# ---------------------------------------------------------------------------
# Module 6c: SpikingRecurrent (Norse) — optional
# ---------------------------------------------------------------------------

class SpikingRecurrent(nn.Module):
    """
    Recurrent SNN using Norse's LSNN (Adaptive LIF) cells.
    Better for sequences / temporal patterns.
    """

    def __init__(self,
                 sensor_size: tuple = (346, 260),
                 n_bins:      int   = 5,
                 n_classes:   int   = 10,
                 hidden_size: int   = 256):
        super().__init__()

        W, H = sensor_size
        self.n_bins = n_bins
        flat_in     = 2 * H * W   # flattened voxel per timestep

        self.encoder = nn.Linear(flat_in, hidden_size)
        self.lsnn    = norse.LIFRecurrentCell(
            input_size  = hidden_size,
            hidden_size = hidden_size,
        )
        self.decoder = nn.Linear(hidden_size, n_classes)

    def forward(self, voxel: "torch.Tensor"):
        """
        Args:
            voxel: (batch, 2, T, H, W)

        Returns:
            output spikes: (T, batch, n_classes)
        """
        batch, pol, T, H, W = voxel.shape
        state = None
        outputs = []

        for t in range(T):
            x   = voxel[:, :, t].reshape(batch, -1)  # (batch, 2*H*W)
            enc = F.relu(self.encoder(x))
            z, state = self.lsnn(enc, state)
            out = self.decoder(z)
            outputs.append(out)

        return torch.stack(outputs)   # (T, batch, n_classes)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def build_loss(loss_type: str = "rate"):
    """
    Build loss function.
    'rate' → cross-entropy on summed spike counts (rate code)
    'mse'  → mean-square error on rate vs one-hot
    """
    if loss_type == "rate":
        return SF.ce_rate_loss()
    elif loss_type == "mse":
        return SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
    else:
        return nn.CrossEntropyLoss()


def train_step(model:     "SpikingClassifier",
               voxel:     "torch.Tensor",
               labels:    "torch.Tensor",
               optimizer: "torch.optim.Optimizer",
               loss_fn    = None) -> Tuple[float, float]:
    """
    One training step.

    Args:
        model:     SpikingClassifier
        voxel:     (batch, 2, T, H, W) on GPU
        labels:    (batch,) int64 class indices on GPU
        optimizer: torch optimizer
        loss_fn:   snnTorch loss or None (auto-select)

    Returns:
        (loss_value, accuracy)
    """
    if loss_fn is None:
        loss_fn = build_loss("rate")

    model.train()
    optimizer.zero_grad()

    spk_rec, mem_rec = model(voxel)   # (T, batch, n_classes)

    # snnTorch rate loss expects (T, batch, n_classes) + (batch,)
    if HAS_SNNTORCH and hasattr(loss_fn, '__self__') or callable(loss_fn):
        try:
            loss = loss_fn(spk_rec, labels)
        except Exception:
            # Fallback: cross-entropy on rate
            rate = spk_rec.sum(0)
            loss = F.cross_entropy(rate, labels)
    else:
        rate = spk_rec.sum(0)
        loss = F.cross_entropy(rate, labels)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    rate = spk_rec.sum(0)
    acc  = (rate.argmax(1) == labels).float().mean().item()
    return loss.item(), acc


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Module 6: SNN Simulation Demo ===\n")

    if not HAS_TORCH:
        print("PyTorch required.")
        exit(1)

    print(f"snnTorch: {HAS_SNNTORCH}")
    print(f"Norse:    {HAS_NORSE}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    if HAS_SNNTORCH:
        # --- SpikingClassifier demo ---
        model = SpikingClassifier(
            sensor_size = (346, 260),
            n_bins      = 5,
            n_classes   = 10,
            hidden_ch   = 64,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"SpikingClassifier parameters: {n_params:,}\n")

        # Dummy batch: (B=4, pol=2, T=5, H=260, W=346)
        B = 4
        voxel  = torch.rand(B, 2, 5, 260, 346, device=device)
        labels = torch.randint(0, 10, (B,), device=device)

        # Forward pass
        t0 = time.perf_counter()
        spk_rec, mem_rec = model(voxel)
        dt = (time.perf_counter() - t0) * 1000

        print(f"Forward pass:")
        print(f"  Input shape:   {tuple(voxel.shape)}")
        print(f"  Spike output:  {tuple(spk_rec.shape)}")
        print(f"  Latency:       {dt:.1f} ms")
        print(f"  Output spike rate: {spike_rate(spk_rec)*100:.1f}%\n")

        # Inference
        _, pred = model.infer(voxel)
        print(f"Predictions:  {pred.tolist()}")
        print(f"Labels:       {labels.tolist()}\n")

        # One training step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss, acc = train_step(model, voxel, labels, optimizer)
        print(f"Training step: loss={loss:.4f}  acc={acc*100:.1f}%")

    else:
        print("snnTorch not installed — install with: pip install snntorch")
        print("Using a demo fallback (pure PyTorch LIF layer):\n")

        class SimpleLIF(nn.Module):
            def __init__(self, beta=0.9, threshold=1.0):
                super().__init__()
                self.beta = beta
                self.threshold = threshold

            def forward(self, x):
                T = x.shape[2]
                mem = torch.zeros_like(x[:, :, 0])
                spikes = []
                for t in range(T):
                    mem = self.beta * mem + x[:, :, t]
                    s   = (mem >= self.threshold).float()
                    mem = mem * (1 - s)
                    spikes.append(s)
                return torch.stack(spikes, dim=2)

        model = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1),
            nn.ReLU(),
        ).to(device)
        print("Minimal demo model created (full SNN requires snnTorch).")
