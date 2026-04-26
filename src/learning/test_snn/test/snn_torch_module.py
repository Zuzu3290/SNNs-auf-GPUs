"""

Defines:
  - CSVMNISTDataStream : loads the real MNIST CSV file, normalises and spike-encodes it
  - MNISTDataStream    : loads MNIST via torchvision download (fallback)
  - SNNLayer           : dynamic multi-layer SNN built from the config
  - forward_pass()     : time-unrolled forward loop
  - train_supervised()     : BPTT with surrogate gradients
  - train_unsupervised()   : STDP-trace weight update (no labels)
  - SNNMetrics             : performance metric collection
"""

import os
import time
import statistics
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
import snntorch.functional as SF
import snntorch.spikegen as spikegen
import matplotlib
matplotlib.use("Agg")               # non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np

from Utils_snntorch import (
    LearningMode,
    get_neuron,
    get_loss_function,
    get_surrogate,
)


# ──────────────────────────────────────────────────────────────────────────────
# CSV MNIST DATA STREAM  (loads your uploaded mnist_train_small.csv)
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class CSVMNISTDataStream:
    """
    Loads MNIST directly from the uploaded CSV file.

    CSV format (no header):
      column 0       : integer label  (0–9)
      columns 1–784  : pixel values   (0–255), one per pixel of a 28×28 image

    Processing pipeline (mirrors snntorch Tutorial 1):
      1. Read CSV with pandas
      2. Separate label column from pixel columns
      3. Normalise pixels from [0, 255] → [0.0, 1.0]
      4. Reshape to [N, 1, 28, 28]  (grayscale channel-first for snntorch)
      5. Spike-encode with spikegen.rate()  →  [num_steps, batch, 1, 28, 28]

    Usage:
        train_stream, test_stream = CSVMNISTDataStream.train_test(
            csv_path="mnist_train_small.csv",
            batch_size=128,
            num_steps=25,
        )
        for spk_input, labels in train_stream.batches():
            # spk_input : [num_steps, batch, 784]   ← flattened for the FC network
            # labels    : [batch]
    """

    def __init__(self, csv_path: str, batch_size: int = 128,
                 num_steps: int = 25, encoding: str = "rate",
                 train: bool = True, train_ratio: float = 0.8,
                 seed: int = 42):

        self.num_steps  = num_steps
        self.batch_size = batch_size
        self.encoding   = encoding

        # ── 1. Read CSV ───────────────────────────────────────────────
        print(f"Loading MNIST CSV from: {csv_path}")
        df = pd.read_csv(csv_path, header=None)
        print(f"  Total samples: {len(df)}  |  Columns: {df.shape[1]}")

        # ── 2. Split label / pixels ───────────────────────────────────
        labels_np = df.iloc[:, 0].values.astype("int64")       # (N,)
        pixels_np = df.iloc[:, 1:].values.astype("float32")    # (N, 784)

        # ── 3. Normalise [0,255] → [0,1] ─────────────────────────────
        pixels_np /= 255.0

        # ── 4. Convert to tensors ─────────────────────────────────────
        pixels_t = torch.tensor(pixels_np)                      # (N, 784)
        labels_t = torch.tensor(labels_np)                      # (N,)

        # ── 5. Train / test split ─────────────────────────────────────
        torch.manual_seed(seed)
        n_total = len(labels_t)
        perm    = torch.randperm(n_total)
        n_train = int(n_total * train_ratio)

        if train:
            idx = perm[:n_train]
        else:
            idx = perm[n_train:]

        pixels_t = pixels_t[idx]
        labels_t = labels_t[idx]
        print(f"  {'Train' if train else 'Test'} split: {len(labels_t)} samples")

        # ── 6. Build DataLoader ───────────────────────────────────────
        dataset = TensorDataset(pixels_t, labels_t)
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=True,
        )

    # ------------------------------------------------------------------
    def _encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        pixels : [batch, 784]  values in [0, 1]
        returns [num_steps, batch, 784]

        Follows Tutorial 1 spike generation:
          rate    → each pixel is a Bernoulli firing probability per timestep
          latency → high-intensity pixels fire early, low-intensity fire late
        """
        # Reshape to image format for spikegen, then flatten back
        imgs = pixels.view(pixels.shape[0], 1, 28, 28)   # [B, 1, 28, 28]

        if self.encoding == "rate":
            spikes = spikegen.rate(imgs, num_steps=self.num_steps)
        elif self.encoding == "latency":
            spikes = spikegen.latency(
                imgs, num_steps=self.num_steps,
                tau=5, threshold=0.01,
                normalize=True, linear=True, clip=True,
            )
        else:
            raise ValueError(f"Unknown encoding '{self.encoding}'.")

        # Flatten spatial dims: [T, B, 1, 28, 28] → [T, B, 784]
        T, B = spikes.shape[0], spikes.shape[1]
        return spikes.view(T, B, -1).float()

    # ------------------------------------------------------------------
    def batches(self):
        """
        Yield (spike_input, labels) for every minibatch.
          spike_input : [num_steps, batch, 784]
          labels      : [batch]  integer class 0–9
        """
        for pixels, labels in self.loader:
            spikes = self._encode(pixels)
            yield spikes, labels

    # ------------------------------------------------------------------
    @classmethod
    def train_test(cls, csv_path: str, batch_size: int = 128,
                   num_steps: int = 25, encoding: str = "rate",
                   train_ratio: float = 0.8):
        """Return (train_stream, test_stream) from a single CSV file."""
        train = cls(csv_path, batch_size, num_steps, encoding,
                    train=True,  train_ratio=train_ratio)
        test  = cls(csv_path, batch_size, num_steps, encoding,
                    train=False, train_ratio=train_ratio)
        return train, test

class MNISTDataStream:
    """
    Loads the real MNIST dataset and spike-encodes it on the fly,
    exactly as shown in snntorch Tutorial 1.

    Encoding options (set via `encoding` arg):
      "rate"    – spikegen.rate()    : pixel brightness → spike probability
                  White pixel (1.0) → fires every step
                  Black pixel (0.0) → never fires
                  Output shape: [num_steps, batch, 1, 28, 28]

      "latency" – spikegen.latency(): bright pixels → early spikes
                  High intensity fires first; low intensity fires last
                  Output shape: [num_steps, batch, 1, 28, 28]

    The images are kept at 28×28 and flattened to 784 inside the
    forward pass by SNNLayer (which expects flat input_size=784).
    """

    # MNIST normalisation transform — Tutorial 5 pattern
    _transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))   # keeps pixels in [0,1]
    ])

    def __init__(self, data_path: str = "./data",
                 batch_size: int = 128,
                 num_steps: int = 25,
                 encoding: str = "rate",
                 train: bool = True):
        self.num_steps  = num_steps
        self.encoding   = encoding
        self.batch_size = batch_size

        dataset = datasets.MNIST(
            data_path,
            train=train,
            download=True,
            transform=self._transform,
        )
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=True,
        )

    # ------------------------------------------------------------------
    def _encode(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs : [batch, 1, 28, 28]  pixel values in [0, 1]
        returns [num_steps, batch, 1, 28, 28]
        """
        if self.encoding == "rate":
            # Tutorial 1 §2.1 — each pixel is a Bernoulli firing probability
            return spikegen.rate(imgs, num_steps=self.num_steps)

        elif self.encoding == "latency":
            # Tutorial 1 §2.3 — bright pixels fire first
            return spikegen.latency(
                imgs,
                num_steps=self.num_steps,
                tau=5,
                threshold=0.01,
                normalize=True,
                linear=True,
                clip=True,
            )
        else:
            raise ValueError(
                f"Unknown encoding '{self.encoding}'. "
                f"Choose 'rate' or 'latency'."
            )

    # ------------------------------------------------------------------
    def batches(self):
        """
        Yield (spike_input, labels) for every minibatch.

        spike_input : [num_steps, batch, 1, 28, 28]
        labels      : [batch]  — integer class indices 0-9
        """
        for imgs, labels in self.loader:
            spikes = self._encode(imgs)   # [T, B, 1, 28, 28]
            yield spikes, labels

    # ------------------------------------------------------------------
    @classmethod
    def train_test(cls, data_path: str = "./data",
                   batch_size: int = 128,
                   num_steps: int = 25,
                   encoding: str = "rate"):
        """
        Convenience constructor — returns (train_stream, test_stream).

        Usage:
            train_stream, test_stream = MNISTDataStream.train_test(
                batch_size=128, num_steps=25, encoding="rate"
            )
        """
        train = cls(data_path, batch_size, num_steps, encoding, train=True)
        test  = cls(data_path, batch_size, num_steps, encoding, train=False)
        return train, test


# ──────────────────────────────────────────────────────────────────────────────
# SNN LAYER  (dynamic multi-layer builder)
# ──────────────────────────────────────────────────────────────────────────────

class SNNLayer(nn.Module):
    """
    Builds a fully-connected multi-layer SNN from a network_structure list.

    network_structure : [input_size, hidden1, hidden2, …, output_size]
                        Comes directly from Settings.network_structure in snn_config.

    neuron_type       : key in NEURON_REGISTRY  (e.g. "leaky", "synaptic", …)
    output_neuron     : key for the final layer  (use "leaky_integrator" for regression)
    timesteps         : number of simulation steps per forward pass
    neuron_cfg        : dict of extra kwargs forwarded to get_neuron()
    """

    def __init__(self, network_structure: list, neuron_type: str = "leaky",
                 output_neuron: str = "leaky_integrator",
                 timesteps: int = 25, neuron_cfg: dict = None):
        super().__init__()
        assert len(network_structure) >= 2, \
            "network_structure must have at least [input, output]."

        self.structure      = network_structure
        self.neuron_type    = neuron_type
        self.output_neuron  = output_neuron
        self.timesteps      = timesteps
        self.neuron_cfg     = neuron_cfg or {}

        self.fc_layers  = nn.ModuleList()
        self.snn_layers = nn.ModuleList()

        # Build pairs: (Linear, SpikingNeuron) for each layer transition
        for i in range(len(network_structure) - 1):
            in_f  = network_structure[i]
            out_f = network_structure[i + 1]
            is_output = (i == len(network_structure) - 2)

            self.fc_layers.append(nn.Linear(in_f, out_f))

            n_type = output_neuron if is_output else neuron_type
            beta   = torch.rand(out_f)
            self.snn_layers.append(
                get_neuron(n_type, in_f, out_f, beta, self.neuron_cfg)
            )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Time-unrolled forward pass.

        Parameters
        ----------
        x : [timesteps, batch_size, input_size]

        Returns
        -------
        spk_rec : [timesteps, batch_size, output_size]  – output spikes
        mem_rec : [timesteps, batch_size, output_size]  – output membrane potential
        """
        T = x.shape[0]

        # Initialise hidden states for every spiking layer
        mem_states = []
        for layer in self.snn_layers:
            mem_states.append(layer.init_leaky())

        spk_rec = []
        mem_rec = []

        for step in range(T):
            x_t = x[step]                    # [batch, ...] may be [B,1,28,28]
            x_t = x_t.view(x_t.size(0), -1) # flatten → [batch, 784]

            for idx, (fc, lif) in enumerate(
                    zip(self.fc_layers, self.snn_layers)):
                x_t = fc(x_t)           # linear projection
                spk_t, mem_states[idx] = lif(x_t, mem_states[idx])
                x_t = spk_t             # pass spikes to next layer

            # x_t is now the output layer spike; mem_states[-1] is output membrane
            spk_rec.append(spk_t)
            mem_rec.append(mem_states[-1])

        return torch.stack(spk_rec), torch.stack(mem_rec)

    # ------------------------------------------------------------------
    def reset_hidden(self):
        """Explicit hidden-state reset (calls snntorch utils.reset)."""
        utils.reset(self)


# ──────────────────────────────────────────────────────────────────────────────
# FORWARD PASS HELPER  (matches snntorch quickstart pattern)
# ──────────────────────────────────────────────────────────────────────────────

def forward_pass(net: SNNLayer, data: torch.Tensor):
    """
    Run a complete time-unrolled forward pass and return accumulated records.

    data : [timesteps, batch, input_size]  — already spike-encoded
    """
    net.reset_hidden()
    spk_rec, mem_rec = net(data)
    return spk_rec, mem_rec


# ──────────────────────────────────────────────────────────────────────────────
# SUPERVISED TRAINING LOOP  (BPTT with surrogate gradients)
# ──────────────────────────────────────────────────────────────────────────────

def train_supervised(net: SNNLayer, data_stream: MNISTDataStream,
                     loss_fn, optimizer: torch.optim.Optimizer,
                     epochs: int, device: torch.device,
                     loss_name: str = "mse_count") -> list:
    """
    Backpropagation Through Time (BPTT) training with surrogate gradients.

    Control flow:
      for each epoch:
        for each batch:
          1. reset hidden states
          2. time-unrolled forward pass  →  spk_rec, mem_rec
          3. compute loss on spk_rec (rate-based) or mem_rec (membrane)
          4. zero_grad → backward → clip → step

    Returns list of per-batch loss values.
    """
    net.train()
    net.to(device)
    loss_hist = []

    for epoch in range(epochs):
        epoch_losses = []

        for spk_input, labels in data_stream.batches():
            spk_input = spk_input.to(device)
            labels    = labels.to(device)

            # ── forward ──────────────────────────────────────────────
            net.reset_hidden()
            spk_rec, mem_rec = net(spk_input)

            # ── loss ─────────────────────────────────────────────────
            if loss_name in ("mse_count", "ce_count"):
                # spk_rec: [T, batch, output] → loss expects (spk, target)
                loss_val = loss_fn(spk_rec, labels)
            elif loss_name in ("ce_rate",):
                loss_val = loss_fn(mem_rec, labels)
            else:
                # membrane regression losses: compare mem to one-hot targets
                one_hot = nn.functional.one_hot(
                    labels, num_classes=mem_rec.shape[-1]
                ).float().unsqueeze(0).expand_as(mem_rec)
                loss_val = loss_fn(mem_rec, one_hot)

            # ── backward ─────────────────────────────────────────────
            optimizer.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            loss_hist.append(loss_val.item())
            epoch_losses.append(loss_val.item())

        mean_loss = statistics.mean(epoch_losses)
        print(f"[Supervised] Epoch {epoch+1:>3}/{epochs}  "
              f"mean loss: {mean_loss:.4e}")

    return loss_hist


# ──────────────────────────────────────────────────────────────────────────────
# UNSUPERVISED TRAINING LOOP  (STDP-trace)
# ──────────────────────────────────────────────────────────────────────────────

def train_unsupervised(net: SNNLayer, data_stream: MNISTDataStream,
                       loss_fn, optimizer: torch.optim.Optimizer,
                       epochs: int, device: torch.device) -> list:
    """
    Unsupervised STDP-trace training.

    Labels are never used.  The loss is a Hebbian correlation between
    pre-synaptic spike traces and post-synaptic spikes, approximating the
    STDP potentiation rule entirely in software.

    Control flow (differs from supervised):
      for each epoch:
        for each batch:
          1. reset hidden states
          2. time-unrolled forward pass  →  full layer-wise spike records
          3. compute STDP trace loss between consecutive layer spikes
          4. zero_grad → backward → step   (no label ever used)

    Note: Because STDP is inherently a layer-local rule, we compute the
    loss between the input spikes and the first hidden layer's output spikes.
    This can be extended to deeper layers by chaining the same operation.

    Returns list of per-batch loss values.
    """
    net.train()
    net.to(device)
    loss_hist = []

    for epoch in range(epochs):
        epoch_losses = []

        for spk_input, _labels in data_stream.batches():   # labels discarded
            spk_input = spk_input.to(device)               # [T, B, in_size]

            # ── forward ──────────────────────────────────────────────
            net.reset_hidden()
            spk_rec, _ = net(spk_input)
            # spk_rec: [T, B, out_size]  — output of the entire network

            # ── STDP loss: correlate input layer with first output ────
            # We compare raw input spikes to output spikes
            loss_val = loss_fn(spk_input, spk_rec)

            # ── backward ─────────────────────────────────────────────
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            loss_hist.append(loss_val.item())
            epoch_losses.append(loss_val.item())

        mean_loss = statistics.mean(epoch_losses)
        print(f"[Unsupervised-STDP] Epoch {epoch+1:>3}/{epochs}  "
              f"mean loss: {mean_loss:.4e}")

    return loss_hist


# ──────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────────────────────

class SNNMetrics:
    """
    Computes and stores SNN-specific performance metrics.

    Metrics tracked
    ---------------
    Supervised:
      - accuracy          : rate-code classification accuracy  (SF.accuracy_rate)
      - mean_firing_rate  : average spikes per neuron per time step
      - sparsity          : fraction of neurons silent (firing rate < threshold)
      - loss_curve        : training loss over iterations

    Unsupervised:
      - mean_firing_rate  : as above
      - sparsity          : as above
      - weight_change_norm: L2 norm of weight updates per epoch

    Shared:
      - synops            : synaptic operations = Σ (spikes × fan-out weights)
      - membrane_variance : variance of output membrane potentials (stability indicator)
    """

    def __init__(self):
        self.history = {
            "accuracy"            : [],
            "mean_firing_rate"    : [],
            "sparsity"            : [],
            "loss"                : [],
            "synops"              : [],
            "membrane_variance"   : [],
            "weight_change_norm"  : [],
        }

    # ------------------------------------------------------------------
    def update_from_forward(self, spk_rec: torch.Tensor,
                             mem_rec: torch.Tensor,
                             labels: torch.Tensor = None,
                             loss_val: float = None):
        """
        Called once per batch after the forward pass.

        spk_rec : [T, batch, output_size]
        mem_rec : [T, batch, output_size]
        labels  : [batch]  — optional; required for accuracy
        loss_val: scalar float — optional
        """
        with torch.no_grad():
            T, B, N = spk_rec.shape

            # ── firing rate ──────────────────────────────────────────
            mean_fr = spk_rec.float().mean().item()
            self.history["mean_firing_rate"].append(mean_fr)

            # ── sparsity (fraction of neurons that never fired) ───────
            fired     = spk_rec.sum(dim=0).float()   # [B, N]
            sparsity  = (fired == 0).float().mean().item()
            self.history["sparsity"].append(sparsity)

            # ── synaptic operations estimate ─────────────────────────
            # Each spike in hidden layer triggers one synaptic op per
            # output weight.  Here approximated as total spike count.
            synops = spk_rec.sum().item()
            self.history["synops"].append(synops)

            # ── membrane variance ────────────────────────────────────
            mem_var = mem_rec.float().var().item()
            self.history["membrane_variance"].append(mem_var)

            # ── accuracy (supervised only) ───────────────────────────
            if labels is not None:
                acc = SF.accuracy_rate(spk_rec, labels)
                self.history["accuracy"].append(acc)

            # ── loss ─────────────────────────────────────────────────
            if loss_val is not None:
                self.history["loss"].append(loss_val)

    # ------------------------------------------------------------------
    def update_weight_change(self, net: nn.Module,
                              prev_params: dict):
        """
        Compute L2 norm of weight changes since prev_params snapshot.
        Call before optimizer.step() to capture Δw.

        prev_params : dict of {name: tensor} from _snapshot_params()
        """
        total_norm = 0.0
        for name, param in net.named_parameters():
            if name in prev_params:
                delta = (param.data - prev_params[name]).norm(2).item()
                total_norm += delta ** 2
        self.history["weight_change_norm"].append(total_norm ** 0.5)

    # ------------------------------------------------------------------
    @staticmethod
    def snapshot_params(net: nn.Module) -> dict:
        """Return a detached copy of all named parameters."""
        return {name: p.data.clone()
                for name, p in net.named_parameters()}

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Return scalar means for all tracked metrics."""
        out = {}
        for key, values in self.history.items():
            if values:
                out[key] = statistics.mean(values)
        return out

    # ------------------------------------------------------------------
    def print_summary(self):
        """Print a formatted metrics table."""
        s = self.summary()
        print("\n" + "─" * 48)
        print(f"{'SNN PERFORMANCE METRICS':^48}")
        print("─" * 48)
        fmt = {
            "accuracy"           : "{:.2%}",
            "mean_firing_rate"   : "{:.4f}",
            "sparsity"           : "{:.2%}",
            "loss"               : "{:.4e}",
            "synops"             : "{:.1f}",
            "membrane_variance"  : "{:.4e}",
            "weight_change_norm" : "{:.4e}",
        }
        for key, val in s.items():
            label = key.replace("_", " ").capitalize()
            f = fmt.get(key, "{:.4f}")
            print(f"  {label:<26} {f.format(val)}")
        print("─" * 48 + "\n")

    # ------------------------------------------------------------------
    def plot_loss(self, save_path: str = None):
        """Plot training loss curve and optionally save to file."""
        if not self.history["loss"]:
            print("No loss history to plot.")
            return
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(self.history["loss"], linewidth=1.2, color="#2c7bb6")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
            print(f"Loss curve saved → {save_path}")
        plt.close()

    def plot_firing_rate(self, save_path: str = None):
        """Plot mean firing rate over training."""
        if not self.history["mean_firing_rate"]:
            print("No firing rate history to plot.")
            return
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(self.history["mean_firing_rate"],
                linewidth=1.2, color="#d7191c")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean Firing Rate")
        ax.set_title("Output Neuron Mean Firing Rate")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
            print(f"Firing rate plot saved → {save_path}")
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION  (shared for both modes)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(net: SNNLayer, data_stream: MNISTDataStream,
             device: torch.device, metrics: SNNMetrics = None,
             mode: str = LearningMode.SUPERVISED):
    """
    Run inference over the test stream and populate metrics.

    For supervised mode: also computes per-batch accuracy.
    For unsupervised mode: only sparsity and firing-rate metrics are collected.
    """
    net.eval()
    net.to(device)

    if metrics is None:
        metrics = SNNMetrics()

    with torch.no_grad():
        for spk_input, labels in data_stream.batches():
            spk_input = spk_input.to(device)
            labels    = labels.to(device)

            net.reset_hidden()
            spk_rec, mem_rec = net(spk_input)

            lbl = labels if mode == LearningMode.SUPERVISED else None
            metrics.update_from_forward(spk_rec, mem_rec, labels=lbl)

    metrics.print_summary()
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR  (wires everything together from a Settings config)
# ──────────────────────────────────────────────────────────────────────────────

def build_and_run(cfg, csv_path: str,
                  mode: str = LearningMode.SUPERVISED,
                  loss_name: str = "mse_count",
                  neuron_type: str = "leaky",
                  encoding: str = "rate",
                  output_dir: str = "/mnt/user-data/outputs"):
    """
    High-level entry point.

    cfg       : Settings instance from snn_config.py
    csv_path  : path to mnist_train_small.csv
    mode      : LearningMode.SUPERVISED | LearningMode.UNSUPERVISED
    loss_name : key into the appropriate loss registry
    neuron_type: key into NEURON_REGISTRY
    encoding  : "rate" or "latency"  (Tutorial 1 spike encoding)

    Steps
    -----
    1. Load MNIST CSV → normalise → spike-encode (Tutorial 1 pattern)
    2. Build SNNLayer from cfg.network_structure  (input fixed at 784)
    3. Select loss function based on mode + loss_name
    4. Run training loop (supervised or unsupervised path)
    5. Evaluate on test split
    6. Print and save metrics plots
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}")
    print(f"Learning mode    : {mode}")
    print(f"Loss function    : {loss_name}")
    print(f"Neuron type      : {neuron_type}")
    print(f"Spike encoding   : {encoding}\n")

    # ── 1. Data  (CSV → normalise → spikegen.rate / latency) ─────────
    train_stream, test_stream = CSVMNISTDataStream.train_test(
        csv_path=csv_path,
        batch_size=cfg.BATCH_SIZE,
        num_steps=cfg.TIMESTEPS,
        encoding=encoding,
    )

    # MNIST is always 784 inputs, 10 output classes
    mnist_input_size  = 784
    mnist_output_size = 10
    structure = [mnist_input_size] + \
                [cfg.HIDDEN_SIZE] * cfg.HIDDEN_LAYERS + \
                [mnist_output_size]
    print(f"Network structure: {structure}")

    # ── 2. Model ─────────────────────────────────────────────────────
    output_neuron = "leaky_integrator" if mode == LearningMode.SUPERVISED \
        else "leaky"
    net = SNNLayer(
        network_structure=structure,
        neuron_type=neuron_type,
        output_neuron=output_neuron,
        timesteps=cfg.TIMESTEPS,
        neuron_cfg={"threshold": cfg.THRESHOLD, "learn_beta": True},
    )

    # ── 3. Loss & Optimizer ──────────────────────────────────────────
    loss_fn   = get_loss_function(loss_name, mode=mode)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # ── 4. Train ─────────────────────────────────────────────────────
    metrics = SNNMetrics()

    if mode == LearningMode.SUPERVISED:
        loss_hist = train_supervised(
            net, train_stream, loss_fn, optimizer,
            epochs=cfg.EPOCHS, device=device, loss_name=loss_name,
        )
    else:
        loss_hist = train_unsupervised(
            net, train_stream, loss_fn, optimizer,
            epochs=cfg.EPOCHS, device=device,
        )

    metrics.history["loss"] = loss_hist

    # ── 5. Evaluate ──────────────────────────────────────────────────
    evaluate(net, test_stream, device, metrics=metrics, mode=mode)

    # ── 6. Save plots ────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    metrics.plot_loss(save_path=os.path.join(output_dir, "loss_curve.png"))
    metrics.plot_firing_rate(
        save_path=os.path.join(output_dir, "firing_rate.png")
    )

    return net, metrics
