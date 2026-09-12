"""
Small standalone probe: is a learnable per-layer membrane decay (beta) a better
adaptive lever than a learnable threshold?

Motivation: every framework wired into this repo (snnTorch/Norse/SpikingJelly/Sinabs)
keeps both threshold and decay FIXED and matched across frameworks on purpose -- see
the parity comments in configuration/network_architecture.yaml. This script does not
touch that config or any adapter. It runs a side-by-side comparison, entirely outside
the main pipeline, between:

  1. frameworks.snn_lif.SNN_LIF        -- fixed threshold, fixed decay (the baseline)
  2. SNN_LIF_LearnableDecay (below)    -- fixed threshold, ONE learned decay per layer

Both start from identical weights (seeded) and an identical decay value (LIF_BETA), so
any difference in accuracy or spike rate comes from letting decay move.

This measures spike-rate SPARSITY as a proxy only. It does NOT measure GPU energy: this
repo's LIF execution is dense (every neuron computed every timestep regardless of spike
activity), so a lower spike rate here would not translate into a lower measured energy
draw without a sparse/event-driven compute path, which does not exist in this pipeline.

    python learnable_decay_probe.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

from skeleton.snn_config import Settings
from frameworks.snn_lif import SNN_LIF, SurrogateSpike, LIF_BETA, LIF_THRESHOLD
from learning.training import aggregate_spike_output
from learning.utilities import ActivityMonitor
from event_data_workflow import NeuromorphicEncoder

NUM_BATCHES        = 100                  # a probe, not a full epoch
PROBE_DATASET_NAME = "DVS128 Gesture"     # deliberate choice, not the N-MNIST prompt default --
                                           # real event-camera data, not converted static images
LEARNABLE_INIT_BETA = 0.1                 # deliberately far from the fixed baseline's 0.9 -- tests
                                           # whether gradient descent can find a working decay from
                                           # a bad starting point, not just nudge around a good one


class LearnableDecayLIFCell(nn.Module):
    """Same neuron as frameworks.snn_lif.LIFCell, but beta is one learned scalar
    per layer instead of the fixed LIF_BETA constant. Threshold stays fixed --
    this isolates decay as the adaptive lever, not threshold."""

    def __init__(self, threshold: float = LIF_THRESHOLD, init_beta: float = LIF_BETA):
        super().__init__()
        init_beta = min(max(init_beta, 1e-4), 1 - 1e-4)
        self.raw_beta = nn.Parameter(torch.tensor(math.log(init_beta / (1 - init_beta))))
        self.threshold = threshold
        self.mem = None

    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_beta)  # keeps decay in (0, 1) whichever way gradient pushes it

    def reset_state(self) -> None:
        self.mem = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mem is None:
            self.mem = torch.zeros_like(x)
        self.mem = self.beta() * self.mem + x
        spk = SurrogateSpike.apply(self.mem, self.threshold)
        self.mem = self.mem * (1.0 - spk)
        return spk


class SNN_LIF_LearnableDecay(SNN_LIF):
    """SNN_LIF with LearnableDecayLIFCell in place of the fixed-beta LIFCell.
    Topology, optimizer type, and loss are otherwise identical to the baseline."""

    def __init__(self, cfg: Settings, init_beta: float = LIF_BETA):
        super().__init__(cfg)
        self.lif1    = LearnableDecayLIFCell(init_beta=init_beta).to(self.device)
        self.lif2    = LearnableDecayLIFCell(init_beta=init_beta).to(self.device)
        self.lif_out = LearnableDecayLIFCell(init_beta=init_beta).to(self.device)

        # base __init__ built these against the now-replaced lif1/lif2 objects
        self.activity  = ActivityMonitor({'lif1': self.lif1, 'lif2': self.lif2})
        self.optimizer = torch.optim.Adam(self.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)


def quick_train(model, train_loader, num_batches: int) -> dict:
    """Cycles the loader across as many epochs as it takes to reach num_batches --
    needed for small datasets (DVS128 Gesture is ~1,077 training samples, so one
    pass at a calibrated batch size is far fewer than num_batches batches)."""
    model.train_mode()
    correct = total = 0
    spike_on = spike_total = 0
    batches_done = 0
    while batches_done < num_batches:
        for data, targets in train_loader:
            if batches_done >= num_batches:
                break
            targets = targets.long()
            model.zero_grad()
            spk_rec = model(data)
            loss = model.loss_fn(spk_rec, targets)
            model.backward_pass(loss)

            with torch.no_grad():
                logits = aggregate_spike_output(spk_rec.float())
                correct     += (logits.argmax(dim=1) == targets).sum().item()
                total       += targets.size(0)
                spike_on    += spk_rec.sum().item()
                spike_total += spk_rec.numel()
            batches_done += 1

    return {"accuracy": correct / total, "spike_rate": spike_on / spike_total}


def main() -> None:
    cfg = Settings()
    cfg.DATASET_NAME = PROBE_DATASET_NAME  # deliberate, non-interactive -- see resolve_dataset_entry
    encoder = NeuromorphicEncoder(cfg)
    train_loader, _ = encoder.get_dataloaders()

    print(f"Probe: dataset={cfg.DATASET_NAME}, {NUM_BATCHES} batches, "
          f"device={cfg.DEVICE}, batch_size={cfg.BATCH_SIZE}")

    torch.manual_seed(0)
    baseline        = SNN_LIF(cfg)
    baseline_result = quick_train(baseline, train_loader, NUM_BATCHES)

    torch.manual_seed(0)
    variant        = SNN_LIF_LearnableDecay(cfg, init_beta=LEARNABLE_INIT_BETA)
    variant_result = quick_train(variant, train_loader, NUM_BATCHES)

    learned_beta = {name: cell.beta().item() for name, cell in
                    (("lif1", variant.lif1), ("lif2", variant.lif2), ("lif_out", variant.lif_out))}

    print(f"\n--- Fixed decay (beta = {LIF_BETA:.2f} everywhere, baseline) ---")
    print(f"  accuracy   : {baseline_result['accuracy'] * 100:.2f}%")
    print(f"  spike rate : {baseline_result['spike_rate'] * 100:.2f}%")

    print(f"\n--- Learnable per-layer decay (started at beta = {LEARNABLE_INIT_BETA:.2f}, "
          f"far from the fixed baseline's {LIF_BETA:.2f}) ---")
    print(f"  accuracy   : {variant_result['accuracy'] * 100:.2f}%")
    print(f"  spike rate : {variant_result['spike_rate'] * 100:.2f}%")
    print(f"  learned beta: {learned_beta}")

    print(
        "\nNOTE: spike rate above is a sparsity PROXY only, not an energy measurement. "
        "This repo's LIF execution is dense (every neuron computed every timestep "
        "regardless of spike activity), so a lower spike rate here does not by itself "
        "lower measured GPU energy draw -- that would require a sparse/event-driven "
        "compute path, which this pipeline does not have."
    )


if __name__ == "__main__":
    main()
