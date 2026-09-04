"""Capacity metrics for the scalability study: Participation Ratio, spike-train
entropy, and (learning/capacity_metrics.py, extended in a later task) mutual
information. Pure functions over tensors/arrays -- no dataset, no GPU, no model object
required, which is what makes them unit-testable in isolation.

See docs/superpowers/specs/2026-09-04-capacity-metrics-design.md for the full design,
the formulas' sources (Scalability.pdf, the colleague's brief), and why the defaults
below are sized for a batch-sized sample rather than a large dataset.

Opt-in only: nothing in this module is called unless
cfg.COMPUTE_CAPACITY_METRICS is true. See learning/training.py's measure_activity()
for the call site.
"""
from __future__ import annotations

import numpy as np
import torch


def firing_rate_matrix(spikes: torch.Tensor) -> np.ndarray:
    """[T, B, ...] spike tensor -> [B, N] per-sample, per-neuron mean firing rate.

    Mean over the time dimension (T, axis 0), then every remaining dimension (channels,
    height, width, or nothing for an already-flat layer like lif_out) is flattened into
    one neuron axis N. This is the same rate-reduction ActivityMonitor's own
    firing_rate_tensor() does per-buffer, just kept per-sample here instead of
    collapsing the batch too, since Participation Ratio and mutual information both
    need per-sample variation to measure something.
    """
    rate = spikes.detach().float().mean(dim=0)  # [B, ...]
    return rate.reshape(rate.shape[0], -1).cpu().numpy()


def participation_ratio(rates: np.ndarray) -> float:
    """rates: [B, N] (B samples, N neurons). PR = (sum(lambda_i))**2 / sum(lambda_i**2),
    the eigenvalues of the neuron-by-neuron covariance matrix -- the exact formula from
    the scalability brief (Scalability.pdf p.3).

    Computed via the Gram-matrix trick rather than the N x N covariance directly: for
    B << N (a wide conv layer has far more neurons than a diagnostic batch has
    samples), the B x B matrix rates_c @ rates_c.T has the SAME nonzero eigenvalues as
    the N x N covariance (up to the shared normalisation the PR ratio cancels out
    anyway), and is far cheaper to eigendecompose.

    Returns 1.0 for a degenerate all-constant input (no variance anywhere), since a
    single unchanging value is definitionally one effective dimension.
    """
    centered = rates - rates.mean(axis=0, keepdims=True)
    gram = centered @ centered.T  # [B, B]
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)  # eigvalsh can return tiny negatives
    total = eigenvalues.sum()
    if total <= 1e-12:
        return 1.0
    return float((total ** 2) / (eigenvalues ** 2).sum())


def spike_entropy(rates: np.ndarray) -> float:
    """rates: [B, N]. Mean firing rate per neuron (averaged over the B samples),
    normalized into a probability distribution over neurons, Shannon entropy of that
    distribution in bits.

    High entropy = activity spread evenly across neurons ("undifferentiated firing",
    the brief's own description of a struggling large network). Low entropy = a
    handful of neurons dominate. Returns 0.0 for an all-silent layer (no firing to
    measure a distribution over).
    """
    per_neuron = rates.mean(axis=0)  # [N]
    total = per_neuron.sum()
    if total <= 1e-12:
        return 0.0
    p = per_neuron / total
    nonzero = p[p > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())
