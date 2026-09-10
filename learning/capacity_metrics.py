"""Capacity metrics for the scalability study: Participation Ratio, spike-train
entropy, and mutual information (Z;Y). Pure functions over tensors/arrays -- no
dataset, no GPU, no model object required, which is what makes them unit-testable in
isolation.

Includes a channel-pooling rate reduction (channel_rate_matrix) so a wide conv layer's
spatial sites don't get counted as separate units, and normalized (size-independent)
variants of Participation Ratio and spike entropy alongside their raw values.

See docs/superpowers/specs/2026-09-04-capacity-metrics-design.md for the full design
and the formulas' sources (Scalability.pdf, the colleague's brief). Per the study
designer's correction (design doc §6b), these functions are called over the FULL test
set, not a single batch-sized sample -- see the Gram-matrix-vs-covariance choice in
participation_ratio() and pca_reduce() below, which is sized accordingly.

Opt-in only: nothing in this module is called unless
cfg.COMPUTE_CAPACITY_METRICS is true. See learning/inference.py's SNNTester.run()
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


def channel_rate_matrix(spikes: torch.Tensor) -> np.ndarray:
    """[T, B, C, H, W] (or [T, B, C] for an already-flat layer) -> [B, C] per-sample,
    per-CHANNEL mean firing rate -- spatial positions within a channel are pooled
    together, not treated as separate units.

    Per the study designer's correction (design doc §6c): a conv layer's raw spike
    tensor has one "unit" per (channel, y, x) site, which for a 128-filter layer on a
    large sensor is well over 100k units -- mostly spatial redundancy within each
    filter, not independent capacity. Pooling each channel's spatial map into one
    number makes the unit count equal to the filter count (12, 32, 64, 128, ...) --
    exactly the axis a width sweep varies -- so Participation Ratio and spike entropy
    measure feature dimensionality instead of spatial tiling.
    """
    rate = spikes.detach().float().mean(dim=0)  # [B, C, H, W] or [B, C]
    if rate.dim() > 2:
        rate = rate.mean(dim=tuple(range(2, rate.dim())))  # pool every spatial dim
    return rate.cpu().numpy()


def participation_ratio(rates: np.ndarray) -> float:
    """rates: [B, N] (B samples, N neurons). PR = (sum(lambda_i))**2 / sum(lambda_i**2),
    the eigenvalues of the neuron-by-neuron covariance matrix -- the exact formula from
    the scalability brief (Scalability.pdf p.3).

    Computed via whichever of the B x B Gram matrix or the N x N covariance is SMALLER
    for the shapes actually passed in, rather than always building the B x B one: the
    two have the SAME nonzero eigenvalues (up to the shared normalisation the PR ratio
    cancels out anyway), so which one is cheaper to eigendecompose depends on which of
    B (sample count) and N (neuron/channel count) is larger. With rates now sampled
    over the full test set (B possibly 10,000+) against a channel count N of a few
    dozen to a couple hundred, N < B is the common case -- the reverse of the old
    B << N diagnostic-batch assumption.

    Returns 1.0 for a degenerate all-constant input (no variance anywhere), since a
    single unchanging value is definitionally one effective dimension.
    """
    centered = rates - rates.mean(axis=0, keepdims=True)
    b, n = centered.shape
    gram = centered @ centered.T if b <= n else centered.T @ centered  # [B,B] or [N,N]
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)  # eigvalsh can return tiny negatives
    total = eigenvalues.sum()
    if total <= 1e-12:
        return 1.0
    return float((total ** 2) / (eigenvalues ** 2).sum())


def participation_ratio_normalized(rates: np.ndarray) -> float:
    """PR divided by the channel count N -- a size-independent reading (0 to 1-ish)
    alongside the raw value, per design doc §6d. N = rates.shape[1]."""
    n = rates.shape[1]
    if n <= 0:
        return 0.0
    return participation_ratio(rates) / n


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


def spike_entropy_normalized(rates: np.ndarray) -> float:
    """Entropy divided by its maximum possible value, log2(N) -- per design doc §6d.
    N=1 has no distribution to spread across (nothing to normalize against), so this
    returns 0.0 rather than dividing by log2(1)=0.
    """
    n = rates.shape[1]
    if n <= 1:
        return 0.0
    return spike_entropy(rates) / np.log2(n)


def pca_reduce(data: np.ndarray, n_components: int = 3) -> np.ndarray:
    """[B, N] -> [B, k]: project onto the top k principal components.

    Same choice as participation_ratio() -- eigendecompose whichever of the B x B
    Gram matrix or the N x N covariance is SMALLER for the shapes actually passed in,
    rather than always building the B x B one. Both paths give the identical [B, k]
    projection: the Gram-matrix path recovers directions in N-space via the dual
    trick (centered.T @ eigenvectors, rescaled), while the covariance path already
    produces them directly as its own eigenvectors. This matters because with rates
    now sampled over the full test set, B (10,000+) commonly exceeds N (a channel
    count of a few dozen to a couple hundred) -- the reverse of the old B << N
    diagnostic-batch assumption this used to hard-code. k is capped at B - 1 (or B if
    that would be non-positive) and at N, since neither a sample set of size B nor a
    feature space of size N can support more meaningful components than its own size.
    """
    b, n = data.shape
    k = max(1, min(n_components, b - 1 if b > 1 else 1, n))
    centered = data - data.mean(axis=0, keepdims=True)
    if b <= n:
        gram = centered @ centered.T  # [B, B]
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        # eigh ascends; take the top k.
        order = np.argsort(eigenvalues)[::-1][:k]
        top_vals = np.clip(eigenvalues[order], 1e-12, None)
        top_vecs = eigenvectors[:, order]
        # Recover the N-dimensional principal directions' projection via the dual
        # trick: centered.T @ top_vecs gives directions in N-space; scaling by
        # 1/sqrt(eigenvalue) normalizes them, then projecting `centered` back onto
        # them gives the [B, k] score.
        components = centered.T @ top_vecs / np.sqrt(top_vals)  # [N, k]
    else:
        cov = centered.T @ centered  # [N, N]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1][:k]
        components = eigenvectors[:, order]  # [N, k] -- already orthonormal directions
    return centered @ components  # [B, k]


def quantile_discretize(data: np.ndarray, n_bins: int = 6) -> np.ndarray:
    """[B, k] continuous -> [B] int symbols.

    Each of the k columns is independently bucketed into n_bins quantile bins (robust
    to columns having very different scales, unlike fixed-width bins), then the k
    per-column bin indices are combined into one joint symbol per sample via base-
    n_bins positional encoding -- symbol = sum(bin_j * n_bins**j). This is the same
    binning-of-activations approach the original deep-learning information-bottleneck
    research used (Tishby et al.), adapted here to work in a PCA-reduced space so it
    stays tractable in more than one dimension.
    """
    b, k = data.shape
    symbols = np.zeros(b, dtype=np.int64)
    for j in range(k):
        column = data[:, j]
        # np.quantile with n_bins+1 edges gives n_bins bins; searchsorted then clips
        # the top edge into the last bin instead of spilling into an (n_bins)-th one.
        edges = np.quantile(column, np.linspace(0, 1, n_bins + 1)[1:-1])
        bin_idx = np.searchsorted(edges, column, side="right")
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        symbols += bin_idx * (n_bins ** j)
    return symbols


def discrete_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Two integer-symbol arrays of the same length -> empirical mutual information in
    bits, from their joint histogram: I(A;B) = sum p(a,b) log2(p(a,b) / (p(a)p(b))).

    The one estimator mutual_information_zy calls, on already-discretized inputs. A
    finite-sample estimate: independent variables give a small positive number, not
    exactly zero (see the test suite's bounds checks).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    a_vals, a_inv = np.unique(a, return_inverse=True)
    b_vals, b_inv = np.unique(b, return_inverse=True)
    joint = np.zeros((len(a_vals), len(b_vals)))
    np.add.at(joint, (a_inv, b_inv), 1)
    joint /= n
    p_a = joint.sum(axis=1, keepdims=True)
    p_b = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0, joint / (p_a * p_b), 1.0)
        terms = np.where(joint > 0, joint * np.log2(ratio), 0.0)
    return float(terms.sum())


def mutual_information_zy(z_rates: np.ndarray, labels: np.ndarray) -> float:
    """z_rates: [B, N]. labels: [B] integer class labels, already discrete -- no
    reduction needed on that side. PCA-reduces and discretizes z only, then
    discrete_mutual_information against the raw labels."""
    z_symbols = quantile_discretize(pca_reduce(z_rates))
    return discrete_mutual_information(z_symbols, np.asarray(labels))
