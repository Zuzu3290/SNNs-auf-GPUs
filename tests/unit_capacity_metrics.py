"""Unit tests for the scalability study's capacity metrics: Participation Ratio,
spike-train entropy, and (added in a later task) mutual information.

    python tests/unit_capacity_metrics.py

Every function is checked against a case with a hand-computable right answer before
it's trusted on real spike data. CPU-only, no dataset, no download, no model.
"""
from __future__ import annotations

import numpy as np
import torch

from _harness import Suite
from learning.capacity_metrics import (
    firing_rate_matrix, participation_ratio, spike_entropy,
    pca_reduce, quantile_discretize, discrete_mutual_information,
    mutual_information_xz, mutual_information_zy,
)

suite = Suite("unit_capacity_metrics")


def test_firing_rate_matrix_shape_and_values() -> None:
    """[T, B, C, H, W] -> [B, N], mean over T, flattened over the rest."""
    # T=4, B=2, one neuron that always fires (rate 1.0) and one that never does (0.0),
    # arranged as a [T, B, 2] tensor so N=2 after flattening (nothing to flatten here).
    spikes = torch.zeros(4, 2, 2)
    spikes[:, :, 0] = 1.0  # neuron 0 always fires
    rates = firing_rate_matrix(spikes)
    suite.check("shape is [B, N]", rates.shape == (2, 2), f"got {rates.shape}")
    suite.check("always-firing neuron has rate 1.0", np.allclose(rates[:, 0], 1.0))
    suite.check("never-firing neuron has rate 0.0", np.allclose(rates[:, 1], 0.0))


def test_firing_rate_matrix_flattens_spatial_dims() -> None:
    """[T, B, C, H, W] collapses C*H*W into one N axis."""
    spikes = torch.rand(3, 5, 2, 4, 4) > 0.5
    rates = firing_rate_matrix(spikes.float())
    suite.check("N = C*H*W", rates.shape == (5, 2 * 4 * 4), f"got {rates.shape}")


def test_participation_ratio_one_neuron_dominates() -> None:
    """One neuron carries all the variance across samples, the rest are constant
    (zero variance) -- PR should be close to 1 (one effective dimension)."""
    rng = np.random.default_rng(0)
    rates = np.zeros((50, 10))
    rates[:, 0] = rng.normal(size=50)  # only neuron 0 varies
    pr = participation_ratio(rates)
    suite.check("PR close to 1 when one neuron carries all variance",
                0.9 <= pr <= 1.5, f"got {pr}")


def test_participation_ratio_all_neurons_independent_equal_variance() -> None:
    """All neurons vary independently with equal variance -- PR should approach the
    neuron count (every dimension equally effective)."""
    rng = np.random.default_rng(0)
    n = 20
    rates = rng.normal(size=(200, n))  # independent, equal variance
    pr = participation_ratio(rates)
    suite.check(f"PR approaches N={n} for independent equal-variance neurons",
                pr >= n * 0.6, f"got {pr}")


def test_spike_entropy_uniform_is_maximal() -> None:
    """Every neuron fires at the same rate -> maximal entropy = log2(N)."""
    n = 8
    rates = np.ones((10, n))  # every neuron, every sample, fires at rate 1.0 -- uniform
    h = spike_entropy(rates)
    suite.check("uniform firing gives H = log2(N)",
                np.isclose(h, np.log2(n), atol=1e-6), f"got {h}, expected {np.log2(n)}")


def test_spike_entropy_one_neuron_dominates_is_zero() -> None:
    """Only one neuron ever fires -> entropy is 0 (no uncertainty about which neuron)."""
    rates = np.zeros((10, 8))
    rates[:, 0] = 1.0  # only neuron 0 fires, at every sample
    h = spike_entropy(rates)
    suite.check("single active neuron gives H = 0", np.isclose(h, 0.0, atol=1e-9),
                f"got {h}")


def test_discrete_mutual_information_identical_arrays_equals_entropy() -> None:
    """Two identical discrete variables: MI(a;a) = H(a) exactly. a is 50/50 over two
    symbols, so H(a) = 1 bit."""
    a = np.array([0, 0, 1, 1] * 25)  # 100 samples, balanced
    mi = discrete_mutual_information(a, a.copy())
    suite.check("MI(a;a) = 1 bit for a balanced binary variable",
                np.isclose(mi, 1.0, atol=1e-6), f"got {mi}")


def test_discrete_mutual_information_independent_is_near_zero() -> None:
    """Two independently shuffled copies of a balanced variable should show MI close
    to 0 -- not exactly 0 (finite-sample estimate), so this is a bounds check."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=2000)
    b = rng.integers(0, 4, size=2000)  # independently drawn, not derived from a
    mi = discrete_mutual_information(a, b)
    suite.check("MI of independent variables is small", 0.0 <= mi < 0.05, f"got {mi}")


def test_discrete_mutual_information_is_never_negative() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, size=500)
    b = rng.integers(0, 5, size=500)
    mi = discrete_mutual_information(a, b)
    suite.check("MI >= 0", mi >= -1e-9, f"got {mi}")


def test_pca_reduce_shape_and_determinism() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 15))
    reduced_a = pca_reduce(data, n_components=3)
    reduced_b = pca_reduce(data, n_components=3)
    suite.check("output shape is [B, k]", reduced_a.shape == (40, 3),
                f"got {reduced_a.shape}")
    suite.check("same input gives same output", np.allclose(reduced_a, reduced_b))


def test_quantile_discretize_shape_and_range() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 3))
    symbols = quantile_discretize(data, n_bins=6)
    suite.check("output is one symbol per sample", symbols.shape == (30,),
                f"got {symbols.shape}")
    suite.check("symbols fit in the joint bin space", symbols.max() < 6 ** 3,
                f"max={symbols.max()}")
    suite.check("symbols are non-negative", symbols.min() >= 0)


def test_mutual_information_xz_and_zy_are_bounded() -> None:
    """End-to-end check on synthetic spike-shaped data: MI must be non-negative and
    not absurdly large for random data -- an approximation's bounds check, not an
    exact-value check, since PCA+binning is a lossy pipeline."""
    rng = np.random.default_rng(0)
    x_rates = rng.random((64, 200))    # e.g. a flattened input batch
    z_rates = rng.random((64, 50))     # e.g. a hidden layer's firing rates
    labels = rng.integers(0, 10, size=64)

    mi_xz = mutual_information_xz(x_rates, z_rates)
    mi_zy = mutual_information_zy(z_rates, labels)

    suite.check("I(X;Z) >= 0", mi_xz >= -1e-9, f"got {mi_xz}")
    suite.check("I(Z;Y) >= 0", mi_zy >= -1e-9, f"got {mi_zy}")
    # Joint symbol space is at most 6**3 = 216 states; MI cannot exceed log2(216).
    suite.check("I(X;Z) is within the joint symbol space's entropy bound",
                mi_xz <= np.log2(6 ** 3) + 1e-6, f"got {mi_xz}")


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_firing_rate_matrix_shape_and_values,
        test_firing_rate_matrix_flattens_spatial_dims,
        test_participation_ratio_one_neuron_dominates,
        test_participation_ratio_all_neurons_independent_equal_variance,
        test_spike_entropy_uniform_is_maximal,
        test_spike_entropy_one_neuron_dominates_is_zero,
        test_discrete_mutual_information_identical_arrays_equals_entropy,
        test_discrete_mutual_information_independent_is_near_zero,
        test_discrete_mutual_information_is_never_negative,
        test_pca_reduce_shape_and_determinism,
        test_quantile_discretize_shape_and_range,
        test_mutual_information_xz_and_zy_are_bounded,
    ]))
