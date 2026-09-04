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


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_firing_rate_matrix_shape_and_values,
        test_firing_rate_matrix_flattens_spatial_dims,
        test_participation_ratio_one_neuron_dominates,
        test_participation_ratio_all_neurons_independent_equal_variance,
        test_spike_entropy_uniform_is_maximal,
        test_spike_entropy_one_neuron_dominates_is_zero,
    ]))
