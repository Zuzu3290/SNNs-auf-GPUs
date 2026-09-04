# Capacity Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 4 metrics `experiment_plan_final.md` §5 identified as missing before ex7 (Mutual Information, Participation Ratio, spike-train entropy, per-layer gradient norms), plus fix a real correctness gap found while designing where to store them — `lif_out` is currently never monitored, and the layer-naming logic will silently mislabel layers once a depth ladder adds hidden LIF layers.

**Architecture:** A depth-safe, dynamic layer-naming/hooking fix in `frameworks/spiking_net.py` + `frameworks/snn_model.py` (always on, benefits every experiment) underpins everything else. On top of it: a new pure-function module `learning/capacity_metrics.py` (PR, entropy, MI — no dataset/GPU needed to test), a deferred-sync gradient-norm tracker added to `learning/training.py`'s existing training loop, and a reuse of `learning/training.py`'s existing once-per-epoch `measure_activity()` diagnostic pass to compute all three spike-based metrics on the final epoch. Everything lands as 5 new nullable columns on the existing `layers.csv` (`skeleton/results.py`, schema v2 → v3), gated behind one new opt-in config flag (`training.compute_capacity_metrics`, default `false`).

**Tech Stack:** Python, PyTorch, plain `numpy` (no new dependency — this repo deliberately keeps `scipy`/`sklearn` out of direct imports). Tests follow the existing `Suite`-based, CPU-only, no-pytest convention (`tests/_harness.py`, run via `python tests/unit_X.py`).

## Global Constraints

- No new dependency: PCA, binning, and the MI estimator are implemented in plain `numpy`.
- The new metrics are **opt-in only** (`training.compute_capacity_metrics`, default `false`); no existing experiment's behavior or output changes.
- The layer-naming/hooking fix (Task 1-2) is **always-on** — it changes `layers.csv` output for every future run (adds a `lif_out` row) regardless of the new flag.
- No `.item()`/`.cpu()` calls inside `learning/training.py`'s per-iteration batch loop — accumulate as GPU-resident tensors, sync once. This is an existing, repeatedly-stated rule in that file; gradient-norm tracking must follow it too.
- Every new pure function in `learning/capacity_metrics.py` must be testable with synthetic tensors, no dataset or GPU required.
- Full design reference: `docs/superpowers/specs/2026-09-04-capacity-metrics-design.md`. Read it if a task below references "the spec" for a formula or rationale not repeated here.

---

### Task 1: Depth-safe LIF layer naming

**Files:**
- Modify: `frameworks/spiking_net.py:139-146` (`named_lif_layers`)
- Test: `tests/unit_layer_naming.py` (new)

**Interfaces:**
- Consumes: `SpikingNet.lif_layers()` (existing, returns `list[BaseLIF]` in forward-pass order) — unchanged.
- Produces: `SpikingNet.named_lif_layers() -> dict[str, BaseLIF]` — same name and return type as today, but the naming rule changes from position-in-a-fixed-3-tuple to "last is always `lif_out`". Task 2 and Task 3 both call this method and depend on the new rule.

**Current bug:** `LIF_SLOTS[i] if i < len(LIF_SLOTS) else f"lif{i+1}"` names layers by position. With exactly 3 LIF layers today this accidentally works, but the moment a 4th LIF layer is inserted before the output (e.g. a future FC hidden layer), it takes index 2 and gets incorrectly named `"lif_out"`, while the true output layer shifts to index 3 and gets the anonymous fallback `"lif4"`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit_layer_naming.py`:

```python
"""Unit tests for depth-safe LIF layer naming and hooking.

    python tests/unit_layer_naming.py

Covers two fixes: named_lif_layers() must always call the LAST lif layer "lif_out"
regardless of how many precede it (not a fixed-position lookup), and every named layer
must actually be hooked by ActivityMonitor (not a hardcoded lif1/lif2 pair).

CPU-only, no dataset, no download.
"""
from __future__ import annotations

from _harness import FRAMEWORKS, Suite, build_model, fresh_cfg
from frameworks.spiking_net import SpikingNet
from frameworks.adapters.base import BaseLIF

suite = Suite("unit_layer_naming")


class _StubLIF(BaseLIF):
    """A do-nothing LIF for testing naming logic in isolation from any real framework."""

    def forward(self, x):
        return x

    def reset(self) -> None:
        pass


def test_three_layers_names_last_as_lif_out() -> None:
    """Today's architecture: exactly 3 LIF layers. The last one must be lif_out."""
    net = SpikingNet([_StubLIF(), _StubLIF(), _StubLIF()])
    named = net.named_lif_layers()
    suite.check("three layers: names are lif1, lif2, lif_out",
                list(named.keys()) == ["lif1", "lif2", "lif_out"])


def test_five_layers_still_names_last_as_lif_out() -> None:
    """Simulates a future depth ladder: 2 extra hidden LIF layers inserted before the
    output. The LAST layer must still be lif_out, not a positional slot 3/4/5."""
    layers = [_StubLIF() for _ in range(5)]
    net = SpikingNet(layers)
    named = net.named_lif_layers()
    names = list(named.keys())
    suite.check("five layers: exactly one is lif_out", names.count("lif_out") == 1)
    suite.check("five layers: lif_out is the LAST one",
                named["lif_out"] is layers[-1])
    suite.check("five layers: names are lif1..lif4 then lif_out",
                names == ["lif1", "lif2", "lif3", "lif4", "lif_out"])


def test_one_layer_is_just_lif_out() -> None:
    """Degenerate case: a single LIF layer is the output, not lif1."""
    layer = _StubLIF()
    net = SpikingNet([layer])
    named = net.named_lif_layers()
    suite.check("one layer: named lif_out, not lif1", list(named.keys()) == ["lif_out"])


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_three_layers_names_last_as_lif_out,
        test_five_layers_still_names_last_as_lif_out,
        test_one_layer_is_just_lif_out,
    ]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_layer_naming.py`
Expected: FAIL on the 5-layer and 1-layer cases (today's code names position 2 as `lif_out` and falls back to `lif4`/`lif5` for the rest; the 1-layer case names it `lif1` not `lif_out`). The 3-layer case passes today (that's the accidental case).

- [ ] **Step 3: Write minimal implementation**

In `frameworks/spiking_net.py`, replace lines 139-146:

```python
    def named_lif_layers(self) -> dict[str, BaseLIF]:
        """Slot name -> layer, using the names the rest of this pipeline expects
        (ActivityMonitor keys, synops_layer_map): lif1, lif2, ..., lif_out.

        The LAST lif layer is always "lif_out", whatever its position -- not a fixed
        3-slot lookup. That fixed lookup used to accidentally work only because this
        architecture happened to have exactly 3 LIF layers; the moment a hidden LIF
        layer is inserted before the output (a depth ladder), a positional lookup would
        misname the new layer "lif_out" and push the real output layer to an anonymous
        fallback name instead.
        """
        lifs = self.lif_layers()
        names = [f"lif{i + 1}" for i in range(len(lifs) - 1)] + ["lif_out"]
        return dict(zip(names, lifs))
```

Note `LIF_SLOTS` (line 28) becomes unused by this method after the change — leave the
constant in place for now (Task 2 does not need it removed, and removing it is out of
scope for this task; a later cleanup can drop it if nothing else references it).

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_layer_naming.py`
Expected: PASS, all 3 tests (this will fail if `SpikingNet.__init__` requires anything
beyond a list of modules — it doesn't; `nn.ModuleList(layers)` accepts the stub LIFs
fine since they subclass `nn.Module` via `BaseLIF`).

- [ ] **Step 5: Commit**

```bash
git add frameworks/spiking_net.py tests/unit_layer_naming.py
git commit -m "fix: name LIF layers by role (last = lif_out), not fixed position"
```

---

### Task 2: Dynamic ActivityMonitor hooking and SynOps layer map

**Files:**
- Modify: `frameworks/snn_model.py:66-69` (`SNNModel.__init__`, the `ActivityMonitor` construction)
- Modify: `frameworks/snn_model.py:125-133` (`synops_layer_map`)
- Test: `tests/unit_layer_naming.py` (add to the file from Task 1)

**Interfaces:**
- Consumes: `SpikingNet.named_lif_layers()` from Task 1 (now depth-safe).
- Produces: `SNNModel.activity` (an `ActivityMonitor`) that hooks every layer `named_lif_layers()` returns, and `SNNModel.synops_layer_map()` that includes every layer with a downstream dense module — both now depth-agnostic. Task 8 (the capacity-metrics wiring) depends on `lif_out` actually being present in `self.activity.recordings()`, which only becomes true after this task.

**Current bug:** both methods hardcode the tuple `("lif1", "lif2")`, so `lif_out` is
never hooked (no spike rate, CV-ISI, or SynOps ever recorded for it — confirmed by
`ex6/README.md`'s own "2 rows per arm — lif1, lif2 — hooked layers only"), and any
future hidden LIF layer from a depth ladder would be silently excluded from both too.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_layer_naming.py` (before the `if __name__ == "__main__":` block):

```python
def test_activity_monitor_hooks_every_lif_layer() -> None:
    """SNNModel.activity must hook ALL named LIF layers, including lif_out -- not a
    hardcoded lif1/lif2 pair."""
    model, cfg = build_model("sj")
    named = model.net.named_lif_layers()
    hooked = set(model.activity.buffers.keys())
    suite.check("lif_out is hooked", "lif_out" in hooked,
                f"hooked={sorted(hooked)}")
    suite.check("every named layer is hooked", hooked == set(named.keys()),
                f"named={sorted(named.keys())} hooked={sorted(hooked)}")


def test_synops_layer_map_includes_every_layer_with_a_downstream_dense() -> None:
    """synops_layer_map() must not hardcode lif1/lif2 either -- lif_out has no
    downstream dense layer (it's the last layer), so it's correctly ABSENT from the
    map, but that must be because dense_after() returns None for it, not because the
    iteration never considered it."""
    model, cfg = build_model("sj")
    mapping = model.synops_layer_map()
    suite.check("lif1 and lif2 are in the synops map",
                {"lif1", "lif2"} <= set(mapping.keys()))
    suite.check("lif_out is correctly absent (no downstream dense layer)",
                "lif_out" not in mapping)
```

Add both to the `suite.run([...])` list at the bottom.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_layer_naming.py`
Expected: FAIL on `test_activity_monitor_hooks_every_lif_layer` (`lif_out` absent from
`hooked`). `test_synops_layer_map_includes_every_layer_with_a_downstream_dense` already
passes today (it only asserts lif1/lif2 presence and lif_out absence, both already
true) — that's fine, it's here as a regression guard for Step 3's change, not a new
failure.

- [ ] **Step 3: Write minimal implementation**

In `frameworks/snn_model.py`, replace lines 66-69:

```python
        named = self.net.named_lif_layers()
        self.activity = ActivityMonitor(named)
```

And replace `synops_layer_map` (lines 125-133):

```python
    def synops_layer_map(self) -> dict:
        """Derived from the layer list rather than hand-written per framework, so it
        cannot fall out of sync with the architecture. Every named LIF layer is
        considered; a layer with no downstream dense module (lif_out, since it is the
        last layer) is correctly absent -- dense_after() returning None is what excludes
        it, not the iteration skipping it."""
        mapping = {}
        for name in self.net.named_lif_layers():
            downstream = self.net.dense_after(name)
            if downstream is not None:
                mapping[name] = downstream
        return mapping
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_layer_naming.py`
Expected: PASS, all 5 tests.

Also run the existing adapter/pipeline suites to confirm this doesn't regress anything
that reads `activity.buffers` or `synops_layer_map()` elsewhere:

Run: `python tests/run_all.py adapters pipeline_integration training_metrics`
Expected: all suites still PASS (they exercise the model build/forward/train path this
change touches).

- [ ] **Step 5: Commit**

```bash
git add frameworks/snn_model.py tests/unit_layer_naming.py
git commit -m "fix: hook every LIF layer dynamically, not a hardcoded lif1/lif2 pair"
```

---

### Task 3: `dense_before()` — the upstream weight layer for a LIF slot

**Files:**
- Modify: `frameworks/spiking_net.py` (add a method to `SpikingNet`, near the existing `dense_after`, ~line 148-166)
- Test: `tests/unit_layer_naming.py` (add to the file from Tasks 1-2)

**Interfaces:**
- Consumes: `SpikingNet.named_lif_layers()` (Task 1), `self.layers` (existing `nn.ModuleList`).
- Produces: `SpikingNet.dense_before(lif_name: str) -> nn.Module | None` — the nearest `Conv2d`/`Linear` **preceding** the given LIF slot in the layer list, or `None` if none exists. Task 7 (gradient norms) depends on this to find which weight layer's gradient belongs to which LIF slot's row.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_layer_naming.py`:

```python
def test_dense_before_pairs_each_lif_with_its_upstream_weight_layer() -> None:
    """dense_before(lif_name) is the Conv2d/Linear that FEEDS that LIF -- the mirror
    image of dense_after(), which finds the downstream one."""
    model, cfg = build_model("sj")
    net = model.net
    conv1_out = net.dense_before("lif1")
    conv2_out = net.dense_before("lif2")
    classifier = net.dense_before("lif_out")
    suite.check("lif1's upstream layer is a Conv2d",
                conv1_out is not None and type(conv1_out).__name__ == "Conv2d")
    suite.check("lif2's upstream layer is a Conv2d",
                conv2_out is not None and type(conv2_out).__name__ == "Conv2d")
    suite.check("lif_out's upstream layer is the classifier Linear",
                classifier is not None and type(classifier).__name__ == "Linear")
    suite.check("lif1 and lif2 have DIFFERENT upstream conv layers",
                conv1_out is not conv2_out)


def test_dense_before_returns_none_with_no_preceding_dense_layer() -> None:
    """A LIF layer with nothing but non-dense modules before it (or nothing at all)
    has no upstream weight layer."""
    import torch.nn as nn
    net = SpikingNet([_StubLIF()])
    suite.check("no dense layer before a lone LIF", net.dense_before("lif_out") is None)
```

Add both to the `suite.run([...])` list.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_layer_naming.py`
Expected: FAIL with `AttributeError: 'SpikingNet' object has no attribute 'dense_before'`.

- [ ] **Step 3: Write minimal implementation**

In `frameworks/spiking_net.py`, add this method to `SpikingNet`, directly after
`dense_after` (after line 166):

```python
    def dense_before(self, lif_name: str) -> nn.Module | None:
        """The nearest dense (Conv2d/Linear) module UPSTREAM of a given LIF slot -- the
        module whose weights actually produced that layer's input.

        Mirrors dense_after(), which walks forward for the SynOps estimate; this walks
        backward, for pairing a layer's gradient norm with the LIF slot it feeds. Used
        by SNNTrainer's gradient-norm tracking so the classifier's gradient lands on
        lif_out's row, whatever the network's current depth.
        """
        target = self.named_lif_layers().get(lif_name)
        if target is None:
            return None
        found = None
        for layer in self.layers:
            if layer is target:
                return found
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                found = layer
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_layer_naming.py`
Expected: PASS, all 7 tests.

- [ ] **Step 5: Commit**

```bash
git add frameworks/spiking_net.py tests/unit_layer_naming.py
git commit -m "feat: add dense_before(), the upstream-weight-layer counterpart to dense_after()"
```

---

### Task 4: `capacity_metrics.py` — firing rate reduction, Participation Ratio, spike entropy

**Files:**
- Create: `learning/capacity_metrics.py`
- Test: `tests/unit_capacity_metrics.py` (new)

**Interfaces:**
- Consumes: nothing from other tasks — pure functions over `torch.Tensor`/`np.ndarray`.
- Produces: `firing_rate_matrix(spikes: torch.Tensor) -> np.ndarray`, `participation_ratio(rates: np.ndarray) -> float`, `spike_entropy(rates: np.ndarray) -> float`. Task 5 adds more functions to the same file. Task 8 calls all of them.

- [ ] **Step 1: Write the failing test**

Create `tests/unit_capacity_metrics.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_capacity_metrics.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'learning.capacity_metrics'`.

- [ ] **Step 3: Write minimal implementation**

Create `learning/capacity_metrics.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_capacity_metrics.py`
Expected: PASS, all 6 tests.

- [ ] **Step 5: Register the new suite and commit**

In `tests/run_all.py`, add `"unit_capacity_metrics.py"` to the `SUITES` list (after
`"unit_training_metrics.py"`, before `"unit_entrypoints.py"`):

```python
SUITES = [
    "unit_neuron_spec.py",
    "unit_adapters.py",
    "unit_neuron_picker.py",
    "unit_shared_net.py",
    "unit_pipeline_integration.py",
    "unit_cli_config.py",
    "unit_seeding.py",
    "unit_training_metrics.py",
    "unit_capacity_metrics.py",
    "unit_layer_naming.py",
    "unit_entrypoints.py",
]
```

(This also registers `unit_layer_naming.py` from Tasks 1-3, if not already added.)

```bash
git add learning/capacity_metrics.py tests/unit_capacity_metrics.py tests/run_all.py
git commit -m "feat: add Participation Ratio and spike-train entropy (capacity_metrics.py)"
```

---

### Task 5: `capacity_metrics.py` — PCA reduction, discretization, and mutual information

**Files:**
- Modify: `learning/capacity_metrics.py` (append functions)
- Test: `tests/unit_capacity_metrics.py` (append tests)

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `pca_reduce(data, n_components=3) -> np.ndarray`, `quantile_discretize(data, n_bins=6) -> np.ndarray`, `discrete_mutual_information(a, b) -> float`, `mutual_information_xz(x_rates, z_rates) -> float`, `mutual_information_zy(z_rates, labels) -> float`. Task 8 calls `mutual_information_xz`/`mutual_information_zy` (the other three are internal helpers, but exported for direct testing).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_capacity_metrics.py` (before the `if __name__ == "__main__":`
block), and update the import line at the top to also pull in the new names:

```python
from learning.capacity_metrics import (
    firing_rate_matrix, participation_ratio, spike_entropy,
    pca_reduce, quantile_discretize, discrete_mutual_information,
    mutual_information_xz, mutual_information_zy,
)
```

```python
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
```

Add all 6 new test functions to the `suite.run([...])` list.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_capacity_metrics.py`
Expected: FAIL with `ImportError` (the new names don't exist in `capacity_metrics.py`
yet).

- [ ] **Step 3: Write minimal implementation**

Append to `learning/capacity_metrics.py`:

```python
def pca_reduce(data: np.ndarray, n_components: int = 3) -> np.ndarray:
    """[B, N] -> [B, k]: project onto the top k principal components.

    Same Gram-matrix trick as participation_ratio() -- B x B eigendecomposition
    instead of N x N, which matters when N (neurons) far exceeds B (a diagnostic
    batch's sample count). k is capped at B - 1 (or B if that would be non-positive)
    since a sample set of size B cannot support more than that many meaningful
    components.
    """
    b = data.shape[0]
    k = max(1, min(n_components, b - 1 if b > 1 else 1))
    centered = data - data.mean(axis=0, keepdims=True)
    gram = centered @ centered.T  # [B, B]
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    # eigh ascends; take the top k.
    order = np.argsort(eigenvalues)[::-1][:k]
    top_vals = np.clip(eigenvalues[order], 1e-12, None)
    top_vecs = eigenvectors[:, order]
    # Recover the N-dimensional principal directions' projection via the dual trick:
    # centered.T @ top_vecs gives directions in N-space; scaling by 1/sqrt(eigenvalue)
    # normalizes them, then projecting `centered` back onto them gives the [B, k] score.
    components = centered.T @ top_vecs / np.sqrt(top_vals)  # [N, k]
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

    The one estimator both mutual_information_xz and mutual_information_zy call, on
    already-discretized inputs. A finite-sample estimate: independent variables give a
    small positive number, not exactly zero (see the test suite's bounds checks).
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


def mutual_information_xz(x_rates: np.ndarray, z_rates: np.ndarray) -> float:
    """x_rates, z_rates: both already reduced to [B, N] via firing_rate_matrix() --
    the raw input goes through the same [T,B,...] -> [B,N] rate reduction as any
    hooked layer's spikes before it gets here. PCA-reduces both sides, discretizes
    both, then discrete_mutual_information."""
    x_symbols = quantile_discretize(pca_reduce(x_rates))
    z_symbols = quantile_discretize(pca_reduce(z_rates))
    return discrete_mutual_information(x_symbols, z_symbols)


def mutual_information_zy(z_rates: np.ndarray, labels: np.ndarray) -> float:
    """z_rates: [B, N]. labels: [B] integer class labels, already discrete -- no
    reduction needed on that side. PCA-reduces and discretizes z only, then
    discrete_mutual_information against the raw labels."""
    z_symbols = quantile_discretize(pca_reduce(z_rates))
    return discrete_mutual_information(z_symbols, np.asarray(labels))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_capacity_metrics.py`
Expected: PASS, all 12 tests.

- [ ] **Step 5: Commit**

```bash
git add learning/capacity_metrics.py tests/unit_capacity_metrics.py
git commit -m "feat: add PCA reduction, discretization, and mutual information estimators"
```

---

### Task 6: Config flag `compute_capacity_metrics`

**Files:**
- Modify: `configuration/SNN_module.yaml` (add the key under `training:`)
- Modify: `skeleton/snn_config.py` (read it into `Settings`)
- Modify: `experiments/ex6/config.yaml` (turn it on for the scalability study)

**Interfaces:**
- Consumes: nothing.
- Produces: `cfg.COMPUTE_CAPACITY_METRICS: bool`, read by Task 7 and Task 8.

- [ ] **Step 1: Write the failing test**

There isn't an existing config-flag test file to extend cheaply for a single boolean,
and `skeleton/snn_config.py`'s existing flags aren't individually unit-tested (they're
exercised indirectly via `tests/_harness.py`'s `fresh_cfg()` in every other suite) — so
this task's test is folded into Task 8's wiring test instead of a standalone one here,
per the plan's task-right-sizing rule (a config flag with no behavior yet is not an
independently meaningful test). Proceed straight to implementation; Task 8's test will
fail loudly if this flag doesn't exist or doesn't read correctly.

- [ ] **Step 2: Add the key to the base YAML**

In `configuration/SNN_module.yaml`, add this line directly after `use_amp` (after line
83, before `grad_accum_steps`):

```yaml
  use_amp: false                # true | false. Changes speed AND numerics; state it explicitly.
  compute_capacity_metrics: false  # scalability-study-only diagnostics: Participation Ratio,
                                    # mutual information, spike entropy, gradient norms. Off by
                                    # default -- costs extra CPU-side computation once per run.
                                    # See docs/superpowers/specs/2026-09-04-capacity-metrics-design.md
  grad_accum_steps: 1           # batches accumulated before an optimizer step
```

- [ ] **Step 3: Read it into `Settings`**

In `skeleton/snn_config.py`, add this line directly after `self.USE_AMP` (after line
117):

```python
        self.USE_AMP                  = training.require_bool("use_amp")
        # Scalability-study-only diagnostics (Participation Ratio, mutual information,
        # spike entropy, per-layer gradient norms). Off by default: every other
        # experiment's runs.csv/layers.csv output is unaffected either way.
        self.COMPUTE_CAPACITY_METRICS = training.require_bool("compute_capacity_metrics")
```

- [ ] **Step 4: Turn it on for ex6**

In `experiments/ex6/config.yaml`, add under the existing `training:` block (near
`lr_scheduler: none`):

```yaml
  compute_capacity_metrics: true   # this study's whole point is these metrics
```

- [ ] **Step 5: Verify nothing broke**

Run: `python tests/run_all.py`
Expected: every suite still PASSES — a required key with a stated default in the base
YAML cannot break any config that doesn't override it, but this confirms `Settings()`
still constructs cleanly everywhere `fresh_cfg()`/`Settings()` is called.

- [ ] **Step 6: Commit**

```bash
git add configuration/SNN_module.yaml skeleton/snn_config.py experiments/ex6/config.yaml
git commit -m "feat: add compute_capacity_metrics config flag, on for ex6"
```

---

### Task 7: Deferred-sync gradient-norm tracking

**Files:**
- Modify: `learning/training.py` (`SNNTrainer.__init__`, a new `record_gradient_norms` method, the call site inside `train()`, and the end-of-`train()` sync)
- Test: `tests/unit_training_metrics.py` (append)

**Interfaces:**
- Consumes: `SpikingNet.dense_before()` (Task 3), `cfg.COMPUTE_CAPACITY_METRICS` (Task 6).
- Produces: `SNNTrainer.grad_norm_means: dict[str, float]`, populated once at the end of
  `train()`. Task 10 reads this attribute.

**Design constraint this task must follow:** `learning/training.py` repeatedly and
deliberately avoids `.item()`/`.cpu()` calls inside the per-iteration batch loop (see
`train()`'s own docstring and the `loss_hist_gpu`/`acc_hist_gpu` pattern) because it
forces a CUDA sync every iteration. Gradient norms must accumulate as GPU-resident
tensors and sync exactly once, at the very end of the run — not per iteration, not per
epoch.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_training_metrics.py` (before its own test-runner block; check the
bottom of the file for the exact `suite.run([...])` call and add to that list):

```python
def _capacity_test_cfg(compute_capacity_metrics: bool):
    """A CPU-buildable Settings, via _harness.fresh_cfg() (which already calls
    apply_dataset_shape() -- required before any model can be built, and NOT done by
    settings_pair(), which only merges the real YAML files with no dataset chosen).
    Training fields set directly on the object rather than through YAML overlay, since
    these tests build a real model and need every shape-dependent field resolved."""
    cfg = fresh_cfg()
    cfg.COMPUTE_CAPACITY_METRICS = compute_capacity_metrics
    cfg.EPOCHS = 1
    cfg.ITERA = 2
    cfg.BATCH_SIZE = 3
    cfg.CALIBRATE_BATCH_SIZE = False
    cfg.WARMUP_ITERATIONS = 0
    cfg.USE_AMP = False
    cfg.GRAD_ACCUM_STEPS = 1
    cfg.LR_SCHEDULER = "none"
    return cfg


def test_gradient_norms_recorded_when_flag_is_on() -> None:
    """With compute_capacity_metrics on, a short training run must populate
    grad_norm_means for every named LIF slot, with non-negative float values."""
    from learning.training import SNNTrainer
    import torch

    cfg = _capacity_test_cfg(compute_capacity_metrics=True)
    model, _ = build_model("sj", cfg)
    inputs = spike_input(cfg, time_steps=4, batch=3)
    targets = torch.randint(0, cfg.NUM_CLASSES, (3,))
    loader = [(inputs, targets), (inputs, targets)]

    trainer = SNNTrainer(model, loader, cfg, torch.device("cpu"))
    trainer.train(csv_path=str(REPO_TMP / "training_results.csv"))

    means = trainer.grad_norm_means
    suite.check("grad_norm_means is populated", len(means) > 0, f"got {means}")
    suite.check("every value is a non-negative float",
                all(isinstance(v, float) and v >= 0.0 for v in means.values()),
                f"means={means}")
    suite.check("lif_out has a recorded gradient norm (the naming/hooking fix)",
                "lif_out" in means, f"keys={list(means.keys())}")


def test_gradient_norms_empty_when_flag_is_off() -> None:
    """Off by default: no gradient-norm tracking happens, no cost, empty dict."""
    from learning.training import SNNTrainer
    import torch

    cfg = _capacity_test_cfg(compute_capacity_metrics=False)
    model, _ = build_model("sj", cfg)
    inputs = spike_input(cfg, time_steps=4, batch=3)
    targets = torch.randint(0, cfg.NUM_CLASSES, (3,))
    loader = [(inputs, targets), (inputs, targets)]

    trainer = SNNTrainer(model, loader, cfg, torch.device("cpu"))
    trainer.train(csv_path=str(REPO_TMP / "training_results_off.csv"))
    suite.check("grad_norm_means stays empty when the flag is off",
                trainer.grad_norm_means == {})
```

Add both to the file's `suite.run([...])` list. (`fresh_cfg`, `build_model`,
`spike_input`, `REPO_TMP` are already imported/defined at the top of this file — see
its existing `test_window_is_none_when_not_knowable` for the file's general pattern;
`fresh_cfg`/`build_model` specifically come from `_harness`, same as every adapter
test.) `build_model("sj", cfg)` builds through the same shim class `learning/main.py`
uses, rather than importing `SNN_SJ` directly — matching how every other suite in this
project constructs a model.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_training_metrics.py`
Expected: FAIL with `AttributeError: 'SNNTrainer' object has no attribute
'grad_norm_means'`.

- [ ] **Step 3: Write minimal implementation**

In `learning/training.py`, in `SNNTrainer.__init__` (after line 116, near
`self.dense_macs_per_layer = {}`):

```python
        self.dense_macs_per_layer = {}
        # Gradient-norm tracking (opt-in, see cfg.COMPUTE_CAPACITY_METRICS): GPU-resident
        # running sums, synced ONCE at the end of train() -- never per-iteration, per
        # this file's deferred-sync rule (see train()'s own docstring).
        self.compute_capacity_metrics = getattr(cfg, "COMPUTE_CAPACITY_METRICS", False)
        self.grad_norm_sums: dict[str, torch.Tensor] = {}
        self.grad_norm_steps: int = 0
        self.grad_norm_means: dict[str, float] = {}
        self.last_capacity_metrics: dict = {}
```

Add a new method to `SNNTrainer`, directly after `forward_pass` (after line 137):

```python
    def record_gradient_norms(self) -> None:
        """Accumulate this optimizer step's per-layer gradient norm as a GPU-resident
        running sum. Called only when self.compute_capacity_metrics is on, and only
        when do_step is True (a full effective batch's gradient is complete, not a
        partial accumulation step). No .item() here -- see finalize_gradient_norms()
        for the one sync, at the end of the whole run.
        """
        net = self.model.net
        for name in net.named_lif_layers():
            layer = net.dense_before(name)
            if layer is None or layer.weight.grad is None:
                continue
            norm = layer.weight.grad.detach().norm(2)
            if name in self.grad_norm_sums:
                self.grad_norm_sums[name] = self.grad_norm_sums[name] + norm
            else:
                self.grad_norm_sums[name] = norm
        self.grad_norm_steps += 1

    def finalize_gradient_norms(self) -> None:
        """The one sync point for gradient norms, called once after the whole training
        run finishes -- mirrors how loss_hist_gpu/acc_hist_gpu are read back once per
        epoch rather than per iteration, just at run granularity since layers.csv wants
        one mean per layer per run, not one per epoch."""
        if self.grad_norm_steps == 0:
            return
        self.grad_norm_means = {
            name: float((total / self.grad_norm_steps).item())
            for name, total in self.grad_norm_sums.items()
        }
```

In the training loop inside `train()`, right after the existing
`self.model.backward_pass(loss_val, scaler=self.scaler, do_step=do_step)` call (line
428) and its `measure_mem` block, add the gradient-norm capture, guarded the same way
`measure_mem` blocks are — but on `do_step`, not `i == 0`:

```python
                do_step = ((step_count + 1) % accum == 0)
                with self.timed(self.bwd_events):
                    self.model.backward_pass(loss_val, scaler=self.scaler, do_step=do_step)
                if measure_mem:
                    mem_breakdown["backward_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                    mem_breakdown["weights_gb"] = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 ** 3)
                    mem_breakdown["gradients_gb"] = sum(p.grad.numel() * p.grad.element_size() for p in self.model.parameters() if p.grad is not None) / (1024 ** 3)
                if do_step and self.compute_capacity_metrics:
                    self.record_gradient_norms()
                if do_step:
                    self.model.zero_grad()
```

And at the very end of `train()`, right before `self.write_csv(csv_path)` (line 583),
add the one-time sync:

```python
        self.finalize_epoch_reports(raw_epoch_records, epochs, timesteps, window_s)
        self.finalize_gradient_norms()
```

(This replaces the single existing line `self.finalize_epoch_reports(...)` with two
lines — the first is unchanged, the second is new.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_training_metrics.py`
Expected: PASS, all tests including the 2 new ones.

- [ ] **Step 5: Commit**

```bash
git add learning/training.py tests/unit_training_metrics.py
git commit -m "feat: track per-layer gradient norms, deferred-sync, opt-in"
```

---

### Task 8: Wire capacity metrics into `measure_activity()`

**Files:**
- Modify: `learning/training.py` (`train()`'s probe-data capture, `measure_activity()`'s
  signature and body, and the call site)
- Test: `tests/unit_training_metrics.py` (append)

**Interfaces:**
- Consumes: `learning.capacity_metrics.firing_rate_matrix`,
  `.participation_ratio`, `.spike_entropy`, `.mutual_information_xz`,
  `.mutual_information_zy` (Tasks 4-5); `cfg.COMPUTE_CAPACITY_METRICS` (Task 6).
- Produces: `SNNTrainer.last_capacity_metrics: dict[str, dict[str, float]]` — outer key
  is the LIF slot name, inner dict has `participation_ratio`, `spike_entropy`,
  `mutual_info_xz`, `mutual_info_zy`. Task 10 reads this attribute (already declared as
  an empty dict in Task 7's `__init__` change).

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_training_metrics.py`:

```python
def test_capacity_metrics_populated_on_final_epoch_only() -> None:
    """With the flag on and 2 epochs, last_capacity_metrics must be populated after
    train() returns, with one entry per hooked LIF layer including lif_out."""
    from learning.training import SNNTrainer
    import torch

    cfg = _capacity_test_cfg(compute_capacity_metrics=True)
    cfg.EPOCHS = 2
    cfg.BATCH_SIZE = 4
    model, _ = build_model("sj", cfg)
    inputs = spike_input(cfg, time_steps=4, batch=4)
    targets = torch.randint(0, cfg.NUM_CLASSES, (4,))
    loader = [(inputs, targets), (inputs, targets)]

    trainer = SNNTrainer(model, loader, cfg, torch.device("cpu"))
    trainer.train(csv_path=str(REPO_TMP / "training_results_capacity.csv"))

    capacity = trainer.last_capacity_metrics
    suite.check("capacity metrics populated for every hooked layer",
                set(capacity.keys()) == set(model.net.named_lif_layers().keys()),
                f"got keys={list(capacity.keys())}")
    for name, values in capacity.items():
        suite.check(f"{name}: has all 4 capacity fields",
                    {"participation_ratio", "spike_entropy",
                     "mutual_info_xz", "mutual_info_zy"} <= set(values.keys()),
                    f"{name} -> {values}")
        suite.check(f"{name}: participation_ratio is non-negative",
                    values["participation_ratio"] >= 0.0)


def test_capacity_metrics_empty_when_flag_is_off() -> None:
    from learning.training import SNNTrainer
    import torch

    cfg = _capacity_test_cfg(compute_capacity_metrics=False)
    cfg.BATCH_SIZE = 4
    model, _ = build_model("sj", cfg)
    inputs = spike_input(cfg, time_steps=4, batch=4)
    targets = torch.randint(0, cfg.NUM_CLASSES, (4,))
    loader = [(inputs, targets), (inputs, targets)]

    trainer = SNNTrainer(model, loader, cfg, torch.device("cpu"))
    trainer.train(csv_path=str(REPO_TMP / "training_results_capacity_off.csv"))
    suite.check("last_capacity_metrics stays empty when the flag is off",
                trainer.last_capacity_metrics == {})
```

Add both to the `suite.run([...])` list. (`_capacity_test_cfg` is the helper Task 7
defines, above — reused here rather than duplicated.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_training_metrics.py`
Expected: FAIL — `last_capacity_metrics` stays `{}` even with the flag on, since
nothing populates it yet.

- [ ] **Step 3: Write minimal implementation**

In `learning/training.py`, change the probe-data capture in `train()` (line 308) to
also keep the targets:

```python
        probe_data, probe_targets = next(iter(self.train_loader))
        timesteps = probe_data.shape[0]  # loader yields [T, B, C, H, W] — the real BPTT unroll length, not a config value
        self.timesteps, self.window_s = timesteps, window_s
```

Change `measure_activity`'s signature and body (lines 139-178) to accept the targets
and epoch-final flag, and compute capacity metrics when asked:

```python
    def measure_activity(self, probe_data: torch.Tensor, timesteps: int,
                          probe_targets: torch.Tensor | None = None,
                          compute_capacity: bool = False) -> dict:
        """One UNTIMED forward pass with spike recording ON, for the epoch's
        activity metrics: SynOps, CV_ISI and the sparse-vs-dense buffer report.
        ...
        (existing docstring unchanged above this point)

        compute_capacity, when True, additionally computes Participation Ratio,
        spike entropy, and both mutual-information variants for every hooked layer,
        using this same probe pass -- no separate diagnostic pass, no extra forward
        call. Only ever True on the FINAL epoch's call (see train()): the numpy-side
        work is redundant on every other epoch since only the last value is kept,
        exactly like last_activity_snapshot itself only keeps the last epoch's copy.
        """
        self.model.activity.resume()
        self.model.activity.clear()
        try:
            with torch.no_grad():
                self.forward_pass(probe_data)
            recordings = self.model.activity.recordings()

            synops = torch.zeros((), device=self.device)
            for name, macs in self.dense_macs_per_layer.items():
                buf = self.model.activity.buffers.get(name)
                rate_t = buf.firing_rate_tensor() if buf is not None else None
                if rate_t is not None:
                    synops = synops + rate_t * macs * timesteps
            snapshot = {k: (v.cpu() if v is not None else None) for k, v in recordings.items()}

            capacity: dict[str, dict[str, float]] = {}
            if compute_capacity and probe_targets is not None:
                from learning.capacity_metrics import (
                    firing_rate_matrix, participation_ratio, spike_entropy,
                    mutual_information_xz, mutual_information_zy,
                )
                x_rates = firing_rate_matrix(probe_data)
                labels = probe_targets.detach().cpu().numpy()
                for name, recorded in recordings.items():
                    if recorded is None:
                        continue
                    z_rates = firing_rate_matrix(recorded)
                    capacity[name] = {
                        "participation_ratio": participation_ratio(z_rates),
                        "spike_entropy": spike_entropy(z_rates),
                        "mutual_info_xz": mutual_information_xz(x_rates, z_rates),
                        "mutual_info_zy": mutual_information_zy(z_rates, labels),
                    }
        finally:
            # Straight back off before the next epoch's timed loop starts.
            self.model.activity.pause()
            self.model.activity.clear()
        return {"synops": float(synops.item()), "snapshot": snapshot, "capacity": capacity}
```

Change the call site inside the epoch loop (line 519) to pass the targets and the
final-epoch flag, and capture the capacity result:

```python
            # ---- untimed activity pass ------------------------------------------
            # After the epoch's timer has stopped and after the energy window has
            # closed, so nothing measured above is affected by the recording hooks.
            activity = self.measure_activity(
                probe_data, timesteps, probe_targets,
                compute_capacity=(self.compute_capacity_metrics and epoch == epochs - 1),
            )
```

And where `activity["synops"]`/`activity["snapshot"]` are read out of
`raw_epoch_records` in `finalize_epoch_reports` — add capacity to the same record dict
at the append site (line 542-565):

```python
            raw_epoch_records.append({
                "epoch":              epoch + 1,
                "n":                  n,
                "fwd_latencies_ms":   fwd_latencies_ms,
                "bwd_latencies_ms":   bwd_latencies_ms,
                "epoch_loss_sum":     epoch_loss_sum.item(),
                "epoch_acc_sum":      epoch_acc_sum.item(),
                "epoch_spike_sum":    epoch_spike_sum.item(),
                "epoch_synops":       activity["synops"],
                "activity_snapshot":  activity["snapshot"],
                "capacity_metrics":   activity["capacity"],
                "epoch_duration":     epoch_duration,
                "gpu":                gpu,
                "gpu_diag":           gpu_diag,
                "energy_j":           energy_j,
                "dynamic_energy_j":   dynamic_energy_j,
                "avg_power_w":        avg_power_w,
                "dynamic_power_w":    dynamic_power_w,
                "idle_power_w":       idle_power_w,
                "gpu_active_s":       gpu_active_s,
                "current_lr":         self.model.get_lr(),
                "mem_breakdown":      mem_breakdown,
            })
```

And in `finalize_epoch_reports`, right next to the existing
`self.last_activity_snapshot = record["activity_snapshot"]` (line 631), add:

```python
            self.last_activity_snapshot = record["activity_snapshot"]
            self.last_capacity_metrics = record["capacity_metrics"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_training_metrics.py`
Expected: PASS, all tests including the 2 new ones from this task and the 2 from Task 7.

Also run the full suite to confirm nothing regressed:

Run: `python tests/run_all.py`
Expected: all suites PASS.

- [ ] **Step 5: Commit**

```bash
git add learning/training.py tests/unit_training_metrics.py
git commit -m "feat: compute capacity metrics on the final epoch's existing diagnostic pass"
```

---

### Task 9: Schema bump — 5 new `layers.csv` columns

**Files:**
- Modify: `skeleton/results.py` (`SCHEMA_VERSION`, `LAYER_COLUMNS`)
- Test: `tests/unit_results_schema.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: `LAYER_COLUMNS` includes `grad_norm_mean`, `participation_ratio`,
  `spike_entropy`, `mutual_info_xz`, `mutual_info_zy`. Task 10 populates them.

- [ ] **Step 1: Write the failing test**

Create `tests/unit_results_schema.py`:

```python
"""Unit tests for the layers.csv schema bump (v2 -> v3) that adds the 5 capacity-
metric columns.

    python tests/unit_results_schema.py

CPU-only, no dataset, no download, no model -- pure schema and CSV-writer checks.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from _harness import Suite
from skeleton.results import LAYER_COLUMNS, SCHEMA_VERSION, append_row

suite = Suite("unit_results_schema")

NEW_COLUMNS = [
    "grad_norm_mean", "participation_ratio", "spike_entropy",
    "mutual_info_xz", "mutual_info_zy",
]


def test_schema_version_bumped() -> None:
    suite.check("SCHEMA_VERSION is 3", SCHEMA_VERSION == 3, f"got {SCHEMA_VERSION}")


def test_layer_columns_include_all_five_new_fields() -> None:
    missing = [c for c in NEW_COLUMNS if c not in LAYER_COLUMNS]
    suite.check("all 5 new columns are present", not missing, f"missing={missing}")


def test_layer_row_with_new_columns_writes_and_leaves_empty_when_none() -> None:
    """A row that doesn't set the new columns must still write, with empty cells --
    the existing 'empty cell, not a shifted header' guarantee."""
    tmp = Path(tempfile.mkdtemp(prefix="snn_unit_")) / "layers.csv"
    row_with_values = {c: None for c in LAYER_COLUMNS}
    row_with_values.update({
        "schema_version": SCHEMA_VERSION, "run_id": "test_run", "layer_index": 0,
        "layer_type": "lif1:Test", "grad_norm_mean": 0.5, "participation_ratio": 2.3,
        "spike_entropy": 1.1, "mutual_info_xz": 0.2, "mutual_info_zy": 0.3,
    })
    row_without_values = {c: None for c in LAYER_COLUMNS}
    row_without_values.update({
        "schema_version": SCHEMA_VERSION, "run_id": "test_run", "layer_index": 1,
        "layer_type": "lif2:Test",
    })
    append_row(tmp, LAYER_COLUMNS, row_with_values)
    append_row(tmp, LAYER_COLUMNS, row_without_values)
    text = tmp.read_text()
    lines = text.strip().splitlines()
    suite.check("header written once, two data rows follow", len(lines) == 3,
                f"got {len(lines)} lines")
    suite.check("second row's new columns are empty cells", ",,,," in lines[2]
                or lines[2].endswith(",,,,"), f"row2={lines[2]!r}")


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_schema_version_bumped,
        test_layer_columns_include_all_five_new_fields,
        test_layer_row_with_new_columns_writes_and_leaves_empty_when_none,
    ]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_results_schema.py`
Expected: FAIL — `SCHEMA_VERSION` is still 2, and the 5 new columns aren't in
`LAYER_COLUMNS`.

- [ ] **Step 3: Write minimal implementation**

In `skeleton/results.py`, change line 44:

```python
# v3 added 5 capacity-metric columns to LAYER_COLUMNS (grad_norm_mean,
# participation_ratio, spike_entropy, mutual_info_xz, mutual_info_zy), for the
# scalability study's opt-in diagnostics -- see
# docs/superpowers/specs/2026-09-04-capacity-metrics-design.md. Bumped rather than
# appended silently for the same reason v2 was: append_row() refuses to write into a
# file whose header differs, so a v2 layers.csv must be moved aside, not appended to.
SCHEMA_VERSION = 3
```

And change `LAYER_COLUMNS` (lines 112-115):

```python
LAYER_COLUMNS: list[str] = [
    "schema_version", "run_id", "layer_index", "layer_type",
    "neurons", "total_spikes", "opportunities", "spike_rate_pct",
    # ---- capacity metrics (v3) -- scalability-study-only, empty unless
    # training.compute_capacity_metrics is true. See skeleton/results_collect.py's
    # build_layer_rows() for how each is populated.
    "grad_norm_mean", "participation_ratio", "spike_entropy",
    "mutual_info_xz", "mutual_info_zy",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_results_schema.py`
Expected: PASS, all 3 tests.

- [ ] **Step 5: Register the suite and commit**

Add `"unit_results_schema.py"` to `tests/run_all.py`'s `SUITES` list (after
`"unit_capacity_metrics.py"`).

```bash
git add skeleton/results.py tests/unit_results_schema.py tests/run_all.py
git commit -m "feat: bump layers.csv to schema v3, add 5 capacity-metric columns"
```

---

### Task 10: Populate the new columns end to end

**Files:**
- Modify: `skeleton/results_collect.py` (`build_layer_rows`)
- Modify: `learning/main.py` (the `build_layer_rows` call site)
- Test: `tests/unit_results_schema.py` (append)

**Interfaces:**
- Consumes: `SNNTrainer.grad_norm_means` (Task 7), `SNNTrainer.last_capacity_metrics`
  (Task 8), the schema from Task 9.
- Produces: fully populated `layers.csv` rows when `compute_capacity_metrics` is on;
  unchanged (empty new cells) output when it's off. This is the final integration task
  — after this, a real run with the flag on produces real capacity-metric data.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit_results_schema.py` (before the `if __name__ == "__main__":`
block), and add the new import at the top:

```python
from skeleton.results_collect import build_layer_rows
```

```python
class _FakeLIF:
    def __init__(self, spike_slots=0):
        self.spike_slots = spike_slots  # 0 -> build_layer_rows falls to the snapshot path

    def neurons(self):
        return 4


class _FakeNet:
    def named_lif_layers(self):
        return {"lif1": _FakeLIF(), "lif_out": _FakeLIF()}


class _FakeModel:
    def __init__(self):
        self.net = _FakeNet()


def test_build_layer_rows_includes_capacity_and_grad_norm_when_provided() -> None:
    import torch as _torch
    model = _FakeModel()
    activity_snapshot = {
        "lif1": _torch.zeros(2, 3, 4),   # [T, B, N]-shaped, any nonzero values fine
        "lif_out": _torch.ones(2, 3, 4),
    }
    capacity_metrics = {
        "lif1": {"participation_ratio": 1.5, "spike_entropy": 0.9,
                 "mutual_info_xz": 0.1, "mutual_info_zy": 0.2},
        "lif_out": {"participation_ratio": 2.5, "spike_entropy": 1.9,
                    "mutual_info_xz": 0.3, "mutual_info_zy": 0.4},
    }
    grad_norm_means = {"lif1": 0.05, "lif_out": 0.02}

    rows = build_layer_rows(model, activity_snapshot, capacity_metrics, grad_norm_means)

    suite.check("one row per named layer", len(rows) == 2, f"got {len(rows)}")
    by_type = {r["layer_type"]: r for r in rows}
    lif1_row = next(r for name, r in by_type.items() if name.startswith("lif1"))
    out_row = next(r for name, r in by_type.items() if name.startswith("lif_out"))
    suite.check("lif1 row carries its participation_ratio",
                lif1_row["participation_ratio"] == 1.5, f"got {lif1_row}")
    suite.check("lif_out row carries its grad_norm_mean",
                out_row["grad_norm_mean"] == 0.02, f"got {out_row}")


def test_build_layer_rows_leaves_new_columns_none_without_capacity_data() -> None:
    """Backward compatible: calling with no capacity_metrics/grad_norm_means args
    (as every pre-existing call site does) must still work and leave the new fields
    None."""
    import torch as _torch
    model = _FakeModel()
    activity_snapshot = {"lif1": _torch.zeros(2, 3, 4), "lif_out": _torch.ones(2, 3, 4)}
    rows = build_layer_rows(model, activity_snapshot)
    suite.check("rows still produced with no capacity args", len(rows) == 2)
    suite.check("new columns are None, not KeyError",
                all(r.get("participation_ratio") is None for r in rows))
```

Add both to the `suite.run([...])` list.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_results_schema.py`
Expected: FAIL — `build_layer_rows()` doesn't accept `capacity_metrics`/
`grad_norm_means` arguments yet (`TypeError: build_layer_rows() takes from 1 to 2
positional arguments but 4 were given`).

- [ ] **Step 3: Write minimal implementation**

In `skeleton/results_collect.py`, replace `build_layer_rows` (lines 151-203):

```python
def build_layer_rows(
    model, activity_snapshot: dict | None = None,
    capacity_metrics: dict | None = None,
    grad_norm_means: dict | None = None,
) -> list[dict]:
    """One row per spiking layer.

    Two sources for the base spike-rate fields, in order of preference:

      1. BaseLIF's own spike counters, when spike counting was switched on. Covers every
         layer including lif_out, and is the same measurement SNNs_2 records.
      2. The ActivityMonitor snapshot the trainer already collects. Free -- no extra pass
         -- and now covers every hooked layer including lif_out (see
         frameworks/snn_model.py's dynamic hooking fix).

    capacity_metrics and grad_norm_means are optional, keyed by the same LIF slot
    names -- populated only when training.compute_capacity_metrics was on for this run
    (see learning/training.py's SNNTrainer.last_capacity_metrics / grad_norm_means).
    Left None for every field when either dict is absent or has no entry for a given
    layer, matching this file's "unmeasured metric writes an empty cell" convention.

    Returns an empty list when neither spike-rate source is available, which simply
    means no layers.csv rows for this run rather than a failure.
    """
    capacity_metrics = capacity_metrics or {}
    grad_norm_means = grad_norm_means or {}
    rows: list[dict] = []
    named = model.net.named_lif_layers() if hasattr(model, "net") else {}

    def _capacity_fields(name: str) -> dict:
        values = capacity_metrics.get(name) or {}
        return {
            "participation_ratio": values.get("participation_ratio"),
            "spike_entropy": values.get("spike_entropy"),
            "mutual_info_xz": values.get("mutual_info_xz"),
            "mutual_info_zy": values.get("mutual_info_zy"),
            "grad_norm_mean": grad_norm_means.get(name),
        }

    for index, (name, layer) in enumerate(named.items()):
        neurons = layer.neurons() if hasattr(layer, "neurons") else 0
        slots = getattr(layer, "spike_slots", 0)
        total = getattr(layer, "spike_total", 0.0)
        if isinstance(total, torch.Tensor):
            total = float(total.item())

        if slots:  # source 1: the layer counted its own spikes
            rows.append({
                "layer_index": index,
                "layer_type": f"{name}:{type(layer).__name__}",
                "neurons": neurons,
                "total_spikes": total,
                "opportunities": slots,
                "spike_rate_pct": (total / slots) * 100.0 if slots else None,
                **_capacity_fields(name),
            })
            continue

        # source 2: the monitor's recording for this layer, if it has one
        recorded = (activity_snapshot or {}).get(name)
        if recorded is None:
            continue
        opportunities = int(recorded.numel())
        spikes = float(recorded.float().sum().item())
        per_sample = recorded.shape[2:] if recorded.dim() > 2 else ()
        neuron_count = 1
        for dim in per_sample:
            neuron_count *= int(dim)
        rows.append({
            "layer_index": index,
            "layer_type": f"{name}:{type(layer).__name__}",
            "neurons": neuron_count or None,
            "total_spikes": spikes,
            "opportunities": opportunities,
            "spike_rate_pct": (spikes / opportunities) * 100.0 if opportunities else None,
            **_capacity_fields(name),
        })
    return rows
```

In `learning/main.py`, update the `activity_snapshot` block (around line 190-194) and
the `build_layer_rows` call site (line 253):

```python
    epoch_log = list(trainer.epoch_log)
    activity_snapshot = getattr(trainer, "last_activity_snapshot", None) or {}
    capacity_metrics = getattr(trainer, "last_capacity_metrics", None) or {}
    grad_norm_means = getattr(trainer, "grad_norm_means", None) or {}
    num_workers = getattr(getattr(train_loader, "loader", None), "num_workers", None)
```

```python
            layer_rows=build_layer_rows(model, activity_snapshot, capacity_metrics, grad_norm_means),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_results_schema.py`
Expected: PASS, all 5 tests.

Run the full suite one more time to confirm the end-to-end change hasn't broken
anything:

Run: `python tests/run_all.py`
Expected: all suites PASS.

- [ ] **Step 5: Commit**

```bash
git add skeleton/results_collect.py learning/main.py tests/unit_results_schema.py
git commit -m "feat: populate capacity-metric columns in layers.csv end to end"
```

---

## Self-Review Notes

**Spec coverage:** §1 (naming/hooking fix) → Tasks 1-2. §2 (the metrics pass,
`dense_before`, gradient norms) → Tasks 3-5, 7-8. §3 (schema/wiring) → Tasks 9-10. §4
(testing) → a test step is embedded in every task rather than deferred to the end.
§5 (out of scope: FC hidden layers, `runs.csv` changes, ex1-ex5 opt-in) → untouched by
every task above, as intended.

**Corrections made from the original spec draft while reading the real code:** the
diagnostics pass moved from a new `learning/inference.py` addition to reusing
`learning/training.py`'s existing per-epoch `measure_activity()` (Task 8) — cheaper, no
new forward pass, and keeps every `layers.csv` column sourced from the same probe
batch. PCA/bin defaults were lowered from 5/8 to 3/6 for a batch-sized sample. Both are
reflected in the spec doc, not just this plan.

**Type/signature consistency check:** `build_layer_rows(model, activity_snapshot,
capacity_metrics, grad_norm_means)` (Task 10) matches the keys `SNNTrainer` actually
produces (`last_capacity_metrics: dict[str, dict[str, float]]` keyed by LIF slot name,
`grad_norm_means: dict[str, float]` keyed by LIF slot name — Tasks 7-8). `dense_before`
(Task 3) is called with the same slot names `named_lif_layers()` (Task 1) produces
throughout. No task introduces a name not defined by an earlier task.
