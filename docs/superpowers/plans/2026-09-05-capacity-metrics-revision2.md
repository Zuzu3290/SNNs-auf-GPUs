# Capacity Metrics Revision 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the study designer's corrections (`docs/superpowers/specs/2026-09-04-capacity-metrics-design.md` §6) to the capacity-metrics feature built in the prior plan: accumulate PR/entropy/MI over the full test set instead of one training probe batch, measure per-channel instead of per-pixel-site, report normalized alongside raw values, and drop the I(X;Z) metric.

**Architecture:** Move the capacity-metrics computation from `learning/training.py`'s per-epoch training probe into `learning/inference.py`'s `SNNTester.run()`, which already iterates the whole test set once per run. Add a channel-pooling reduction and normalized-metric functions to `learning/capacity_metrics.py`. Remove `mutual_information_xz`. Bump the results schema v3→v4 to match.

**Tech Stack:** Python, PyTorch, plain `numpy` (unchanged from the prior plan — no new dependency).

## Global Constraints

- No new dependency.
- Capacity metrics remain opt-in (`training.compute_capacity_metrics`, default `false`) — this revision doesn't change that gate, only where and how the computation happens.
- Gradient-norm tracking (entirely in `learning/training.py`, unrelated to this revision) is untouched — do not modify `record_gradient_norms`, `finalize_gradient_norms`, or their call sites.
- `SpikingNet.dense_before`/`named_lif_layers` (from the prior plan) are untouched.
- No new dependency; PCA/binning/MI stay plain-numpy.
- Full design reference for every change below: `docs/superpowers/specs/2026-09-04-capacity-metrics-design.md` §6.

---

### Task 1: `channel_rate_matrix` and normalized-metric functions

**Files:**
- Modify: `learning/capacity_metrics.py`
- Modify: `tests/unit_capacity_metrics.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `channel_rate_matrix(spikes: torch.Tensor) -> np.ndarray`,
  `participation_ratio_normalized(rates: np.ndarray) -> float`,
  `spike_entropy_normalized(rates: np.ndarray) -> float`. Task 3 calls all three.
  `mutual_information_xz` and its test are REMOVED in this task.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit_capacity_metrics.py` (update the import line at top to include the
new names and drop `mutual_information_xz`):

```python
from learning.capacity_metrics import (
    firing_rate_matrix, channel_rate_matrix, participation_ratio,
    participation_ratio_normalized, spike_entropy, spike_entropy_normalized,
    pca_reduce, quantile_discretize, discrete_mutual_information,
    mutual_information_zy,
)
```

```python
def test_channel_rate_matrix_pools_spatial_dims_per_channel() -> None:
    """[T, B, C, H, W] -> [B, C], averaging over T AND over H, W per channel --
    NOT flattening every (channel, y, x) site the way firing_rate_matrix() does."""
    # T=2, B=3, C=2, H=2, W=2. Channel 0 is all 1s, channel 1 is all 0s.
    spikes = torch.zeros(2, 3, 2, 2, 2)
    spikes[:, :, 0, :, :] = 1.0
    rates = channel_rate_matrix(spikes)
    suite.check("shape is [B, C], not [B, C*H*W]", rates.shape == (3, 2),
                f"got {rates.shape}")
    suite.check("channel 0 rate is 1.0 for every sample", np.allclose(rates[:, 0], 1.0))
    suite.check("channel 1 rate is 0.0 for every sample", np.allclose(rates[:, 1], 0.0))


def test_channel_rate_matrix_handles_already_flat_layers() -> None:
    """A layer with no spatial dims at all (e.g. lif_out: [T, B, C]) needs no spatial
    pooling -- output shape is unchanged from firing_rate_matrix()'s in that case."""
    spikes = torch.rand(4, 5, 10)  # [T, B, C], no H/W
    rates = channel_rate_matrix(spikes)
    suite.check("shape is [B, C] when there were no spatial dims to pool",
                rates.shape == (5, 10), f"got {rates.shape}")


def test_participation_ratio_normalized_is_pr_over_n() -> None:
    rng = np.random.default_rng(0)
    rates = rng.normal(size=(50, 10))
    pr = participation_ratio(rates)
    pr_norm = participation_ratio_normalized(rates)
    suite.check("normalized PR equals raw PR / N",
                np.isclose(pr_norm, pr / 10), f"pr={pr}, pr_norm={pr_norm}")


def test_spike_entropy_normalized_is_h_over_log2_n() -> None:
    n = 8
    rates = np.ones((10, n))  # uniform firing -> H = log2(N) exactly
    h_norm = spike_entropy_normalized(rates)
    suite.check("uniform firing gives normalized entropy of 1.0 (H == log2(N))",
                np.isclose(h_norm, 1.0, atol=1e-9), f"got {h_norm}")


def test_spike_entropy_normalized_single_channel_is_defined() -> None:
    """N=1: log2(1) = 0, a division by zero the function must guard against --
    a single channel has no distribution to spread across, so normalized entropy
    is trivially 0.0 (no room for spread) rather than raising or returning NaN."""
    rates = np.ones((10, 1))
    h_norm = spike_entropy_normalized(rates)
    suite.check("N=1 normalized entropy is 0.0, not NaN/inf",
                h_norm == 0.0, f"got {h_norm}")
```

Add all 5 to the `suite.run([...])` list; REMOVE `test_mutual_information_xz_and_zy_are_bounded`
and replace it with an MI-only test (Task 1 doesn't touch MI's own math, just removes
the `_xz` variant — do this removal now since the import line above already drops it):

```python
def test_mutual_information_zy_is_bounded() -> None:
    """mutual_information_xz is removed (see design doc §6e) -- only the zy variant
    remains. Bounds check only, since PCA+binning is a lossy approximation."""
    rng = np.random.default_rng(0)
    z_rates = rng.random((64, 50))
    labels = rng.integers(0, 10, size=64)
    mi_zy = mutual_information_zy(z_rates, labels)
    suite.check("I(Z;Y) >= 0", mi_zy >= -1e-9, f"got {mi_zy}")
    suite.check("I(Z;Y) is within the joint symbol space's entropy bound",
                mi_zy <= np.log2(6 ** 3) + 1e-6, f"got {mi_zy}")
```

Remove `test_mutual_information_xz_and_zy_are_bounded` from the `suite.run([...])` list
and add `test_mutual_information_zy_is_bounded`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_capacity_metrics.py`
Expected: FAIL — `ImportError` (the new names don't exist; `mutual_information_xz`
still does, so the import line itself would only fail on the new names, not on the
removed one, since Python imports are resolved eagerly — confirm the failure names the
new functions).

- [ ] **Step 3: Write minimal implementation**

In `learning/capacity_metrics.py`, add after `firing_rate_matrix`:

```python
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


def participation_ratio_normalized(rates: np.ndarray) -> float:
    """PR divided by the channel count N -- a size-independent reading (0 to 1-ish)
    alongside the raw value, per design doc §6d. N = rates.shape[1]."""
    n = rates.shape[1]
    if n <= 0:
        return 0.0
    return participation_ratio(rates) / n


def spike_entropy_normalized(rates: np.ndarray) -> float:
    """Entropy divided by its maximum possible value, log2(N) -- per design doc §6d.
    N=1 has no distribution to spread across (nothing to normalize against), so this
    returns 0.0 rather than dividing by log2(1)=0.
    """
    n = rates.shape[1]
    if n <= 1:
        return 0.0
    return spike_entropy(rates) / np.log2(n)
```

Remove `mutual_information_xz` entirely from `learning/capacity_metrics.py` (the whole
function and its docstring). Update the module's header docstring (lines 1-12) to drop
the "extended in a later task" language (it's finished, not pending) and mention the
channel-pooling + normalized additions.

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_capacity_metrics.py`
Expected: PASS, all tests (5 new + the replaced MI test + the untouched Task 4/5 tests
minus the removed `_xz` one).

- [ ] **Step 5: Commit**

```bash
git add learning/capacity_metrics.py tests/unit_capacity_metrics.py
git commit -m "feat: add channel-pooled rate matrix and normalized PR/entropy; drop I(X;Z)"
```

---

### Task 2: Remove the capacity-metrics pass from `learning/training.py`

**Files:**
- Modify: `learning/training.py`
- Modify: `tests/unit_training_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `measure_activity()` reverts to its pre-capacity-metrics signature and
  behavior (SynOps/CV-ISI snapshot only). `SNNTrainer.last_capacity_metrics` is removed
  — Task 3 introduces the replacement on `SNNTester` instead.
- **Gradient-norm tracking is NOT touched by this task** — `self.compute_capacity_metrics`,
  `self.grad_norm_sums`, `record_gradient_norms`, `finalize_gradient_norms`,
  `self.grad_norm_means` all stay exactly as they are.

- [ ] **Step 1: Remove the now-obsolete tests**

In `tests/unit_training_metrics.py`, remove `test_capacity_metrics_populated_on_final_epoch_only`
and `test_capacity_metrics_empty_when_flag_is_off` (both currently present, testing
behavior this task removes), and remove both from the `suite.run([...])` list. Keep
`_capacity_test_cfg` (still used by the gradient-norm tests) and both gradient-norm
tests (`test_gradient_norms_recorded_when_flag_is_on`,
`test_gradient_norms_empty_when_flag_is_off`) untouched.

- [ ] **Step 2: Run the suite to see the current (soon-to-be-obsolete) behavior**

Run: `python tests/unit_training_metrics.py`
Expected: still PASS at this point (you've only deleted tests, not changed
production code yet) — confirms you removed the right two and didn't break the
gradient-norm ones.

- [ ] **Step 3: Revert `learning/training.py`**

Revert the probe-data capture (currently `probe_data, probe_targets = next(iter(self.train_loader))`)
back to:

```python
        probe_data, _ = next(iter(self.train_loader))
```

Revert `measure_activity()`'s signature and body back to not taking `probe_targets`/
`compute_capacity`, and not computing or returning a `"capacity"` key:

```python
    def measure_activity(self, probe_data: torch.Tensor, timesteps: int) -> dict:
        """One UNTIMED forward pass with spike recording ON, for the epoch's
        activity metrics: SynOps, CV_ISI and the sparse-vs-dense buffer report.

        Separated from the timed loop deliberately. ActivityMonitor's hooks fire on
        every LIF call, and while they force no CUDA sync they do detach, take a
        threading lock and — the part that actually matters — RETAIN one spike tensor
        per layer per timestep for the whole forward. At T=16 with two hooked layers
        that is 32 tensors held live inside the region being timed, which is real
        allocator pressure and can move a timing in ways that do not reproduce.

        So recording stays paused throughout training (see train()) and happens here
        instead, once per epoch, outside any timer. The cost is one extra forward pass
        per epoch; what is bought is a timed number that measures the network and
        nothing else.

        WHAT THIS CHANGES IN THE OUTPUT: SynOps and CV_ISI are now sampled once per
        epoch on one batch, rather than accumulated per iteration. The per-iteration
        SynOps series is therefore a broadcast of the epoch's value, exactly as GPU
        energy already was -- see iteration_series().
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
        finally:
            # Straight back off before the next epoch's timed loop starts.
            self.model.activity.pause()
            self.model.activity.clear()
        return {"synops": float(synops.item()), "snapshot": snapshot}
```

(This is verbatim the pre-capacity-metrics version of the method — capacity metrics
now live in `learning/inference.py`, see Task 3.)

Revert the call site inside the epoch loop back to:

```python
            # ---- untimed activity pass ------------------------------------------
            # After the epoch's timer has stopped and after the energy window has
            # closed, so nothing measured above is affected by the recording hooks.
            activity = self.measure_activity(probe_data, timesteps)
```

Remove `"capacity_metrics": activity["capacity"],` from the `raw_epoch_records.append(...)`
block (leave every other key in that dict unchanged).

Remove `self.last_capacity_metrics: dict = {}` from `__init__`, and remove
`self.last_capacity_metrics = record["capacity_metrics"]` from `finalize_epoch_reports`
(the line directly after `self.last_activity_snapshot = record["activity_snapshot"]`
— remove only the added line, keep the original one).

- [ ] **Step 4: Run tests to verify nothing regressed**

Run: `python tests/unit_training_metrics.py`
Expected: PASS — the two gradient-norm tests, plus every pre-existing test in this
file, all still pass. (`measure_activity` reverting to fewer parameters cannot break
any caller other than the one call site you just updated.)

- [ ] **Step 5: Commit**

```bash
git add learning/training.py tests/unit_training_metrics.py
git commit -m "refactor: remove capacity-metrics pass from training.py (moves to inference.py)"
```

---

### Task 3: Accumulate capacity metrics over the full test set in `learning/inference.py`

**Files:**
- Modify: `learning/inference.py`
- Test: `tests/unit_inference_capacity.py` (new)

**Interfaces:**
- Consumes: `channel_rate_matrix`, `participation_ratio`,
  `participation_ratio_normalized`, `spike_entropy`, `spike_entropy_normalized`,
  `mutual_information_zy` (Task 1); `cfg.COMPUTE_CAPACITY_METRICS`.
- Produces: `SNNTester.capacity_metrics: dict[str, dict[str, float]]`, and the same
  dict included as `"capacity_metrics"` in `run()`'s returned dict. Task 5 reads this
  from `test_results`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit_inference_capacity.py`:

```python
"""Unit tests for SNNTester's capacity-metrics accumulation over the full test set.

    python tests/unit_inference_capacity.py

CPU-only, no real dataset -- a tiny synthetic loader stands in for the test set, per
the existing _harness.py conventions (build_model, fresh_cfg, spike_input).
"""
from __future__ import annotations

import torch

from _harness import Suite, build_model, fresh_cfg, spike_input
from learning.inference import SNNTester

suite = Suite("unit_inference_capacity")


def _capacity_test_cfg(compute_capacity_metrics: bool):
    cfg = fresh_cfg()
    cfg.COMPUTE_CAPACITY_METRICS = compute_capacity_metrics
    cfg.LATENCY_SAMPLES = 0  # skip the separate bs=1 latency pass, irrelevant here
    return cfg


def test_capacity_metrics_populated_over_the_whole_test_set() -> None:
    """With the flag on, running a multi-batch test loader must accumulate every
    batch into capacity_metrics, covering every hooked layer including lif_out --
    not just the last batch seen."""
    cfg = _capacity_test_cfg(compute_capacity_metrics=True)
    model, _ = build_model("sj", cfg)
    # 5 small batches, standing in for "however many batches calibration hands you" --
    # the accumulation must not assume one fixed batch size.
    loader = [
        (spike_input(cfg, time_steps=4, batch=4, seed=i),
         torch.randint(0, cfg.NUM_CLASSES, (4,)))
        for i in range(5)
    ]
    tester = SNNTester(model, loader, cfg, torch.device("cpu"))
    results = tester.run(csv_path="/tmp/snn_unit_inference_capacity_test.csv")

    capacity = results["capacity_metrics"]
    suite.check("capacity_metrics populated for every hooked layer",
                set(capacity.keys()) == set(model.net.named_lif_layers().keys()),
                f"got keys={list(capacity.keys())}")
    for name, values in capacity.items():
        needed = {"participation_ratio", "participation_ratio_normalized",
                  "spike_entropy", "spike_entropy_normalized", "mutual_info_zy"}
        suite.check(f"{name}: has all 5 capacity fields", needed <= set(values.keys()),
                    f"{name} -> {values}")
        suite.check(f"{name}: participation_ratio_normalized is in [0, ~1.5]",
                    0.0 <= values["participation_ratio_normalized"] <= 1.5,
                    f"got {values['participation_ratio_normalized']}")
    suite.check("tester.capacity_metrics attribute matches the returned dict",
                tester.capacity_metrics == capacity)


def test_capacity_metrics_empty_when_flag_is_off() -> None:
    cfg = _capacity_test_cfg(compute_capacity_metrics=False)
    model, _ = build_model("sj", cfg)
    loader = [
        (spike_input(cfg, time_steps=4, batch=4, seed=i),
         torch.randint(0, cfg.NUM_CLASSES, (4,)))
        for i in range(3)
    ]
    tester = SNNTester(model, loader, cfg, torch.device("cpu"))
    results = tester.run(csv_path="/tmp/snn_unit_inference_capacity_test_off.csv")
    suite.check("capacity_metrics stays empty when the flag is off",
                results["capacity_metrics"] == {})


if __name__ == "__main__":
    raise SystemExit(suite.run([
        test_capacity_metrics_populated_over_the_whole_test_set,
        test_capacity_metrics_empty_when_flag_is_off,
    ]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_inference_capacity.py`
Expected: FAIL — `KeyError: 'capacity_metrics'` (the key doesn't exist in `run()`'s
return dict yet), or `AttributeError` on `tester.capacity_metrics`.

- [ ] **Step 3: Write minimal implementation**

In `learning/inference.py`, in `SNNTester.__init__` (after the existing
`self.dense_macs_per_layer = {}`), add:

```python
        # Capacity-metrics accumulation (opt-in, see cfg.COMPUTE_CAPACITY_METRICS):
        # populated across every batch of the FULL test set in run(), per the study
        # designer's correction (design doc §6b) -- not a single training probe batch.
        self.compute_capacity_metrics = getattr(cfg, "COMPUTE_CAPACITY_METRICS", False)
        self.capacity_metrics: dict[str, dict[str, float]] = {}
```

Inside `run()`'s batch loop, right after the existing
`last_activity_snapshot = self.model.activity.recordings()` line, add accumulation
into two new local variables declared just before the loop starts (alongside the
existing `all_preds_gpu, all_targets_gpu = [], []` line):

```python
        all_preds_gpu, all_targets_gpu = [], []
        raw_batch_records: list[dict] = []
        last_activity_snapshot = {}
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        capacity_rate_chunks: dict[str, list] = {}
        capacity_label_chunks: list = []
```

and, inside the `with torch.no_grad():` loop, right after
`last_activity_snapshot = self.model.activity.recordings()`:

```python
                last_activity_snapshot = self.model.activity.recordings()
                if self.compute_capacity_metrics:
                    from learning.capacity_metrics import channel_rate_matrix
                    for name, recorded in last_activity_snapshot.items():
                        if recorded is None:
                            continue
                        capacity_rate_chunks.setdefault(name, []).append(
                            channel_rate_matrix(recorded)
                        )
                    capacity_label_chunks.append(targets.detach().cpu().numpy())
```

After the loop ends (after the `with torch.no_grad():` block, before
`t_run_elapsed = time.perf_counter() - t_run_start`), add the one-time computation
over the accumulated full-test-set arrays:

```python
        if self.compute_capacity_metrics and capacity_rate_chunks:
            from learning.capacity_metrics import (
                participation_ratio, participation_ratio_normalized,
                spike_entropy, spike_entropy_normalized, mutual_information_zy,
            )
            labels_all = np.concatenate(capacity_label_chunks)
            for name, chunks in capacity_rate_chunks.items():
                rates_all = np.concatenate(chunks, axis=0)  # [full_test_set_size, C]
                self.capacity_metrics[name] = {
                    "participation_ratio": participation_ratio(rates_all),
                    "participation_ratio_normalized": participation_ratio_normalized(rates_all),
                    "spike_entropy": spike_entropy(rates_all),
                    "spike_entropy_normalized": spike_entropy_normalized(rates_all),
                    "mutual_info_zy": mutual_information_zy(rates_all, labels_all),
                }
```

Add `"capacity_metrics": self.capacity_metrics,` to the dict `run()` returns at the
end (alongside the existing `"confusion_matrix": cm,` line).

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_inference_capacity.py`
Expected: PASS, both tests.

Also run the pre-existing inference-adjacent suites to confirm nothing regressed:

Run: `python tests/run_all.py pipeline_integration training_metrics`
Expected: all PASS.

- [ ] **Step 5: Register the new suite and commit**

Add `"unit_inference_capacity.py"` to `tests/run_all.py`'s `SUITES` list (after
`"unit_capacity_metrics.py"`).

```bash
git add learning/inference.py tests/unit_inference_capacity.py tests/run_all.py
git commit -m "feat: accumulate capacity metrics over the full test set in SNNTester"
```

---

### Task 4: Schema v3 → v4 — drop `mutual_info_xz`, add the two normalized columns

**Files:**
- Modify: `skeleton/results.py`
- Modify: `tests/unit_results_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LAYER_COLUMNS` without `mutual_info_xz`, with
  `participation_ratio_normalized` and `spike_entropy_normalized` added. Task 5
  populates them.

- [ ] **Step 1: Update the test**

In `tests/unit_results_schema.py`, update `NEW_COLUMNS` and the write/read test's
fixture rows:

```python
NEW_COLUMNS = [
    "grad_norm_mean", "participation_ratio", "participation_ratio_normalized",
    "spike_entropy", "spike_entropy_normalized", "mutual_info_zy",
]
```

Update `test_schema_version_bumped` to expect `4`:

```python
def test_schema_version_bumped() -> None:
    suite.check("SCHEMA_VERSION is 4", SCHEMA_VERSION == 4, f"got {SCHEMA_VERSION}")
```

Update the write/read test's two fixture rows (`row_with_values`) to use the new
column set instead of `mutual_info_xz`/`mutual_info_zy` (drop `mutual_info_xz`, keep
`mutual_info_zy`, add the two normalized fields):

```python
    row_with_values.update({
        "schema_version": SCHEMA_VERSION, "run_id": "test_run", "layer_index": 0,
        "layer_type": "lif1:Test", "grad_norm_mean": 0.5, "participation_ratio": 2.3,
        "participation_ratio_normalized": 0.7, "spike_entropy": 1.1,
        "spike_entropy_normalized": 0.4, "mutual_info_zy": 0.3,
    })
```

Update the two `build_layer_rows`-based tests (`test_build_layer_rows_includes_capacity_and_grad_norm_when_provided`,
`test_build_layer_rows_leaves_new_columns_none_without_capacity_data`) the same way:
their `capacity_metrics` fixtures drop `mutual_info_xz`, add
`participation_ratio_normalized`/`spike_entropy_normalized`, keep `mutual_info_zy`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python tests/unit_results_schema.py`
Expected: FAIL — `SCHEMA_VERSION` is still 3, columns don't match.

- [ ] **Step 3: Write minimal implementation**

In `skeleton/results.py`, update the `SCHEMA_VERSION` comment and value:

```python
# v4 replaced the v3 mutual_info_xz column with mutual_info_zy-only (I(X;Z) dropped
# per the study designer's correction -- unreliable at these dimensions/sample sizes,
# see docs/superpowers/specs/2026-09-04-capacity-metrics-design.md §6e) and added
# participation_ratio_normalized / spike_entropy_normalized (§6d). No v3 data exists
# yet, so this is a straight column-set edit, not a migration.
SCHEMA_VERSION = 4
```

Update `LAYER_COLUMNS`:

```python
LAYER_COLUMNS: list[str] = [
    "schema_version", "run_id", "layer_index", "layer_type",
    "neurons", "total_spikes", "opportunities", "spike_rate_pct",
    # ---- capacity metrics (v4) -- scalability-study-only, empty unless
    # training.compute_capacity_metrics is true. See skeleton/results_collect.py's
    # build_layer_rows() for how each is populated.
    "grad_norm_mean",
    "participation_ratio", "participation_ratio_normalized",
    "spike_entropy", "spike_entropy_normalized",
    "mutual_info_zy",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python tests/unit_results_schema.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skeleton/results.py tests/unit_results_schema.py
git commit -m "feat: bump layers.csv to schema v4 -- drop mutual_info_xz, add normalized PR/entropy"
```

---

### Task 5: Wire it through `results_collect.py` and `main.py`

**Files:**
- Modify: `skeleton/results_collect.py`
- Modify: `learning/main.py`
- Test: `tests/unit_results_schema.py` (already updated in Task 4 — this task makes
  `build_layer_rows()` match)

**Interfaces:**
- Consumes: schema from Task 4; `SNNTester.capacity_metrics` / `test_results["capacity_metrics"]`
  from Task 3.
- Produces: fully populated `layers.csv` rows, sourced from the tester instead of the
  trainer.

- [ ] **Step 1: Update `_capacity_fields()` in `skeleton/results_collect.py`**

Replace the `_capacity_fields` helper inside `build_layer_rows()`:

```python
    def _capacity_fields(name: str) -> dict:
        values = capacity_metrics.get(name) or {}
        return {
            "participation_ratio": values.get("participation_ratio"),
            "participation_ratio_normalized": values.get("participation_ratio_normalized"),
            "spike_entropy": values.get("spike_entropy"),
            "spike_entropy_normalized": values.get("spike_entropy_normalized"),
            "mutual_info_zy": values.get("mutual_info_zy"),
            "grad_norm_mean": grad_norm_means.get(name),
        }
```

(Same shape as before — only the dict of field names changes, matching Task 4's schema.)

- [ ] **Step 2: Run the schema test to verify it passes**

Run: `python tests/unit_results_schema.py`
Expected: PASS, all tests (Task 4 already updated the test fixtures to expect this
exact field set).

- [ ] **Step 3: Update `learning/main.py`'s wiring**

`capacity_metrics` now comes from the TESTER's result, not the trainer's attribute
(the trainer no longer produces it, per Task 2). Remove this line from where it
currently sits (alongside `activity_snapshot`, before `del trainer`):

```python
    capacity_metrics = getattr(trainer, "last_capacity_metrics", None) or {}
```

Add it back in AFTER `test_results = tester.run(...)` runs (a few lines below where it
was removed from):

```python
    visualize = select_inference_mode(args.inference)
    tester       = SNNTester(model, test_loader, cfg, device, visualize=visualize)
    test_results = tester.run(csv_path=str(run_results_dir / "test.csv"))
    capacity_metrics = test_results.get("capacity_metrics") or {}
    print("\n Testing complete!")
```

(`grad_norm_means = getattr(trainer, "grad_norm_means", None) or {}` stays exactly
where it is — that one still comes from the trainer, unaffected by this revision.)

- [ ] **Step 4: Run the full suite**

Run: `python tests/run_all.py`
Expected: all suites PASS (pre-existing `sinabs`-missing failures are expected and
unrelated).

- [ ] **Step 5: Commit**

```bash
git add skeleton/results_collect.py learning/main.py
git commit -m "feat: source capacity metrics from the full-test-set pass, not the trainer"
```

---

### Task 6: Documentation corrections

**Files:**
- Modify: `experiment_plan_final.md`
- Modify: `experiments/ex6/README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Correct the batch-size scope in `experiment_plan_final.md`**

Find §4 ("What gets locked before anything else runs — ex6"). The batch size row
currently reads as if `64`/`2` is locked for the whole study. Correct it to state: ex6's
batch size decision applies to the N-MNIST pilot only; ex7 onward (N-Caltech101)
calibrate their own batch size per `calibrate_batch_size()`'s existing occupancy
policy (30-35% VRAM), which will differ substantially by sensor size (expect single
digits on N-Caltech101, vs. 64 on N-MNIST) — this is not a confound to eliminate, it's
correct dataset-dependent behavior. Reference design doc §6a for the reasoning.

Also update the metrics table in §5 (the one recording PR/entropy build status) to
reflect: capacity metrics are now computed over the full test set (not one probe
batch), measured per-channel (not per-pixel-site), reported raw+normalized, and
`mutual_info_xz` is dropped — point at design doc §6 rather than restating it in full.

- [ ] **Step 2: Correct `experiments/ex6/README.md`**

In §12 (the Results section filled in previously), the "Batch size confirmed" line
should be reworded to make clear this batch size is confirmed for the N-MNIST pilot
specifically, not "the whole study" — since ex6 is N-MNIST-only, this is a scoping
clarification, not a reversal of ex6's own actual finding (VRAM headroom on N-MNIST's
largest planned variant is still exactly what was measured and confirmed).

- [ ] **Step 3: Commit**

```bash
git add experiment_plan_final.md experiments/ex6/README.md
git commit -m "docs: correct batch-size scope -- ex6's decision is N-MNIST-only, not study-wide"
```

## Self-Review Notes

**Spec coverage:** §6a (batch size scope) → Task 6 (docs only — no code enforces batch
size per-dataset today since ex7's configs don't exist yet; nothing to fix in code).
§6b (accumulate over full eval split) → Task 3. §6c (channel units) → Task 1. §6d (raw
+ normalized) → Tasks 1, 4, 5. §6e (drop I(X;Z)) → Tasks 1, 4, 5. §6f (schema v4) →
Task 4.

**Type/signature consistency:** `capacity_metrics` dict keys used in Task 5's
`_capacity_fields()` (`participation_ratio`, `participation_ratio_normalized`,
`spike_entropy`, `spike_entropy_normalized`, `mutual_info_zy`) match exactly what
Task 3's `SNNTester.run()` populates and what Task 1's functions return. `LAYER_COLUMNS`
(Task 4) matches the same field names. No task introduces a name another task doesn't
already define.
