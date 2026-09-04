# Capacity metrics for the scalability study — design

**Status:** approved design, ready for implementation planning.

**Why:** `experiment_plan_final.md` §5 identified 4 metrics needed before ex7 can run
that don't exist anywhere in the codebase — Mutual Information, Participation Ratio,
spike-train entropy, and per-layer gradient norms. This spec designs their
implementation. It also fixes a real gap found while designing where to store them:
`lif_out` (the network's output layer) is currently never monitored at all, and the
layer-naming logic will silently mislabel layers once ex8 adds hidden layers.

---

## 1. Always-on fix: dynamic, depth-safe layer hooking

**Current state (`frameworks/snn_model.py`):**

```python
self.activity = ActivityMonitor(
    {name: named[name] for name in ("lif1", "lif2") if name in named}
)
```

Hardcoded to exactly `lif1`/`lif2`. `lif_out` is never hooked — confirmed by
`ex6/README.md`'s own "2 rows per arm — lif1, lif2 — hooked layers only." No spike
rate, CV-ISI, or SynOps has ever been recorded for the output layer.

**Current state (`frameworks/spiking_net.py`, `named_lif_layers()`):**

```python
LIF_SLOTS = ("lif1", "lif2", "lif_out")
...
return {
    LIF_SLOTS[i] if i < len(LIF_SLOTS) else f"lif{i + 1}": layer
    for i, layer in enumerate(lifs)
}
```

Names LIF layers by **position**, not role. With exactly 3 LIF layers this
accidentally works (index 2 = "lif_out" = the actual last layer). The moment ex8 inserts
a hidden LIF layer between `lif2` and the output, that new layer takes index 2 and gets
incorrectly named `"lif_out"`, while the real output layer shifts to index 3 and gets
the anonymous fallback `"lif4"`.

**Fix:**

- `named_lif_layers()`: name the **last** element of `lif_layers()` `"lif_out"`
  unconditionally; name every layer before it `"lif1"`, `"lif2"`, ... in order. Correct
  for 3 layers today and for however many ex8 ends up adding.
- `SNNModel.__init__`: hook **every** name `named_lif_layers()` returns, not a
  hardcoded pair.

**Consequence:** every future run's `layers.csv` gains a `lif_out` row (spike rate,
CV-ISI, SynOps) it never had before. This is a data-completeness fix, not a
scalability-study-specific one — it applies to every experiment, on by default, no
config flag.

**Scope boundary:** this fix does not add configurable FC hidden layers to
`spiking_net.py` — that is separate work already noted in `experiment_plan_final.md`
§10. This fix only makes the naming/hooking safe for whenever that work lands.

## 2. The opt-in capacity-metrics pass

**Config:** `training.compute_capacity_metrics: bool = false`. Off by default; only
ex6-ex10 configs set it `true`. No other experiment's behavior changes.

**Corrected integration point (found by reading the real code, not assumed):** the
original draft of this spec proposed a new pass in `learning/inference.py`. That's
unnecessary — `learning/training.py`'s `SNNTrainer.measure_activity()` already does
almost exactly this, once per epoch, for SynOps/CV-ISI: an untimed forward pass over a
fixed `probe_data` batch (grabbed once at the start of `train()`, line ~308:
`probe_data, _ = next(iter(self.train_loader))`), with `ActivityMonitor` capturing
every hooked layer's spikes. `trainer.last_activity_snapshot` — the *last epoch's*
version of that snapshot — is what `layers.csv`'s existing columns are already built
from (`learning/main.py` line 193, `skeleton/results_collect.py`'s `build_layer_rows`).

So capacity metrics reuse the exact same pass, source, and grain as every other
per-layer column already in `layers.csv`, rather than inventing a second one from test
data:

- **X** (raw input): the same `probe_data` `[T, B, C, H, W]` already captured.
- **Z** (per-layer spikes): the same `ActivityMonitor.recordings()` already captured.
- **Y** (labels): `probe_targets`, the second element of that same `next(iter(...))`
  call — currently discarded (`probe_data, _ = ...`); change to
  `probe_data, probe_targets = next(iter(self.train_loader))` and thread it through to
  `measure_activity()`.

Computed only on the **final** epoch's call to `measure_activity()` (guarded by a new
`compute_capacity: bool` parameter, `True` only when `epoch == epochs - 1` and the
config flag is on) — the numpy-side PCA/binning/histogram work is redundant every other
epoch since only the last value is ever kept, matching how `last_activity_snapshot`
itself already only keeps the last epoch's copy.

**Sample size caveat, stated rather than solved:** this uses one batch (`cfg.BATCH_SIZE`
samples — 64 in ex6), not a separately assembled larger sample. That is the same
sample size every other `layers.csv` column already accepts for this run, and keeps
this change from touching the loader/pass logic at all. If 64 samples turns out to be
too few for a stable MI estimate once ex7 runs for real, accumulating a few batches is
a follow-up, not a blocker to landing this — YAGNI until there's a real run showing it
matters.

**No change to `learning/inference.py`.**

**New module: `learning/capacity_metrics.py`.** Pure functions operating on plain
tensors/arrays — no dataset, no GPU, no model object required, which is what makes them
unit-testable in isolation:

```python
def firing_rate_matrix(spikes: torch.Tensor) -> np.ndarray:
    """[T, B, C, H, W] (or [T, B, N]) -> [B, N] per-sample, per-neuron mean rate."""

def participation_ratio(rates: np.ndarray) -> float:
    """rates: [B, N]. PR = (sum(lambda_i))**2 / sum(lambda_i**2) via the Gram-matrix
    trick (eigenvalues of X @ X.T give the same nonzero spectrum as the N x N
    covariance, cheaply, even when N >> B)."""

def spike_entropy(rates: np.ndarray) -> float:
    """rates: [B, N]. Mean rate per neuron, normalized to a distribution over neurons,
    Shannon entropy in bits."""

def pca_reduce(data: np.ndarray, n_components: int = 3) -> np.ndarray:
    """[B, N] -> [B, k]. Same Gram-matrix trick as participation_ratio."""

def quantile_discretize(data: np.ndarray, n_bins: int = 6) -> np.ndarray:
    """[B, k] continuous -> [B] int symbols, via per-column quantile binning then
    base-n_bins encoding into one joint symbol per sample."""

def discrete_mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """Two integer-symbol arrays, same length -> empirical MI in bits, from the joint
    histogram. The one estimator both MI variants below call."""

def mutual_information_xz(x_rates: np.ndarray, z_rates: np.ndarray) -> float:
    """x_rates, z_rates: both already reduced via firing_rate_matrix() -- the raw
    input goes through the same [T,B,...] -> [B,N] rate reduction as any hooked
    layer's spikes before it gets here. PCA-reduce + discretize both sides, then
    discrete_mutual_information."""

def mutual_information_zy(z_rates: np.ndarray, labels: np.ndarray) -> float:
    """PCA-reduce + discretize z only (labels are already discrete), then
    discrete_mutual_information."""
```

`n_components=3` / `n_bins=6` (216 possible joint states) rather than the rounder 5/8
originally proposed: with a 64-sample probe batch (ex6's `batch_size`), an 8^5 state
space would be almost entirely empty cells, making the MI estimate close to noise.
Smaller defaults keep the joint histogram populated enough to mean something for a
batch-sized sample. Documented as a tuning knob, not treated as exact.

No new dependency: PCA and binning are implemented with plain `numpy` (already a direct
pin); this project treats `scipy`/`sklearn` as deliberately absent from direct imports
(see `requirements.txt`'s pinning notes), so this design doesn't introduce either.

**Gradient norms — separate integration point, in `learning/training.py`, following the
file's existing deferred-sync rule.** This file is explicit and repeated about one
thing: no `.item()`/`.cpu()` call inside the batch loop, because it forces a CUDA sync
every iteration (see `train()`'s own docstring, and the `loss_hist_gpu`/`acc_hist_gpu`
pattern). Gradient norms follow the same shape:

- A new `SpikingNet.dense_before(lif_name)` method (mirrors the existing `dense_after`,
  but walks backward): the nearest `Conv2d`/`Linear` **preceding** a given LIF slot —
  the module whose weights actually produced that layer's input.
- After `backward_pass()` returns and only when `do_step` is `True` (the accumulated
  gradient for a full effective batch is complete, not a partial micro-batch), for each
  named LIF slot, read `dense_before(name).weight.grad`, call `.norm(2)` — **stays a GPU
  tensor**, added into a running per-layer sum (`self.grad_norm_sums: dict[str,
  torch.Tensor]`), never `.item()`'d inside the loop. Increment a step counter once per
  call, not per layer.
- **One sync for the whole run**, not per epoch: at the very end of `train()`, after the
  epoch loop, divide each layer's running sum by the step count and call `.item()` once
  per layer to get `self.grad_norm_means: dict[str, float]`.

This pairing reaches `lif_out` too (paired with the classifier `Linear`) once the
always-on naming/hooking fix from §1 lands, giving the front-layer/final-layer ratio the
depth-ceiling stopping rule (`experiment_plan_final.md` §6, ex8) needs, for however many
layers the network ends up with.

## 3. Where the results land

Reusing `layers.csv` (`skeleton/results.py`'s `LAYER_COLUMNS`, `schema_version` bumped
from 2 to 3), not a new file — one row per hooked layer per run already exists there.
Five new columns, all nullable, populated only when `compute_capacity_metrics` is on:

| column | comes from |
|---|---|
| `grad_norm_mean` | training.py, every layer |
| `participation_ratio` | capacity_metrics.py, every hooked layer |
| `spike_entropy` | capacity_metrics.py, every hooked layer |
| `mutual_info_xz` | capacity_metrics.py, every hooked layer, vs. the raw input |
| `mutual_info_zy` | capacity_metrics.py, every hooked layer, vs. the labels |

`skeleton/results_collect.py` populates these the same way it already populates
everything else: read off what was actually measured, `None` if the flag was off or a
value wasn't produced — matching the existing "empty cell, not a shifted header"
convention.

**Wiring, end to end:** `SNNTrainer` gains two new attributes, populated the same way
`last_activity_snapshot` already is — `self.last_capacity_metrics: dict = {}` (set from
`measure_activity()`'s return value on the final epoch) and `self.grad_norm_means:
dict[str, float] = {}` (set once, after the epoch loop, in `train()`). In
`learning/main.py`, alongside the existing `activity_snapshot = getattr(trainer,
"last_activity_snapshot", None) or {}` (line ~193), add the same pattern for both new
attributes, and pass all three into `build_layer_rows(model, activity_snapshot,
capacity_metrics, grad_norm_means)` — extending that function's signature rather than
adding a new one, since it already owns "build one row per layer."

## 4. Testing

New file `tests/unit_capacity_metrics.py`, following the existing `Suite`-based,
CPU-only, no-dataset style (`tests/unit_training_metrics.py` is the template). Every
function is checked against a case with a hand-computable right answer before it's
trusted on real data:

- `discrete_mutual_information`: identical arrays → MI equals their entropy exactly
  (e.g. `a = b = [0,0,1,1]` → both are 1 bit); independent shuffled arrays → MI near 0.
- `participation_ratio`: one neuron carries all the variance → PR ≈ 1; all neurons
  independent and equal-variance → PR ≈ N.
- `spike_entropy`: uniform firing across N neurons → H = log2(N); one neuron fires,
  rest silent → H = 0.
- `pca_reduce` / `quantile_discretize`: shape and determinism checks (same input, same
  output), not exact values, since these feed an approximation.
- End-to-end `mutual_information_xz`/`_zy`: bounds checks (MI ≥ 0, MI ≤ min entropy of
  either side) on synthetic tensors shaped like real spike data.

New file `tests/unit_layer_naming.py` (or added to an existing adapter test) for the
always-on fix: build a network via the existing `_harness.py` `build_model()` helper,
check `named_lif_layers()` labels the last LIF `lif_out` regardless of how many LIF
layers exist — parametrize over 3 layers (today's architecture) and a synthetic 5-layer
case (simulating post-ex8), confirming the naming logic itself, not just today's
accidental case.

## 5. Out of scope

- Configurable FC hidden layers in `spiking_net.py` (ex8 prerequisite, tracked
  separately in `experiment_plan_final.md` §10).
- Any change to `runs.csv` (the run-level summary) — all 5 new values are per-layer,
  so `layers.csv` is the right grain, not `runs.csv`.
- Making capacity metrics available for framework-comparison experiments (ex1-ex5) —
  explicitly scalability-study-only per the opt-in flag.
