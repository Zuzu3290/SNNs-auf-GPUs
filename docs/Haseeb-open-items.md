# To DO

- different prams for different frameworks like Threshold vs v_th  in snn_torch vs norse. Implement them either in config as frameworks blocks or somre mapping , example beta has range 0-1, and tau: 10-1000 Hz even they have same meaning, found beta nad tau as clashing so far.

---

- No loss function in Norse:
Norse has no built-in loss functions — it only provides neuron dynamics (LIF, etc.) and leaves loss entirely to you. So the options are:

Option 1 — Keep SF.mse_count_loss (what we did)
It's purely tensor math operating on [T, B, num_classes] — it doesn't use any SNNTorch neuron internals. Norse produces the same shaped spike tensor, so it works fine.

Option 2 — Standard PyTorch CrossEntropy on summed spikes


loss_fn = torch.nn.CrossEntropyLoss()
# usage: loss_fn(spk_rec.sum(0), targets)  # sum over T → [B, num_classes]
For this project specifically, Option 1 is actually the right choice. You're comparing frameworks — if both models use the exact same loss function, you isolate the variable to the neuron dynamics only. Switching loss functions between SNNTorch and Norse would make the comparison unfair.

---





# Improevements in Future

- **MVSEC and TUM-VIE removed, replaced with DSEC** — both had real, separate problems:
  MVSEC had no genuine optical-flow ground truth without a from-scratch loader (tonic's
  wrapper only exposes depth/pose); TUM-VIE's mocap only bracketed the start/end of each
  recording, not a full trajectory, and its "smallest" recording still ran well past 30GB.
  DSEC (`event_data_workflow/regression_datasets.py`, `DSECRaw`) replaces both: real
  official optical-flow ground truth (`optical_flow_forward_event`/`_backward_event`),
  real disparity/depth targets. **DSEC does not provide 3D pose estimation** — if that's
  still wanted, no dataset currently in scope offers it; double-checked directly against
  DSEC's actual `target_selection` options (disparity + optical flow only).

  **Correction after actually looking at DSEC's `__init__`**: its own `split="test"` has
  *no local ground truth at all* — tonic's DSEC class raises if you request `target_selection`
  there (it's held out for DSEC's own leaderboard). So `load_raw()` uses `split="train"` (the
  18 of 40 recordings that actually have optical-flow ground truth) and splits across
  recordings ourselves — same manual approach as the other datasets, not the "real train/test
  split, no manual splitting needed" originally claimed here before this was verified.

  **Downloaded and verified for real** (single recording `thun_00_a`, events-only,
  `data_selection="events_left"`, 301MB total — 285MB events + 11.2MB flow, well under the
  ~500MB budget): events load as a real 134,133,270-row `(x,y,t,p)` array, same structured
  format as every other dataset, confirming the raw event pipeline needs zero DSEC-specific
  changes. Optical-flow target came back shaped **(41, 480, 640, 3)** — real numbers, not
  the earlier theoretical "H×W dense map" description.

  **`len(ds) == 1` for one named recording** — DSEC hands you one entire recording (134M
  events, 41 flow frames) as a single sample, not a short clip the way N-MNIST/DVS128
  Gesture do. Unusable directly. **Resolved**: `DSECRaw` now uses DSEC's own
  `optical_flow_forward_timestamps` — each flow frame ships with an exact `(start_us,
  stop_us)` window in the same absolute microsecond epoch as the event stream's own `t`
  field — to expand one recording into N `(event_sub_window, single_flow_frame)` samples,
  one per flow frame. Verified against real data: correct windowed event counts (e.g.
  804,964 events for one ~100ms window), event `t` ranges falling exactly inside the
  requested window, flow frame shape `(480, 640, 3)` (channels: flow_x, flow_y,
  valid_mask). Valid-mask checked directly — genuine binary `{0,1}`, ~19% coverage
  (LiDAR-derived ground truth, sparse by nature, not a bug).

- **All three blockers from the previous pass are now resolved, verified against real
  downloaded data (not dummy tensors), not just written as code that compiles:**
  1. **Target-extraction adapter** — `pad_events_passthrough_target` (data_pipeline.py)
     now stacks same-shaped targets into a real tensor (DSEC's per-window flow frames are
     always `(H, W, 3)`, so this always applies) instead of leaving a plain list; falls
     back to a list only if shapes ever differ (unused today, kept for safety).
     `SNNTrainer.train()` (learning/training.py) now branches on `cfg.TASK_TYPE`: skips
     `.long()` for regression (keeps float), skips TRADES entirely for regression (a
     classification-specific adversarial formulation — cross_entropy doesn't apply to a
     dense flow target), and skips the argmax-based accuracy computation (reports `acc=0.0`
     for regression rather than crashing on a shape-mismatched argmax — no real regression
     metric, e.g. flow endpoint-error, is wired yet; the loss curve is the real signal).
  2. **Which target** — settled: optical flow (`optical_flow_forward_event`), the original
     ask (real per-pixel motion, not depth/pose).
  3. **Temporal slicing + target alignment** — resolved directly in `DSECRaw` using DSEC's
     own timestamp windows (see finding above), rather than trying to force
     tonic's generic `SliceByTime`/`SlicedDataset` (event-side-only) to also track a
     matching target frame.

  **Dense output head — built**: `learning/frameworks/personal/dense_head.py`'s
  `DenseDecoder` (shared, framework-agnostic — pure `nn.ConvTranspose2d`/`Upsample`/`Conv2d`,
  only the LIF layers differ per framework) mirrors the conv backbone's two conv+pool
  stages with two deconv+upsample stages, then resizes to the exact `(SENSOR_H, SENSOR_W)`
  via `F.interpolate` (conv/pool/upsample arithmetic doesn't invert to an exact size on its
  own). Replaces the old flat-vector `REGRESSION_OUTPUT_DIM` head in all 4
  `frameworks/personal/snn_*_regression.py` files — that flat head has no consumer left
  now that MVSEC/TUM-VIE are gone (kept as a config knob and a `mse_regression` loss option
  in case a future pose-like target returns, but nothing in the registry uses it today).
  New `flow_masked_mse` loss (learning/utilities.py) computes MSE against DSEC's
  `(flow_x, flow_y, valid_mask)` layout, masked to valid pixels only.

  **Verified end-to-end against real `thun_00_a` data** (not dummy tensors): real windowed
  batch → real model forward → real masked-flow loss → real backward → confirmed actual
  parameter change after the optimizer step (Δ≈0.001, consistent with lr=0.001), for all 4
  frameworks. Shapes: model output `(T, B, 2, SENSOR_H, SENSOR_W)` (or `(B, 2, H, W)` for
  SpikingJelly, which reduces T itself), loss ≈97–98 on an untrained model against real
  flow magnitudes up to ~50px — plausible, not checked for correctness beyond "a real
  gradient exists and updates weights."

  **What's still genuinely open, not silently glossed over**: `SNNTester`/
  `AdversarialEvaluator` still assume classification-shaped results (confusion matrix,
  per-class precision/recall, argmax predictions) — meaningfully more work than the
  training-side fix above, not touched in this pass. A full `main.py` run on DSEC would
  train successfully now but fail once it reaches the test phase. Also open: no
  regression-appropriate test metric exists yet (e.g. flow endpoint-error), and only one
  recording (`thun_00_a`) has been downloaded/verified — the full `split="train"` path (18
  recordings) is wired but untested at that scale.

  docs/results/run_benchmark.py + make_plots.py already pick `REGRESSION_MODELS`
  automatically for DSEC — the training phase should now work through `run_benchmark.py`
  too, but it will still fail at the test-metrics step for the same `SNNTester` reason
  above.

  **Also open**: `DATASET_REGISTRY`'s DAVIS Camera Pose and DSEC entries carry
  `num_classes: 1` — a placeholder so `Settings.apply_dataset_shape()`/`display()` and
  the in-repo (non-regression) framework files have a value to build an output layer
  from, not a real class count. It has no bearing on the real regression path: the
  actual (gitignored) regression models use `DenseDecoder`'s per-pixel output, inferred
  from `SENSOR_H`/`SENSOR_W` directly, not from a neuron-count field. Revisit this
  placeholder once the in-repo model files gain real regression support.



# Deferred — Haseeb parameter alignment (not applied, needs a decision)

- **Norse `threshold: 1.0` + `input_scale: 10.0`** — Haseeb's config runs Norse's
  threshold at 1.0, but compensates with an explicit `input_scale: 10.0` gain factor
  before the input hits the neuron, plus a `circ` surrogate (alpha=0.5) instead of
  `atan`. This project's own prior finding (EXP-004) is that `threshold=1.0` alone
  kills Norse — dead neurons, no spikes — because Norse's `tau_mem_inv`/`dt` locked
  gain silently attenuates input relative to snnTorch/SpikingJelly. Haseeb's
  `input_scale` is exactly the compensating knob for that attenuation.
  **Not applied**: `build_norse_layer()` (learning/frameworks/snn_norse.py) has no
  `input_scale`, `reset_method`, `v_leak`, or explicit `dt` parameter today. Copying
  just the threshold value without adding the compensating gain would reintroduce the
  known dead-neuron bug. Adding `input_scale` support is new code, not a config copy,
  and wasn't done in this pass — flagging so it isn't silently lost.

- **`n_time_bins: 20` vs our default `16`** — Haseeb bins N-MNIST into 20 timesteps;
  this project defaults to 16. Lower-risk than the Norse item above (just a config
  value, not a semantics change), but not applied — would shift every framework's
  timestep count simultaneously, which changes wall-clock/energy comparisons, so it's
  left as an explicit choice for later rather than snuck into this pass.

---

# Found Issue --- Status

## Firing-rate Hz read a config key that doesn't exist — silently ignored `temporal_slice_duration`
**Status: fixed.** `SNNTrainer.train()` (`learning/training.py`) and `SNNTester.run()`
(`learning/inference.py`) computed `window_s` from
`getattr(self.cfg, 'TEMPORAL_SLICE_DURATION_US', 15000)` — but `Settings` only ever sets
`cfg.TEMPORAL_SLICE_DURATION` (no `_US` suffix; already in microseconds, the same value
`data_pipeline.py` divides by 1000 to get ms). The `getattr` fallback (15000) happens to
match the shipped YAML default, so this was invisible until
`architecture.temporal_slice_duration` was ever changed away from 15000 — at which point
every `firing_rate_hz` figure in both the per-epoch training report and the test summary
would have silently kept using the stale default instead of the real value.
**Fixed**: both call sites now read `cfg.TEMPORAL_SLICE_DURATION`.

## `avg_spike_rate` denominator omitted batch size
**Status: fixed.** `SNNTester.run()` (`learning/inference.py`) computed the test-run-wide
`avg_spike_rate` as `total_spikes / (len(self.batch_log) * timesteps_cfg * self.num_classes)`.
`total_spikes` sums spikes across every sample in every batch, but that denominator only
counted one sample's worth of neuron-timestep slots per batch (missing `x B`). Result: the
summary-level `avg_spike_rate` / `avg_firing_rate_hz` was inflated by roughly the average
batch size, while the per-batch `spike_rate` printed for each individual batch
(`possible_spikes = T * B * num_classes`) was already correct — the bug was only in the
run-wide rollup.
**Fixed**: a `total_possible_spikes` accumulator now sums each batch's real
`possible_spikes` inside the loop, and `avg_spike_rate = total_spikes / total_possible_spikes`
— correct regardless of whether batch size is constant across the run (the test loader has
no `drop_last`, so the final batch legitimately can differ in size from the rest).

## Dataset-name mismatch silently defaulted to N-MNIST with no log trace
**Status: fixed.** `resolve_dataset_entry()` (`event_data_workflow/dataset_registry.py`):
when `cfg.DATASET_NAME` matched nothing in `DATASET_REGISTRY` *and* stdin wasn't a TTY
(any script/CI/notebook run), it fell straight through to `DATASET_REGISTRY["1"]`
(N-MNIST) with zero logging — `logger.warning` only fired on the interactive
invalid-choice path. `SNN_module.yaml` shipped with `dataset_name: MNIST`, which never
matches `"N-MNIST"` (exact-uppercase match only), so every non-interactive run on the
default config silently trained N-MNIST with no indication the requested name never
resolved.
**Fixed**: the non-interactive fallback now logs a warning naming the exact
`DATASET_NAME` that failed to match before defaulting. Also removed the now-redundant
`dataset:` block from `SNN_module.yaml` — `DATASET_REGISTRY` is the single source of
truth for which datasets exist; the YAML-level name was never anything more than an
initial lookup key into it, and the registry's own interactive-select/N-MNIST-default
behavior in `resolve_dataset_entry()` covers the case where no name is configured at all.