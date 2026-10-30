# Case study: is a continuous-time, variable-Δt LIF worth implementing here?

## The proposal being evaluated

Replace the fixed-decay `LIFCell` in [snn_lif.py](snn_lif.py) with a
continuous-time variant: track a `last_update_time` per neuron, compute
`Δt = current_time - last_update_time` on every step, and decay the
membrane potential by `exp(-Δt/τ)` instead of a constant `beta`. The pitch
is that this makes the neuron model react to *real, irregular* elapsed
hardware/event time rather than an artificial fixed clock tick.

## Verdict: not worth implementing in this codebase, as it stands

## Why

`LIFCell.forward()` (snn_lif.py:48-54) does:

```python
self.mem = self.beta * self.mem + x
```

`beta` is a fixed scalar (`LIF_BETA = 0.9`, snn_lif.py:12), applied once per
entry in the `for step in range(data.size(0))` loop in
`SNN_LIF.forward()` (snn_lif.py:101). Swapping this for
`exp(-Δt/τ)` only changes anything if `Δt` actually varies between steps.

It doesn't, because of how `data` is built before it ever reaches this
model. `event_data_workflow/data_pipeline.py:303-311` constructs every
frame with tonic's `ToFrame`:

```python
if self.wf.FRAME_MODE == "n_time_bins":
    to_frame = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=self.wf.N_TIME_BINS)
else:
    to_frame = transforms.ToFrame(sensor_size=sensor_size, time_window=self.wf.TIME_WINDOW_US)
```

Both modes produce fixed-width bins — either a fixed count of equal-width
bins per recording, or a fixed `time_window_us` per bin — with empty bins
zero-filled rather than dropped (`FixedToFrame`, referenced at
data_pipeline.py:307, exists specifically so `ComposedTransform` doesn't
skip that zero-fill). By the time `data[step]` reaches `LIFCell.forward()`,
step index `step` corresponds to a bin of constant width. `Δt` between
consecutive steps is the same constant for every step, every sample, every
batch — it's fixed at the framing stage, not at the neuron.

`exp(-Δt/τ)` with a constant `Δt` is a constant. It would compute to the
same number every call, i.e. it reduces to today's fixed `beta`
algebraically. The proposed change would add a `last_update_time` buffer,
a subtraction, and an `exp()` call per forward pass, and produce bit-for-bit
the same decay behavior as the one-line multiply that's there now.

## What would actually make it non-trivial

The variability has to exist somewhere upstream of the neuron for a
variable-Δt decay to be anything other than a more expensive way to compute
a constant. That means removing the `ToFrame` step entirely and feeding the
network raw event tuples `(x, y, p, t)` with their native irregular
timestamps — no binning, no fixed step count. That's not a `LIFCell`
change; it's a redesign of:

- the dataset/encoder (`NeuromorphicEncoder`, data_pipeline.py) — stop
  producing `[T, B, C, H, W]` frame tensors, start producing ragged
  per-sample event streams
- batching — irregular event counts per sample can't be `torch.stack`'d
  into `[T, B, ...]` the way `SNN_LIF.forward()` expects; batching would
  need padding/masking or a different collation strategy entirely
- the training loop's shape assumptions (`forward_pass` in
  `learning/training.py:125-133`, which permutes `[T, B, ...]` tensors)

That is a full ingestion-and-batching redesign, not a neuron swap, and it
would need to be justified by a concrete case where fixed-width binning is
itself the accuracy bottleneck — not assumed as a general improvement.

## Where this belongs if pursued

`dt_aware_lif` (sibling repo) already has fixed-vs-dt-aware LIF comparison
scripts (`compare_fixed_vs_dt_lif*.py`, `verify_equivalence*.py`) across
torch/norse/spikingjelly. Those operate on synthetic/controlled inputs
where `Δt` is set directly, which sidesteps the framing problem above and
is the right place to explore the *neuron math* in isolation. Bringing it
into this codebase's `LIFCell` only becomes meaningful once (or if) the
data pipeline stops binning events into fixed-width frames.
