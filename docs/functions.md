# Function & Call-Site Design Notes

Design rationale for functions/call orderings that aren't obvious from the
function's own docstring — usually because the "why" lives at the call site
(learning/main.py) rather than inside the function itself.

## `calibrate_batch_size` (learning/utilities.py)

What it measures and why: a real forward+backward pass, on synthetic data
created directly on the GPU (`torch.rand(..., device=device)` — this never
touches host RAM, the CPU-side pipeline, or the cache controller; it's a
pure VRAM probe), at the dataset's actual sensor resolution. It reads peak
VRAM afterward and picks the largest batch size that stays within a target
fraction of total VRAM (`batch_vram_fraction`, default 34%) — not the
largest one that merely avoids OOM, so a run always has real headroom left
for whatever else needs the card.

This exists because model VRAM cost doesn't scale with batch size alone —
it scales with sensor resolution too, and a global fixed batch size that
works for a small sensor can OOM outright on a larger one. Measured
directly (see `event_data_workflow/vram_batch_scaling_task.md`): N-MNIST's
34×34 sensor trains fine at batch_size=128, but N-Caltech101's 240×180
sensor OOMs at that same batch_size — Conv1's activations alone, held
across all 16 BPTT timesteps, cost ~46x more per sample at that resolution.
Calibrating live, per dataset, is what makes one config work across sensor
sizes that differ by two orders of magnitude without a person hand-tuning
a number per dataset.

Fully automatic, no manual step: since it reads `cfg.SENSOR_H`/`SENSOR_W`
(already set by `apply_dataset_shape()` from the dataset registry entry),
adding a new dataset to the registry gets it calibrated on the very next
run — nothing to re-measure by hand.

Can be disabled: `training.calibrate_batch_size: false` in
`SNN_module.yaml` uses `training.batch_size` exactly as written, no probe
run. Off by default is not recommended for a new dataset — the fixed value
carries no guarantee it fits that dataset's sensor resolution.

Two independent, unrelated constraints bound the result, whichever binds
first wins:
- `resource_policy.batch_vram_fraction` (default 0.34) — an *efficiency*
  target: use roughly this share of VRAM, no more.
- `resource_policy.max_batch_size` (default 256) — a *learning-quality*
  ceiling, unrelated to how much VRAM is free: large-batch training trades
  gradient-update count for smoother (less noisy) gradients, which tends to
  generalize worse past a point. This caps how far that trade goes,
  regardless of card size.

A small-sensor dataset (e.g. N-MNIST's 34×34) can hit `max_batch_size`
well before `batch_vram_fraction`'s target band — the VRAM math alone
would justify a much larger batch, but the ceiling stops it short, landing
under the target band. That's expected, not a misconfiguration: the
ceiling is deliberately independent of the VRAM target, not a second way
of expressing it.

Called from `learning/main.py`, before `NeuromorphicEncoder` builds any
DataLoader:

```python
cfg.apply_dataset_shape(sensor_h=sensor_h, sensor_w=sensor_w, in_channels=in_channels,
                         num_classes=dataset_entry["num_classes"])
if cfg.CALIBRATE_BATCH_SIZE:
    cfg.BATCH_SIZE = calibrate_batch_size(ModelClass, cfg, device, timesteps=wf.N_TIME_BINS,
                                           data_vram_fraction=wf.BATCH_VRAM_FRACTION)

encoder = NeuromorphicEncoder(cfg)
train_loader, test_loader = encoder.get_dataloaders()
```

`timesteps` must be `wf.N_TIME_BINS`, not a separately configured constant — see
below for why.

- `cfg.BATCH_SIZE` must be set before the encoder builds `train_loader` and
  `test_loader`, since both loaders read it at construction time.
- One calibration policy covers both phases — there's no separate
  inference batch size. Inference's real VRAM footprint is smaller than
  training's (no backward graph, no gradients, no optimizer state), so a
  batch size that fits training's larger footprint fits inference too.

Also recomputes `cfg.ITERA` (`training.iterations_per_epoch`): once
`BATCH_SIZE` is settled, `learning/main.py` sets
`cfg.ITERA = len(train_loader.loader)` — one epoch becomes (up to
`batch_size - 1` samples short of) one real pass over the actual training
set, not a fixed iteration count someone picked independently of batch size
or dataset size. `len(DataLoader)` rather than
`ceil(len(dataset)/batch_size)`: the loader's own `__len__` already accounts
for `drop_last=True` (floor, not ceil) — using `ceil()` previously
overcounted by one iteration the loader never actually produces, since it
exhausts (dropping the final partial batch) one iteration earlier. Same
toggle as batch size — `calibrate_batch_size: false` leaves
`iterations_per_epoch` exactly as written in `SNN_module.yaml`, for someone
who wants to size an
epoch by hand.

## BPTT is not a tunable parameter, and T has exactly one source

How gradients get computed backward through an unrolled network is not
configurable anywhere in this codebase: once the network is unrolled T
steps, calling `.backward()` runs full backpropagation-through-time via
each framework's own autograd. There's no config for this because it
isn't a parameter, it's a computation. All four frameworks
(`frameworks/snn_torch.py`, `snn_norse.py`, `snn_spikingjelly.py`,
`snn_sinabs.py`) run the same full, untruncated BPTT over the same T —
none does truncated BPTT, gradient checkpointing, or detaches the graph
mid-sequence (verified: no `detach()`/truncation calls in any of them).
That equivalence is what makes cross-framework comparisons meaningful —
no framework is taking an algorithmic shortcut the others aren't.

T itself — the actual unroll length — has exactly one source:
`WorkflowSettings.N_TIME_BINS` (`framing.n_time_bins` in
`data_workflow.yaml`), which sets how many frames `ToFrame` produces per
sample. Every model just loops over however many frames land in the
tensor (`data.size(0)` in e.g. `frameworks/snn_torch.py`) — nothing reads
a separate "timesteps" config.

`SNN_module.yaml` used to carry its own `training.timesteps` constant,
read into `cfg.TIMESTEPS` and used only for post-hoc reporting (the
SynOps estimate and firing-rate-in-Hz calculations in
`learning/training.py`/`learning/inference.py`) and, more seriously, to
build the synthetic probe tensor inside `measure_batch_vram`
(`learning/utilities.py`) — the real forward+backward pass that
`calibrate_batch_size` uses to size `BATCH_SIZE`. That constant had
drifted out of sync with `N_TIME_BINS` (25 vs. the real 16), which meant
every SynOps/firing-rate figure was off by ~1.56x and the VRAM
calibration probe was measuring memory at the wrong T. Removed entirely:
`cfg.TIMESTEPS` no longer exists. `training.py`/`inference.py` now read
the real T straight off the probed batch's shape (`probe_data.shape[0]`,
before the `tensor_format() == "BT"` permute), and
`calibrate_batch_size`/`measure_batch_vram` take `timesteps` as an
explicit argument, sourced from `wf.N_TIME_BINS` at the call site — one
source of truth, no second number to keep in sync by hand.

## `NeuromorphicEncoder.compute_prefetch_depth` (event_data_workflow/data_pipeline.py)

What it measures and why: how many *raw input* batches (pre-model, just
the input tensor — kilobytes to low megabytes per sample, not the model's
activation graph) to queue ahead of the GPU, sized from live free VRAM and
this dataset's real per-sample size, capped [1, 32].

This is a different quantity from what `calibrate_batch_size` sizes, not a
duplicate of it — `calibrate_batch_size` sizes one training step's full
computational graph (activations + gradients for every BPTT timestep);
this sizes a queue of not-yet-processed input tensors sitting ahead of
that. Both draw from the same card, though, which is exactly where a real
bug lived: this runs *after* `calibrate_batch_size`'s own probe passes
have freed their test allocations, but *before* real training has claimed
anything — a live VRAM snapshot at that moment looks more available than
it's about to be once training starts. The fix: subtract
`batch_vram_fraction` of total VRAM (the share `calibrate_batch_size`
already earmarked) before sizing the queue, so the two don't double-book
the same memory. Before this fix, the two calibrations disagreeing about
how much VRAM was actually free caused a real out-of-memory crash — not a
clean Python exception, a native crash during CUDA event/pinned-memory
cleanup, because the allocator was already in a bad state.

Can be disabled: `resource_policy.calibrate_prefetch_depth: false` in
`data_workflow.yaml` uses `prefetch_depth_fallback` exactly as written, no
probe run.

## `select_inference_mode` (learning/utilities.py)

Prompts the user, right before the inference/testing phase, to choose
between statistics-only output and statistics + a live matplotlib
visualization of input frames vs. predictions.

- Prints a numbered picker and reads one line via `input()`.
- No stdin attached (Colab, CI, batch — having a terminal to launch the
  process doesn't guarantee stdin is a live keyboard) raises `EOFError`,
  caught to default to statistics-only rather than crashing the run over
  a cosmetic choice.
- An invalid or empty selection also falls back to statistics-only rather
  than erroring, since this is a cosmetic choice, not a config error.

## `SystemResourceMonitor.dataloader_config` (event_data_workflow/system_monitor.py)

Turns live system state into the actual `DataLoader` construction policy —
this is the decision point for how data physically moves from disk/RAM into
the GPU during a run: how many CPU worker processes read and transform
samples in parallel, how many batches each worker stages ahead of time
(`prefetch_factor`), whether host memory is pinned for faster CPU→GPU
copies (`pin_memory`), and whether worker processes are kept alive across
epochs instead of respawned (`persistent_workers`). Called from
`NeuromorphicEncoder.create_loaders()` as `monitor.dataloader_config(self.cfg, device)`.

- Reads live state via `self.snapshot()` (available RAM) and
  `psutil.cpu_count(logical=False)` — the same snapshot mechanism every other
  resource decision in this codebase uses (e.g. `AdaptiveCacheController`'s
  cache-mode choice in `cache_engine.py`), rather than a separate probe.
- Worker count is every physical (not logical/hyperthreaded) core, full stop
  — no fraction, no per-worker byte-budget estimate. Physical, not logical,
  because a CPU-bound worker gets little from a sibling hyperthread, and
  Windows CUDA pinned-memory handles run out well before the logical-core
  count is reached (reproduced directly: a real crash at 32 workers on this
  machine's 16-physical/32-logical-core CPU, clean run at 16).
- The one exception: `num_workers=0` when available RAM is under
  `memory_safety_margin_gb` (the same margin `AdaptiveCacheController` uses)
  — a genuinely RAM-starved host can't safely take on worker processes at
  all, regardless of CPU count.
