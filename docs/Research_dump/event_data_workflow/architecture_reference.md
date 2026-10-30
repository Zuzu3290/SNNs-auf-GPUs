# Event Data Workflow — Architecture Reference

This describes how information actually *moves* through the pipeline —
what happens once versus what happens continuously, what decides versus
what executes, and what the CPU and GPU are each doing at every stage —
rather than listing files and classes. For "which file has which class,"
see `event_data_workflow/README.md`; this document is the flow those
classes implement.

---

## The core split: decide once, act continuously

The single most important structural fact about this pipeline is that
**setup and execution are two different phases with two different
rhythms**, and conflating them is the easiest way to misread how the
system behaves.

**Decided once, before training starts, then frozen for the entire run:**
- *Which dataset* is loaded at all — `resolve_dataset_entry()`
  (`event_data_workflow/dataset_registry.py`) matches `cfg.DATASET_NAME`
  against `DATASET_REGISTRY` once, before any other decision in this list;
  everything below (sensor shape, class count, cache sizing) derives from
  whichever entry this step returns. `DATASET_REGISTRY` is the single
  source of truth for which datasets exist — `configuration/SNN_module.yaml`
  carries no dataset declaration of its own anymore, only the
  `cfg.DATASET_NAME` lookup key this step consumes.

  *Fixed:* a name that matched nothing in the registry used to fall
  through silently to the N-MNIST default in any non-interactive run
  (scripts, CI, notebooks) — no log line indicated the requested dataset
  had never actually resolved. The non-interactive fallback now logs a
  warning naming the exact `DATASET_NAME` that failed to match before
  defaulting, so a typo'd or stale dataset name is visible in the run's
  own log instead of silently substituting N-MNIST.
- *Which* cache mechanism holds the data — RAM (`memory`) or disk
  (`disk`). Chosen by `AdaptiveCacheController.determine_dataset_strategy()`,
  called exactly once per split (train, test) inside
  `NeuromorphicEncoder.build()`.
- *How many* DataLoader worker processes run, and their pin-memory/
  prefetch settings — `dataloader_config()`, also called once.
- *How deep* the GPU-side prefetch buffer is — `compute_prefetch_depth()`,
  live-computed from free VRAM and this dataset's real per-sample size
  (see the calibration section below), not a fixed constant.
- The device (CPU/GPU) and whether GPU probing is even active
  (`cuda_enabled`).

Nothing re-runs any of this mid-training. There is no logic anywhere in
the pipeline that re-picks the cache tier, resizes a cache budget, or
changes worker count once training has begun — that dynamic-adjustment
design existed at an earlier point (a VRAM-resident cache tier that
resized itself against live GPU headroom) and was deliberately removed;
the dataset is never cached in VRAM at all now, specifically so training
never has to compete with its own data pipeline for GPU memory.

**Happens continuously, for the life of the run:**
- Every single sample request checks the cache fresh: hit → served
  instantly; miss → transform, then insert. This repeats every batch,
  every epoch, for as long as new (or evicted) samples keep showing up.
- The background prefetch thread keeps pulling and staging batches ahead
  of the training loop the entire time it's iterating.
- The diagnostics layer (`SystemResourceMonitor`, `PipelineMonitor`)
  keeps sampling live RAM/VRAM/CPU state and logging — but only
  *observes*; it never feeds back into any decision above.

So: **one decision, many executions.** The controller doesn't keep
"controlling" — it picks a mechanism once and gets out of the way; the
mechanism itself is what stays active.

---

## Following one sample's actual journey

Layer order is strict and enforced (`determine_dataset_strategy()` raises
if it's ever handed an already-sliced dataset) because caching after
slicing would multiply cache entries by the slice count — N slices from
one recording would mean N separate cache entries for data that's
identical up to the slice boundary, instead of one.

**1. Raw load.** A recording is read off disk in its native form —
structured event arrays (t, x, y, polarity) via `tonic`. No transform, no
caching yet. This step is cheap; nothing expensive has happened.

**2. Cache lookup.** The wrapping cache object (`MemoryCachedDataset` or
`DiskCachedDataset`, whichever tier was picked once at startup) checks
whether this index has been seen before.

- **Hit** (already cached): the *already-transformed* result is returned
  directly. No Denoise, no ToFrame, no recomputation — this is the entire
  reason the cache exists, since Denoise+ToFrame is the expensive step
  and would otherwise re-run every single epoch.
- **Miss** (first time this index has been touched, or it was evicted):
  the raw recording is pulled and run through the deterministic transform
  chain (`Denoise` → `ToFrame`) *once*, and the result is what actually
  gets stored — not the raw events. This is enforced by
  `PreTransformedDataset`'s own `__getitem__`,
  specifically because tonic's own cache classes only skip re-fetching
  the *raw* sample on a hit — they'd otherwise still re-run whatever
  transform they're given on every access, hit or miss, defeating the
  cache's entire purpose.

**3. Live augmentation (train split only).** `RandomRotation` is
stochastic — baking it into the cache would freeze the same "random"
rotation onto that sample forever. So it's kept deliberately outside the
cached value and re-applied fresh on *every* access, cache hit or miss —
threaded through as a separate `live_transform` argument, composed after
the cached (deterministic) value is retrieved, never baked in.

**4. Temporal slicing (optional, if enabled).** Only ever applied to
*cached, raw* recordings — never to a raw dataset directly, and never
after a cache tier has already baked in a transform. `create_sliced_dataset()`
cuts each cached recording into fixed-duration (or fixed-event-count)
windows, building the slice index once and persisting it to
`metadata/{split}/slice_metadata.h5` so a later run loads the index
instead of rebuilding it.

**5. Batch assembly.** `DataLoader` groups individual samples (or slices)
into batches, using however many worker processes `dataloader_config()`
decided on at startup, each independently doing steps 1-4 for its own
share of indices in parallel.

**6. Prefetch and hand-off to GPU.** This is where `PrefetchedLoader`
(built in `data_pipeline.py`, wrapping the DataLoader from step 5) takes
over. It keeps a background thread pulling completed batches from the
DataLoader into a queue, and — separately — keeps up to `PREFETCH_DEPTH`
of those batches already copied onto the GPU via a dedicated CUDA stream,
ahead of when the training loop actually asks for them. By the time
`SNNTrainer`/`SNNTester` requests the next batch, it's typically already
GPU-resident; no copy has to happen in that moment.

**7. Training loop.** Receives a batch that's already on the GPU,
computes the forward/backward pass, updates weights. From this loop's
point of view, steps 1-6 are invisible — it just asks for the next batch
and (almost always) gets one immediately.

---

## The two worlds running side by side

The pipeline is fundamentally two things happening concurrently, not one
thing happening in sequence:

```
CPU world (steps 1-6, continuous)          GPU world (step 7, continuous)
─────────────────────────────────          ──────────────────────────────
cache lookup → (miss: transform)            forward pass
   → live augmentation → batch              backward pass
   → prefetch queue → H2D copy              optimizer step
        (background thread + CUDA stream, overlapped with the row below)
                    ──────────────►  already-resident batch handed over
```

Before the prefetch mechanism existed, these two rows ran strictly
sequentially per batch — CPU finishes preparing, *then* GPU starts
computing, over and over. The entire point of `PrefetchedLoader` is that
the CPU row for batch N+1 runs *while* the GPU row for batch N is still
executing, so the GPU stops paying the CPU's preparation time as its own
idle time.

---

## The monitoring plane — watches, never participates

Running alongside both worlds, but never feeding back into either:

- **`SystemResourceMonitor`** (shared singleton, `monitor`) — the
  decision input. `AdaptiveCacheController` and `dataloader_config()`
  read it *once*, at startup, to make their one-time choices. After
  that, it's only used for `enter_phase("training"/"testing")` — a log
  line comparing free VRAM against a safety margin, no corrective action.
- **`PipelineMonitor`** — the observation output. A background-thread
  sampler (CPU%, GPU util/power/clock, RAM) tagged by phase, used for
  per-epoch reporting (`phase_summary()`, `phase_energy_j()`) and for the
  offline diagnostics harness. Also purely observational — nothing it
  measures changes what the training loop or cache does.

This separation is deliberate: the pipeline's actual behavior (which
cache, how many workers, how deep the prefetch buffer) is fully
determined before the first batch is ever requested, and the monitoring
layer exists to *report* on that fixed configuration's real-world
performance, not to steer it.

---

## What resets between splits, what doesn't

`AdaptiveCacheController.determine_dataset_strategy()` is called
separately for `train` and `test` — each split gets its own independent
cache-tier decision (they can legitimately end up on different tiers if
their sizes differ enough to cross a threshold) and its own cache
directory (`cache/{split}/`). Within a split, the cache persists for the
entire run — it is never cleared or reset between epochs. Epoch 2 of
training sees exactly the same cache state epoch 1 left behind: whatever
was cached stays that way until the process exits or `clear_cache()` is
called explicitly.

---

## Batch size, iterations, and epochs — what's calibrated, what's not

Three numbers govern how much data a run actually sees, and only two of
them are live-calibrated — the split is deliberate, not an oversight.

**Batch size and iterations-per-epoch are resource/coverage facts** —
they have a correct answer derivable from the hardware and the dataset,
so calibrating them removes a class of user error (under-using VRAM,
or training on a partial/oversampled epoch without realizing it):

- `calibrate_batch_size()` (`learning/utilities.py`) probes the real GPU
  with a real forward+backward pass and picks the batch size that lands
  in a target VRAM band, capped by `resource_policy.max_batch_size` — a
  separate, learning-quality ceiling (large-batch training trades
  gradient-update count for smoother/noisier gradients), independent of
  how much VRAM is actually free. See `docs/functions.md` for the full
  two-probe mechanism and why the two constraints can disagree.
- Once batch size is settled, `learning/main.py` recomputes
  `cfg.ITERA` (`training.iterations_per_epoch`) as
  `len(train_loader.loader)` — one epoch becomes (up to `batch_size - 1`
  samples short of) one real pass over the actual training set, not a
  fixed iteration count picked independently of batch size or dataset
  size. `len(DataLoader)` rather than `ceil(N/batch_size)` since the
  loader's own length already accounts for `drop_last=True`.
- One toggle gates both: `training.calibrate_batch_size` in
  `SNN_module.yaml`. `false` uses `batch_size` and
  `iterations_per_epoch` exactly as written — manual control, no probe.

**Epoch count is a convergence judgment call, not a resource fact** — no
live measurement tells you "the model has learned enough," only a human
deciding from loss curves, time budget, or when to stop. So
`training.epochs` is never calibrated, toggle on or off: it's the one
number this pipeline always leaves to whoever's running it.
