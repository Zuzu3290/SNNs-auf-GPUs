# System Boundaries and Tuning Reference

Every threshold that decides how the pipeline behaves at runtime — which
cache tier, how many workers, what batch size, how many batches per epoch.
Check here first when diagnosing a speed or memory problem, before
re-deriving it from the code again.

---

## 1. Cache tier selection

`event_data_workflow/cache_engine.py`, `AdaptiveCacheController.determine_dataset_strategy()`.
Checked in this order, first match wins:

| # | Condition | Mode chosen | What it means |
|---|---|---|---|
| 1 | GPU VRAM usage > 75% (`GPU_PRESSURE_THRESHOLD`, `system_monitor.py`) | `disk` | A busy GPU means CUDA's pinned-memory allocator competes for the same RAM a memory/hybrid cache would use — disk avoids that fight |
| 2 | `available_ram - 2.0GB safety margin >= 6.0GB` **and** dataset's real (post-transform) size `< 70%` of that | `memory` | Whole dataset held in RAM, no disk I/O at all — fastest, but only viable when there's genuinely enough free RAM |
| 3 | Disk has `> 1.2x` the dataset's size free | `disk` | Already-transformed result cached to disk permanently — the transform runs once per sample, ever, not once per epoch |
| 4 | Total installed RAM `>= 16.0GB` **and** disk has `> 1.5x` the dataset's size free | `hybrid` | Same permanent disk cache as "disk", plus a bounded RAM hot layer on top. Measured to be near-useless under this project's access pattern — see below |
| else | — | `RuntimeError` | Genuinely not enough of either resource to proceed at all |

**Boundary values, and where they live:**
- `memory_safety_margin_gb = 2.0` — RAM always held back from every calculation, never offered to a cache
- `memory_cache_threshold_gb = 6.0` — minimum free RAM before "memory" mode is even considered
- hybrid mode's total-RAM gate = `16.0` (lowered from `32.0` earlier this session)
- `max_cached_recordings = 20000` (`configuration/data_workflow.yaml`) — hybrid mode's hot layer cap, alongside a byte budget computed from live RAM

**Why "disk" is checked before "hybrid" — measured directly, not theorized:**
fetching + transforming one raw sample (`Denoise` + `ToFrame`) costs ~136 ms,
paid once per sample, ever, once cached to disk. A plain disk *read* of
that same already-cached sample costs ~40 ms every access; a RAM lookup of
it costs ~0.0004 ms — in isolation, a huge case for keeping a RAM layer on
top of the disk cache. But training reshuffles the dataset every epoch
(`shuffle=True`), and hybrid's RAM layer is a small, FIFO-evicted window
(hundreds to low thousands of slots out of tens of thousands of samples).
Simulated directly — a real `BoundedRecordingCache` against a fully
shuffled index order, 5 epochs — the actual hit rate measured **~0.01%**:
functionally never triggers. A FIFO cache only pays off when the same item
is revisited before it's evicted; full random reshuffling defeats that
almost completely at this cache-to-dataset ratio. So hybrid mode adds real
bookkeeping overhead for close to zero benefit under *this* access pattern,
and "disk" (no extra layer, same permanent cache) is checked first again.
Hybrid stays available as a fallback for a genuinely different access
pattern (e.g. non-shuffled or small-working-set access), where a hot layer
could actually accumulate real hits.

---

## 2. DataLoader worker count

`event_data_workflow/data_pipeline.py` / `system_monitor.py`, `dataloader_config()`.

```
worker_budget_gb = (available_ram_gb - 2.0) * 0.3
if worker_budget_gb < 0.5:      → num_workers = 0   (GPU-only mode, no multiprocessing)
else:                            → num_workers = min(cfg.NUM_WORKERS, os.cpu_count())
```

**The tunable lever:** `cfg.NUM_WORKERS` comes from `training.num_workers` in
`configuration/SNN_module.yaml` — currently `6`. This machine has **32
logical CPUs**. There's already a live warning for exactly this gap
("num_workers=N but this machine has M logical CPUs"). Raising
`training.num_workers` closer to the core count (leaving a few free for the
OS/other processes — e.g. 16–24, not necessarily all 32) is a real, direct,
untried lever for CPU-side throughput on this machine specifically. It
hasn't been tested this session — flagged here, not applied, since it's a
genuine tuning decision, not a bug fix.

---

## 3. Batch size calibration

`learning/utilities.py`, `calibrate_batch_size()`.

- Target VRAM band: **30%–35%**, aiming for **34%** — deliberately not "the
  largest batch that fits," so the model has headroom to grow later.
- Growth multiplier `1.5`, safety margin `0.85`, ceiling `256`, floor `1` —
  all judgment calls, not swept (see `Hardcoded_Values_Review.md`).
- Runs once, before training starts, using a real forward+backward probe
  (with AMP + gradient accumulation simulated) — not a guess.

---

## 4. Prefetch depth

`event_data_workflow/data_pipeline.py`, `PrefetchedLoader` (composes
`AsyncGPUPrefetcher` + `CudaPrefetcher` from `prefetch.py`).

- `PREFETCH_DEPTH` (default `8`) — how many batches sit GPU-resident at
  once. The CPU-side raw-batch queue matches this depth automatically
  (fixed this session — it used to default to a much shallower `2`,
  independent of `depth`, which throttled the GPU below what `depth` was
  chosen to sustain).

---

## 5. Epoch length vs. batch size

`iterations_per_epoch` (`SNN_module.yaml`, currently `400`) caps how many
batches run per epoch. Whether that cap actually triggers depends entirely
on `batch_size`, which is itself decided live by calibration (§3) — so the
same config value can mean "a full epoch" or "a truncated one" depending on
what batch size gets calibrated that run. This interaction is real and
currently **not resolved as a deliberate choice** — see
`Hardcoded_Values_Review.md` for the open decision (always run the full
dataset, vs. keep an intentional, explicit partial-epoch cap).

---

## How to use this when something looks wrong

1. **Wall time much higher than expected** → check §1 (which cache mode got
   picked — the `[CACHE CONTROLLER]` log line says so directly) and §5
   (is this epoch actually covering more/less of the dataset than the one
   being compared against).
2. **GPU idle episodes reported during training** → almost always §1 (cache
   mode) or §2 (worker count) — CPU-side prep not keeping up, not a GPU or
   model problem.
3. **Out of memory / crashes** → §1's RAM boundaries or §3's VRAM band —
   check what was actually measured at the moment, not just the config
   defaults, since every one of these is a *live* measurement, not a fixed
   number.
