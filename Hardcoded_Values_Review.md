# Hardcoded Values — Pending Review

Tracking doc for constants introduced this session that were engineering
judgment calls, not derived or empirically swept — flagged for a later
review pass (e.g. a walkthrough/video review), not urgent. None of these
currently sit in a path that would break the running system if left as-is;
noted per item where that's not quite true.

---

## `learning/utilities.py` — `calibrate_batch_size()` / `probe_batch_size()`

New this session. Not yet wired into `main.py`/`NeuromorphicEncoder` —
currently only exercised by `diagnostics/evaluate_batch_size_calibration.py`.
Doesn't touch the running training path yet, so nothing here is live-critical
today, but every constant below becomes load-bearing the moment it is wired in.

| Constant | Value | Where it came from | Open question |
|---|---|---|---|
| `baseline_sensor_px` | `34 * 34` (N-MNIST) | N-MNIST is the one dataset with a long-confirmed-safe batch size (128) to scale from | Is N-MNIST the right reference point, or should the baseline be re-derived per-architecture if `network_architecture.yaml` ever changes the conv stack? |
| `baseline_batch_size` | `128` | The global default in `SNN_module.yaml` | Same coupling risk — if the global default changes, this stale constant silently drifts from it (it's a literal, not a read of the config) |
| growth multiplier | `1.5` (hardcoded inline, not a parameter) | Chosen after `2.0` (doubling) was confirmed by testing to overshoot badly enough to crash `torch.cuda.empty_cache()` itself | Not swept — `1.5` was picked as "clearly gentler than 2.0," not tested against e.g. `1.25` or `1.3` for whether it's still too aggressive on a smaller card, or unnecessarily slow on a larger one |
| `safety_margin` | `0.85` | Standard practice in real batch-size finders (never return the exact found ceiling) — the specific number is a judgment call, not swept | Is 15% headroom enough given the fragmentation behavior already observed (N-Caltech101's own search found 28 as fitting, then a same-size re-verification OOM'd)? Only tested down to one margined value, not stress-tested across many runs |
| `max_batch_size` | `256` | Arbitrary upper bound so the growth phase can't run away | Never actually reached in any test run so far (N-MNIST capped at 192, everything else lower) — untested whether 256 is a sane ceiling for a dataset with a much smaller sensor than N-MNIST |
| `min_batch_size` | `1` | Obvious floor | No real issue here, listed for completeness |

**Also open, not just the constants:** the multi-dataset sweep this session
found real allocator fragmentation *within a single process* (calibrating
back-to-back for N-MNIST → N-Caltech101 → DVS128 Gesture degraded state
across datasets). The `safe_empty_cache()` fix (added `gc.collect()`,
guarded the cache-clear call itself) addresses what was tested, but the
sweep was only run a handful of times — not enough runs to call the fix
fully proven, just no-longer-immediately-broken.

---

## `event_data_workflow/data_pipeline.py` — `calibrate_events_per_slice()` (removed)

Was Case A from `Case_Study_Evaluation_Report.md`. Removed entirely, not kept
as an opt-in: `SliceByEventCount` gives each slice a variable, scene-dependent
duration, which is inconsistent with this project's `n_time_bins` framing
(fixed number of equal-time bins per sample) — the two combined mean a
time-bin index maps to a different real dt on every sample, which conflicts
with the fixed time constants the SNN dynamics are defined against.
`SliceByTime` doesn't have this problem and remains the only slicing strategy.
See `docs/Event-Based_camera.md`.

---

## `event_data_workflow/dataset_registry.py` — N-Caltech101 entry

**This one IS live and already verified**, unlike the two above — flagged
for review of the *reasoning*, not because it's unverified.

| Constant | Value | Where it came from | Open question |
|---|---|---|---|
| `batch_size` | `16` | Empirically confirmed in `vram_batch_scaling_task.md` (128 OOMs, 16 completes real forward+backward) | Confirmed working, not confirmed *optimal* — no value between 16 and 128 was tried, so whether e.g. 24 or 32 also fits with the fixes made later this session (cache-namespacing bug, grad-accum flush bug) is unknown |
| `grad_accum_steps` | `8` | Chosen so `16 * 8 = 128`, matching the pre-existing global default exactly | Matching the old global default was a convenience choice for "keep training dynamics comparable," not a value derived from what's actually best for this dataset |

---

## `event_data_workflow/cache_engine.py` — hybrid-mode's total-RAM gate

**This one IS live and already changed**, not just flagged — found while
chasing a "GPU idle again" report, and fixed in the same session.

| Constant | Value | Where it came from | Open question |
|---|---|---|---|
| hybrid-mode `total_ram_gb` gate | was `32.0`, now `16.0` | Arbitrary round number — not derived from anything measurable | `16.0` is also a round-number judgment call, not a swept value. It was picked because the real safety math (`threshold_gb = available_for_cache * 0.5`) already scales off *live* available RAM, not total installed RAM, making the total-RAM gate mostly redundant — but no specific lower bound was tested against an actual low-RAM machine to confirm 16GB is a safe floor rather than just "clearly better than 32 for this one machine" |

Two more real bugs surfaced by turning hybrid mode on for the first time
(it had never been reachable before this session, since the old `32.0` gate
excluded every machine tested on):

1. **Cache-directory collision.** "disk" mode and "hybrid" mode both wrote
   to `cache/<dataset>/<split>` — but disk mode caches the *already-transformed*
   dense frame there, and hybrid mode caches the *raw* event array and
   re-transforms on every read. Whichever mode ran last left the cache in
   the wrong format for the other, and reading it crashed
   (`tonic`'s `Denoise` receiving a dense frame instead of a structured
   event array). Fixed: each mode now gets its own subdirectory
   (`.../disk/`, `.../hybrid/`).
2. **`max_cached_recordings: 500`** (`configuration/data_workflow.yaml`) —
   see below.

## `configuration/data_workflow.yaml` — `max_cached_recordings`

**Also live and already changed.** Hybrid mode's RAM hot layer is bounded
by two independent limits: a recording count and a byte budget computed
from live available RAM. `500` was far below what the byte budget alone
already permitted — measured on this machine, the byte budget allows
roughly ~5,000 cached recordings per worker, so the flat `500` cap was the
one actually binding, discarding ~90% of the real, measured RAM budget and
forcing the expensive per-sample transform to re-run on nearly the whole
dataset every epoch instead of just once.

| Constant | Value | Where it came from | Open question |
|---|---|---|---|
| `max_cached_recordings` | was `500`, now `20000` | Not derived from anything — a round number that predates hybrid mode ever actually running | `20000` was picked to sit comfortably above the ~5,000-per-worker the byte budget allows on this machine, so the byte budget (the value actually derived from live measurement) becomes the real governor again. Not tested on a dataset with much larger per-sample size, where 20000 recordings could itself overshoot a small byte budget before the byte check catches it — the two caps interact, and only one machine/dataset combination has been observed |

This machine has 31.3GB total RAM — under the old `32.0` threshold by less
than a gigabyte, which locked it out of hybrid mode (a bounded, disk-backed
RAM cache) entirely. With `total_ram_gb < 32.0` and typically not enough
*available* RAM for full "memory" mode either (measured as low as 1.66GB
free at one point this session, due to unrelated other applications, not
this codebase), every dataset fell to the slowest tier — full disk I/O with
no RAM cushioning — which is the direct, measured cause of the CPU-bound
GPU-idle pattern reported. Confirmed on this machine specifically:
`disk_free_gb=157.48` (never the limiting factor), `total_ram_gb=31.30`
(the actual gate that was tripping).

---

## Suggested next step, when reviewed

For each table above: either (a) replace the constant with a value derived
from a real measurement/sweep the way the batch-size ceiling itself now is,
or (b) leave it as a named, documented judgment call with the reasoning
recorded here — but not silently trust it as "must be right because it's
already in the code."
