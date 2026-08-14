# Session Log — VRAM Cache Removal, Monitoring Consolidation, Prefetch Merge

A narrative account of one working session's changes to `event_data_workflow/`
and the `learning/` training/inference loop: what changed, why, what
performance impact each change actually has, and the open diagnostic question
it ended on. Written in the same spirit as `session_log.md` — the "what
happened and why," not just a diff.

---

## 1. Starting point

The adaptive cache controller (`cache_engine.py`) had a VRAM-resident cache
tier (`GPURecordingCache`, `force_mode: gpu_memory`) left over from an earlier
design that assumed three separate hardware *configurations* — CPU-only,
hybrid, GPU-only. That assumption no longer held: this project runs one fixed
topology (CPU always loads/caches, GPU always trains), so `gpu_memory` wasn't
a legitimate deployment option, it was a dataset cache competing with the
model's own weights/activations/gradients for the same VRAM.

## 2. What changed, in order

**VRAM cache tier removed.** `GPURecordingCache`, `compute_gpu_cache_budget()`,
`GPU_PHASE_CAPS`/`GPU_EMERGENCY_MARGIN`/`GPU_MAX_CACHE_GB`, and the
`gpu_memory` branch of `determine_dataset_strategy()` are gone.
`AdaptiveCacheController` now only ever returns `MemoryCachedDataset` /
`DiskCachedDataset` / `BoundedRecordingCache` — RAM, disk, or both. The GPU is
never a dataset storage location, only the training device. **Impact:** zero
risk of the dataset cache and the model's own VRAM allocations competing —
the exact failure mode measured earlier in this project's history (a 6.9GB
dataset against a ~0.7GB VRAM budget turned a ~6min run into 46+ minutes,
because the cache evicted almost every entry before the next epoch reached it
again).

**GPU-phase VRAM headroom check restored, differently.** The phase-cap
numbers above weren't only about the dataset cache — they were meant to
protect the model's own VRAM. That capability now lives as
`SystemResourceMonitor.enter_phase("training"/"testing")` — a diagnostic-only
log line comparing free VRAM against a phase-specific safety margin. It takes
no corrective action; it only makes it visible in the logs when a phase is
running close to the edge.

**One shared `SystemResourceMonitor` instead of three independent ones.**
Before: `AdaptiveCacheController`, `dataloader_config()`, and the
trainer/tester's phase monitor each built and queried their own
`SystemResourceMonitor`, independently re-probing the same live RAM/disk/VRAM
state. Now: one instance, `monitor`, built once at module import in
`system_monitor.py`, imported directly everywhere (`from .system_monitor
import monitor`) and called like a library — no constructor plumbing, no
singleton-accessor functions. `NeuromorphicEncoder.__init__` calls
`monitor.configure(...)` once, early, as soon as the run's cache path and
device are known.

**`GPUStats` dismantled — its properties absorbed into `PipelineMonitor`.**
`GPUStats` (per-epoch utilization/power/energy, `training.py`/`inference.py`
only) and `PipelineMonitor` (continuous background sampler, previously only
used by the offline `diagnostics/gpu_utilization_harness.py`) were two
independent classes each running their own background sampling thread against
the same GPU. `PipelineMonitor` gained `reset_epoch_memory()`/
`epoch_memory_gb()` (peak/current VRAM per phase), `measure_idle_baseline()`/
`dynamic_power_w()` (idle-power-subtracted "dynamic" power reporting — now
wired into **both** `training.py` and `inference.py`; previously only
inference had it), and `phase_summary()`/`phase_energy_j()` (replacing
`end_epoch()`/`gpu_energy_j()`). One epoch is now just one phase label
(`f"epoch_{n}"`) in the same phase-tagged sample stream the diagnostics
harness already used. `event_data_workflow/gpu_stats.py` is deleted.
**Impact:** one background sampling thread instead of two; the offline
harness and the real training/inference path now share the exact same
instrumentation instead of two independently-maintained reimplementations.

**`AsyncGPUPrefetcher` + `CudaPrefetcher` merged into one `PrefetchedLoader`,
moved into `data_pipeline.py`.** These two classes (CPU-side background-thread
prefetch, and CUDA-stream H2D double-buffering) lived in their own
`prefetch.py` file and were wrapped around the DataLoader identically, by
hand, in both `training.py` and `inference.py`. Now `NeuromorphicEncoder.
create_loaders()` builds the DataLoaders and wraps them once — `self.
train_loader`/`self.test_loader` come out already prefetch-ready.
`training.py`/`inference.py` just iterate them directly; neither imports or
constructs a prefetcher anymore. `event_data_workflow/prefetch.py` is
deleted. Caught and fixed a double-wrap bug from an earlier draft of this
change, where both the encoder and the trainer/tester were each wrapping the
loader.

**`device_idx` removed everywhere.** `training.device` in `SNN_module.yaml`
is `cpu | cuda | auto` — never an indexed `cuda:N` — so `device.index` is
always `None` and `(device.index or 0) if device.type == "cuda" else 0`
always evaluated to the same constant `0`, duplicated identically across five
files. `gpu_total_memory_gb()`, `SystemResourceMonitor`, `PipelineMonitor`,
and `read_gpu_runtime_diagnostics()` all hardcode device `0` internally now.
Also removed `AdaptiveCacheController`'s `device`/`cuda_enabled`/`device_idx`
attributes, which had been dead (computed, never read) since the VRAM cache
tier was removed.

**Smaller fixes along the way:**
- `system_monitor.py`'s `import pynvml` no longer wrapped in a pointless
  `try/except ImportError` — `pynvml` (via `nvidia-ml-py`) is a hard pin in
  `requirements.txt`, and a sibling file already imported it directly with no
  guard.
- `dataset_registry.py` gained `storage_size_gb` per entry (sourced from
  `docs/Event-Based_camera.md`'s measured/documented download sizes; `None`
  where no figure exists, e.g. DSEC).
- No leading-underscore names anywhere touched this session (including a
  rename pass through `PipelineMonitor`'s previously-`_`-prefixed internals:
  `_phase`→`phase`, `_t_origin`→`t_origin`, `_stop_event`→`stop_event`,
  `_thread`→`thread`, `_open_violation`→`open_violation`).

## 3. What this does *not* change

Nothing here changes model accuracy, loss, or training dynamics — every
change this session is in the measurement/caching/loading plumbing around
the training loop, not the loop's own math. The one behavior-visible change
is training.py now also reports idle-baseline-subtracted "Dynamic Power"
(previously inference-only) and prints real (not suppressed) GPU
memory/utilization lines on a CPU-only run — `phase_summary()`/`summary()`
now always return a populated dict (with zeros) instead of `{}`, so those
print lines show `0.0 GB / 0.0 GB (0.0% peak)` instead of vanishing on a
CPU-only run. Cosmetic, not a behavior regression.

## 4. Open diagnostic: `gpu_idle`/`ram_low` warnings from a real training run

A real run (not a smoke test) logged:
```
gpu_idle sustained for 5.1s (phase='unset',   started at t=0.2s)
gpu_idle sustained for 5.1s (phase='epoch_0', started at t=323.3s)
gpu_idle sustained for 5.2s (phase='epoch_1', started at t=482.3s)
ram_low  sustained for 5.2s (phase='epoch_1', started at t=489.7s)
gpu_idle sustained for 5.1s (phase='epoch_1', started at t=550.8s)
```

**`phase='unset'` at t=0.2s — a false alarm, and a labeling gap on our side.**
`SNNTrainer.train()` calls `pipeline_monitor.start()` then immediately
`measure_idle_baseline()`, which *deliberately* holds the GPU idle for ~1s to
sample its resting power draw — plus the "measure FLOPs first" dense-MAC
probe that runs before the epoch loop's first `set_phase("epoch_0")`. Neither
is tagged yet, so `PipelineMonitor` — correctly, by its own rules — flags
that window as unexplained idle time under its default `"unset"` phase.
**Not yet fixed**: call `pipeline_monitor.set_phase("startup")` before
`measure_idle_baseline()` so this window is labeled instead of looking like
an unexplained stall.

**The `epoch_0`/`epoch_1` warnings — real, and unresolved.** These are
genuine, brief (5.1-5.2s, just over the 5.0s reporting floor) stalls during
actual training. The `ram_low` warning sits between two `gpu_idle` warnings
in time, which is suggestive but not confirmed: `AdaptiveCacheController`'s
`memory` mode (tonic's `MemoryCachedDataset`) is lazy but has **no eviction
bound** — every distinct sample touched is kept forever, unlike `hybrid`
mode's `BoundedRecordingCache`, which is lazy *and* FIFO-bounded. If `memory`
mode was selected for this run and the upfront size estimate
(`estimate_dataset_memory_footprint()`, extrapolated from only 10 sampled
items) undercounted real per-sample size, or if available RAM at
decision-time was higher than it stayed once 6 DataLoader worker processes
were all running concurrently, RAM usage climbing past budget — with a
resulting stall as the OS comes under memory pressure — is a plausible
mechanism. **Not confirmed**: which cache tier this run actually selected.
`AdaptiveCacheController.log_diagnostics()` prints a `[CACHE CONTROLLER]`
block with the picked mode near the top of every run's output; that wasn't
captured for this run. Next step if this recurs: capture that line, and if
it's `memory` mode on a RAM-constrained run, either force `hybrid` mode
(bounded) or `disk` mode via `data_workflow.yaml`'s `cache.force_mode`, or
lower `memory_cache_threshold_gb`/raise `memory_safety_margin_gb` so the
adaptive picker is more conservative about choosing `memory` mode on this
machine.

## 5. Files touched

`event_data_workflow/cache_engine.py`, `event_data_workflow/system_monitor.py`,
`event_data_workflow/data_pipeline.py`, `event_data_workflow/dataset_registry.py`,
`event_data_workflow/gpu_stats.py` (deleted), `event_data_workflow/prefetch.py`
(deleted), `event_data_workflow/README.md`, `learning/training.py`,
`learning/inference.py`, `learning/utilities.py`, `learning/main.py`,
`configuration/data_workflow.yaml`, `docs/results/run_benchmark.py`,
`docs/results/make_plots.py`, plus documentation updates across
`docs/Hardware/event_data_workflow.md`, `docs/event_data_workflow/
Dataset_workflow.md`, `docs/event_data_workflow/caching_pipeline_refactor.md`,
`docs/Event-Based_camera.md`, `docs/frameworks/realtime_nir_evaluation.md`,
`docs/v_model_diagnosis.md`.
