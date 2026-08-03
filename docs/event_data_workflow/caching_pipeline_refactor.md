## Caching pipeline refactor — report

This documents a single working session that took `event_data_workflow`'s
caching layer from "functionally plausible but internally inconsistent" to
verified-correct across all three hardware configurations the project
actually needs: CPU-only, hybrid (CPU RAM/disk cache + GPU training), and
GPU-only (VRAM-resident cache). Supersedes `pipeline_coordinator.md`, which
documented two files (`pipeline_coordinator.py`, `temporal_slicer.py`) that
no longer exist — their contents were split up and relocated, see below.

---

### Why this mattered

The project's goal is training SNNs efficiently and comparably across
frameworks on GPUs. `event_data_workflow` isn't the SNN training logic — it's
the plumbing that supplies real event-camera data to that training loop under
whatever hardware happens to be available (a laptop, a shared cluster GPU,
free Colab, a GPU-only embedded box). If that plumbing is wrong, any
comparison built on top of it — accuracy across cache modes, energy per
sample, hardware efficiency — inherits the error silently.

---

### 1. Bugs found and fixed

**Augmentation-freezing bug (the most consequential one).**
`GPURecordingCache` and `BoundedRecordingCache` were baking the *entire*
transform pipeline — including `RandomRotation`, which is supposed to vary
every epoch — into the cached value on a recording's first touch. Every
later access, every later epoch, replayed that same frozen rotation instead
of a fresh one. This silently weakened data augmentation specifically in
hybrid and GPU-only modes (not in plain memory/disk mode), which means any
accuracy difference observed *between* cache modes could have been partly
this bug, not a real architectural effect.

*Fix:* `AdaptiveCacheController.determine_dataset_strategy()` now takes
`transform` (deterministic, e.g. Denoise+ToFrame — safe to cache) and
`live_transform` (stochastic, e.g. rotation — must run fresh) as separate
parameters. `live_transform` is threaded through each cache tier's own
post-lookup hook so it re-runs on every access, hit or miss, instead of
being baked in once. Verified in the sample test: encode/deterministic
transform runs only on cache misses; live/stochastic transform runs on every
access and produces a different result each time.

**CUDA / DataLoader-worker-process crash risk.**
`GPURecordingCache` holds live CUDA tensors. DataLoader worker *processes*
can't safely share or reconstruct that state (fork can't reinitialize CUDA;
spawn duplicates the context and makes cross-process CUDA tensor handoff
fragile). Nothing previously stopped a GPU-resident cache from being handed
to a multi-worker DataLoader.

*Fix:* a duck-typed `requires_single_process_loading` flag, set on
`GPURecordingCache`, checked by `NeuromorphicEncoder.loader_kwargs()` to force
`num_workers=0` for that specific loader — without `data_pipeline.py` needing
to import or `isinstance`-check the concrete cache class.

**CPU-only runs could still be influenced by an unused GPU.**
`SystemResourceMonitor` used to call `torch.cuda.is_available()` directly,
so a machine with a physically-present GPU that a run explicitly wasn't
supposed to use (`training.device: cpu`) could still have its cache/worker
decisions steered by that GPU's VRAM numbers.

*Fix:* `SystemResourceMonitor` now takes an explicit `cuda_enabled` flag,
threaded through every caller (`AdaptiveCacheController`,
`dataloader_config()`), so GPU probing only happens when the run actually
requested CUDA.

**Interactive pickers could crash instead of falling back.**
Both the dataset picker and the new hardware-configuration picker check
`sys.stdin.isatty()` before prompting. Testing surfaced that `isatty()` can
report `True` in some non-interactive contexts (certain CI runners,
notebook-cell execution, and this development harness itself) with no real
input behind it — `input()` then raises `EOFError` and crashes the run.

*Fix:* both pickers now catch `EOFError` and fall back to their default
(N-MNIST for the dataset picker; autodetected hardware config for the new
one) instead of crashing.

---

### 2. Architecture simplified

**S3-FIFO → plain FIFO eviction.**
The original cache used the S3-FIFO algorithm (Small/Main/Ghost queues,
frequency counters) — designed to protect a *skewed* access pattern's hot
working set from one-hit-wonder pollution. This pipeline's actual access
pattern is a shuffled `DataLoader`: every recording is equally likely to
recur each epoch, with no popularity skew for that machinery to exploit.
`BaseRecordingCache` (renamed from `BaseS3FIFOCache`) now does plain
FIFO — a single `deque` for insertion order, evict-oldest-on-capacity — with
no measurable loss in practice for this workload.

**Removed `auto_tune_slicing` / `AdaptiveTemporalSlicer`.**
A ~100-line subsystem that sampled 50 recordings per run and derived a slice
duration heuristically — solving "we don't know this dataset's timing
characteristics," a real problem for an arbitrary unknown dataset, but not
one this pipeline has: it works from a fixed, known dataset registry whose
characteristics can be measured once, offline, and set explicitly via
`slice_duration_ms`. Removing it traded a runtime heuristic for a
reproducible config value — strictly better for a research pipeline where
"same config → same result" matters.

**Removed confirmed-dead code**, verified by grepping for callers before
deleting anything (not guessed): `AdaptiveCacheController.get_system_metrics()`,
a duplicate `clear_cache()` implementation (consolidated to one, in
`AdaptiveCacheController`, delegated to from `NeuromorphicEncoder`),
`PipelineMemoryCoordinator.max_cache_bytes()`, `.max_recordings()`,
`.prefetcher()`, `.prefetch_queue_size()`.

**Split `pipeline_coordinator.py`.**
The file had drifted into three unrelated jobs sharing one file: DataLoader
worker-count sizing, an async prefetch thread, and a per-timestep spike
buffer used inside the *model's* forward pass. Each piece now lives next to
its one real caller instead:

| Was in `pipeline_coordinator.py` | Now lives in |
|---|---|
| `dataloader_config()` | `data_pipeline.py` (plain function, its only caller) |
| `DenseTimestepBuffer` | `activity_reg.py` (its only caller — a training-loop concern, not a data-pipeline one) |
| `AsyncGPUPrefetcher` | new `prefetch.py` (its actual callers: `training.py`, `inference.py`) |

`pipeline_coordinator.py` and `temporal_slicer.py` are both deleted.

**Clarified `force_mode` as the adaptive on/off toggle**, instead of adding
a second, redundant boolean flag on top of it — `force_mode: null` probes
live resources and picks a strategy; any other value forces that strategy.
Adding a separate flag would have reintroduced the exact
overlapping-feature problem this session was removing elsewhere.

**Removed orphaned config**: the dead `temporal_slicing.auto_tune` YAML key
(left behind once the feature was removed), and `architecture.device` — a
YAML value that turned out to never be read by any code at all (`cfg.DEVICE`
only ever reads `training.device`; confirmed by grep before removal).

---

### 3. New capability: async prefetching

`AsyncGPUPrefetcher` (`event_data_workflow/prefetch.py`) closes a
previously-documented-but-never-implemented gap — `prefetch_queue_size()`
existed with no actual prefetcher behind it. It wraps a `DataLoader` and
runs it one batch ahead in a background **thread** (not process — CUDA state
stays valid across a thread boundary, unlike a forked/spawned worker
process), so the GPU doesn't sit idle waiting for the CPU to prepare the
next batch. Real overlap, not fake concurrency: numpy/tonic's C-level ops
release the GIL during computation, and the main thread's CUDA syncs release
it while waiting on the GPU — both give the prefetch thread room to run.
Wired into both `SNNTrainer.train()` and `SNNTester.run()`.

---

### 4. New capability: hardware-configuration picker

`training.device: auto` in `SNN_module.yaml` now triggers an interactive
picker in `main.py` (`select_hardware_config()`), matching the existing
dataset-picker convention:

1. **CPU only** — `device=cpu`; data loading, caching, and training all stay
   on CPU (end-to-end, verified this session).
2. **Hybrid** — `device=cuda`, `force_mode` left adaptive so the resource
   probe picks memory/disk/hybrid based on live RAM.
3. **GPU only** — `device=cuda`, `force_mode=gpu_memory` — the VRAM-resident
   cache path this session's fixes specifically targeted.

Any explicit `cpu`/`cuda` value in the config skips the prompt entirely, so
existing scripted/reproducible runs are unaffected. Non-interactive contexts
(Colab, CI, batch jobs) autodetect instead of prompting: CUDA available →
hybrid, else CPU-only.

---

### 5. Verification

A synthetic-data test (`test_cache_engine_sample.py`, no real dataset
download required) exercises exactly what changed:

- Plain FIFO eviction order (oldest-inserted entries evicted first,
  regardless of access frequency)
- `encode_transform` runs only on cache misses; `live_transform` runs on
  every access and produces different output each time — both for the CPU
  cache tier and `GPURecordingCache`
- `SystemResourceMonitor` respects `cuda_enabled` regardless of actual
  hardware presence
- `AdaptiveCacheController.determine_dataset_strategy()` end-to-end for
  `memory` and `gpu_memory` forced modes, including confirming the cached
  tensor actually lives on CUDA

**18/18 checks pass.** The same testing process independently isolated a
stack-overflow crash in `tonic.DiskCachedDataset` when given non-standard
synthetic input — confirmed, by calling `tonic`'s class directly with no
code from this session involved, to be a pre-existing third-party library
issue unrelated to this work, not a regression.

---

### Current architecture

```
SystemResourceMonitor (cuda_enabled-gated RAM/disk/VRAM probe)
        │
        ├──► dataloader_config()          [data_pipeline.py]
        │        num_workers / pin_memory / prefetch_factor
        │
        └──► AdaptiveCacheController.determine_dataset_strategy()
                 │
                 ├── MemoryCachedDataset   (memory mode)
                 ├── DiskCachedDataset     (disk mode)
                 ├── BoundedRecordingCache (hybrid: CPU RAM hot layer over disk)
                 └── GPURecordingCache     (gpu_memory: VRAM-resident cache)
                          │
                          ▼
                 DataLoader ──► AsyncGPUPrefetcher ──► training / inference loop
                                                              │
                                                              ▼
                                                    DenseTimestepBuffer
                                                    (per-timestep spike buffer,
                                                     activity_reg.py)
```

---

### Net result

CPU-only, hybrid, and GPU-only now mean exactly what their names say, each
independently selectable via one config value or an interactive prompt, each
verified rather than assumed to work, and each free of the dead machinery
that made the previous version hard to explain in a report without
hand-waving past code that turned out not to do anything.
