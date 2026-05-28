# Event Data Workflow — Architecture Reference

Graphical reference for the neuromorphic data pipeline implemented in
`src/learning/event_data_workflow/`. Two execution paths are documented:
the standard **CPU + GPU** training path and the embedded **GPU-only** path.

---

## File Map

| File | Responsibility |
|------|---------------|
| `system_monitor.py` | RAM / VRAM / disk probing (`SystemResourceMonitor`, `CacheMetrics`) |
| `cache_engine.py` | Strategy selection, `BoundedRecordingCache`, `GPURecordingCache`, `AdaptiveCacheController` |
| `pipeline_coordinator.py` | RAM budget splits, DataLoader config, GPU pressure guard (`PipelineMemoryCoordinator`) |
| `data_pipeline.py` | End-to-end assembly: raw → cache → slice → DataLoader (`NeuromorphicEncoder`) |
| `temporal_slicer.py` | Stateless temporal windowing (`create_sliced_dataset`) |
| `activity_reg.py` | Per-layer spike recording and regularization losses (`DenseTimestepBuffer`) |

---

## Path 1 — CPU + GPU (Standard Training)

```
┌──────────────────────────────────────────────────────────────────────┐
│  RAW DATASET  (Tonic: NMNIST / DVS-CIFAR10 / FileDataset)           │
│  Events stored as structured NumPy arrays on disk                    │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  system_monitor.py — SystemResourceMonitor.snapshot()                │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  total_ram_gb          available_ram_gb   ram_usage_pct     │    │
│  │  disk_available_gb     disk_exists                          │    │
│  │  gpu_memory_gb         gpu_available_gb                     │    │
│  │    = min(free_driver, total − memory_reserved)              │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────┬───────────────────────────────────────────────────────┘
               │ CacheMetrics
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  pipeline_coordinator.py — PipelineMemoryCoordinator.from_system()   │
│                                                                      │
│  total_budget = available_ram − 2 GB safety margin                   │
│                                                                      │
│  ┌──────────────────┬───────────────────┬─────────────────────┐     │
│  │  Recording cache │  Worker buffers   │  Prefetch queue     │     │
│  │      40 %        │      30 %         │      30 %           │     │
│  └──────────────────┴───────────────────┴─────────────────────┘     │
│                                                                      │
│  GPU pressure guard (> 75 % VRAM used):                              │
│    recording_cache_fraction × 0.5  (halved to avoid OOM)            │
│                                                                      │
│  DataLoader config:                                                  │
│    num_workers = settings.NUM_WORKERS  (SNN_module.yaml)             │
│    pin_memory  = True  (enables async CPU→GPU transfers)             │
│    prefetch_factor = 2                                               │
└──────────────┬───────────────────────────────────────────────────────┘
               │ max_recordings cap, DataLoader params
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cache_engine.py — AdaptiveCacheController.determine_strategy()      │
│                                                                      │
│  Strategy decision tree:                                             │
│                                                                      │
│  RAM > 8 GB and dataset fits (< 70% of free)?                        │
│    YES ──► MEMORY — MemoryCachedDataset  (full dataset in RAM)       │
│                                                                      │
│  RAM ≥ 32 GB and disk free > 1.5× dataset?                           │
│    YES ──► HYBRID — DiskCachedDataset (primary)                      │
│                   + BoundedRecordingCache (RAM hot layer)            │
│                     max_bytes = budget ÷ num_workers                 │
│                                                                      │
│  Disk free > 1.2× dataset?                                           │
│    YES ──► DISK — DiskCachedDataset  (neuromorphic default)          │
│                                                                      │
│  GPU pressure > 75 % and disk exists?                                │
│    YES ──► DISK  (CUDA pinned memory competes with RAM cache)        │
│                                                                      │
│  Fallback ──► NO_CACHE  (on-the-fly)                                 │
└──────────────┬───────────────────────────────────────────────────────┘
               │ Cached raw recordings  (events, target)
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  data_pipeline.py — NeuromorphicEncoder.apply_pipeline()             │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  CACHE LAYER  (always on raw recordings, BEFORE slicing)  │      │
│  │  wrap_dataset(raw_train, num_workers=N)                   │      │
│  └──────────────────────────┬────────────────────────────────┘      │
│                             │                                        │
│           use_temporal_slicing?                                      │
│               │                                                      │
│    YES ────────▼───────────────────────────────────────────          │
│  │  temporal_slicer.py — create_sliced_dataset()            │        │
│  │  Denoise → ToFrame → slice into T-ms windows             │        │
│  └──────────────────────────────────────────────────────────┘        │
│    NO  ────► transforms baked directly into cache wrapper            │
│                                                                      │
│  DataLoader (pin_memory=True, num_workers=N)                         │
└──────────────┬───────────────────────────────────────────────────────┘
               │ Pinned CPU tensors  [T, B, C, H, W]
               │ async non_blocking=True transfer ──►
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  GPU — SNN Training Loop                                             │
│                                                                      │
│  for t in range(T):                                                  │
│    spk = model(frame[t])           ← forward pass                   │
│                                                                      │
│  activity_reg.py — DenseTimestepBuffer  (one per hidden layer)       │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  push(spk)  ← forward hook per timestep                  │      │
│  │  stack()    ── [T, B, N] dense tensor for loss            │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                      │
│  loss = task_loss                                                    │
│       + activity_regularization(hidden)   ← dead/saturated neurons  │
│       + stdp_regularization(hidden, out)  ← causal spike ordering   │
│                                                                      │
│  loss.backward()   optimizer.step()                                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Path 2 — GPU-Only (Embedded / No-Disk Deployment)

Activated when `determine_strategy()` detects: no disk, insufficient RAM, VRAM ≥ 500 MB.

```
┌──────────────────────────────────────────────────────────────────────┐
│  RAW DATASET  (streaming source or in-memory Tonic dataset)          │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│  system_monitor.py — SystemResourceMonitor.snapshot()                │
│                                                                      │
│  disk_exists = False          available_ram_gb < threshold           │
│  gpu_available_gb ≥ 0.5 GB   ← triggers GPU-only cache path         │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  cache_engine.py — compute_gpu_cache_budget()                        │
│                                                                      │
│  VRAM ownership (lowest → highest priority):                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  [emergency 15%]─[params + activations + gradients + optim] │    │
│  │  [prefetch queue]──────────────[recording cache ← last]     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  cache_budget = min(                                                 │
│      free_vram × phase_cap,                                          │
│      free_vram − emergency_margin,                                   │
│      GPU_MAX_CACHE_GB  (2 GB absolute ceiling)                       │
│  )                                                                   │
│                                                                      │
│  Phase caps  (GPU_PHASE_CAPS):                                       │
│  ┌────────────┬───────────────────────────────────────────────┐     │
│  │  warmup    │  5 %  — activations not yet stable            │     │
│  │  train     │ 10 %  — backward pass competes for VRAM       │     │
│  │  backward  │  5 %  — most dangerous; minimize footprint    │     │
│  │  eval      │ 25 %  — no gradients or optimizer state       │     │
│  │  inference │ 30 %  — no backward pass at all               │     │
│  └────────────┴───────────────────────────────────────────────┘     │
│                                                                      │
│  GPURecordingCache                                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  prepare_item()  — moves tensor to CUDA on first access     │    │
│  │  set_phase(p)    — recalculates max_bytes + evicts to cap   │    │
│  │  evict_under_pressure()  — called every 50 accesses:        │    │
│  │    probe mem_get_info → if free < emergency_margin → evict  │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────┬───────────────────────────────────────────────────────┘
               │ VRAM-resident tensors
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  pipeline_coordinator.py — PipelineMemoryCoordinator                 │
│                                                                      │
│  worker_budget_gb < 0.5 GB  → GPU-only mode:                        │
│    num_workers        = 0                                            │
│    pin_memory         = False  (no CPU→GPU boundary)                │
│    persistent_workers = False                                        │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  data_pipeline.py — NeuromorphicEncoder                              │
│                                                                      │
│  Single-process DataLoader (no forking; tensors already on CUDA)     │
│  Temporal slicing recommended: slice before GPU caching              │
│  to keep individual cache entries small                              │
└──────────────┬───────────────────────────────────────────────────────┘
               │ CUDA tensors  (no H2D transfer)
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  GPU — SNN Training Loop                                             │
│                                                                      │
│  Training loop integration with phase-aware cache:                   │
│                                                                      │
│  cache.set_phase("train")                                            │
│  for batch in dataloader:                                            │
│      output = model(batch)                                           │
│      cache.set_phase("backward")  ← shrink budget before backward   │
│      loss.backward()                                                 │
│      optimizer.step()                                                │
│      cache.set_phase("train")     ← restore budget                  │
│                                                                      │
│  cache.set_phase("eval")          ← larger budget during evaluation  │
│  for batch in test_loader:                                           │
│      with torch.no_grad():                                           │
│          output = model(batch)                                       │
│                                                                      │
│  activity_reg.py — same as CPU+GPU path                              │
│  (DenseTimestepBuffer, activity_regularization, stdp_regularization) │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Invariants

| Rule | Where Enforced |
|------|---------------|
| Cache wraps **raw** recordings, never sliced datasets | `wrap_dataset()` raises `ValueError` if `slice_map` attribute detected |
| Cache budget ÷ `num_workers` in hybrid mode | `wrap_dataset(num_workers=N)` in `data_pipeline.py` |
| GPU VRAM: emergency 15 % always reserved | `compute_gpu_cache_budget()` |
| CUDA required — CPU-only is not a valid deployment | `PipelineMemoryCoordinator.__init__` raises `RuntimeError` if no CUDA |
| GPU pressure > 75 % → recording cache halved | `PipelineMemoryCoordinator.effective_cache_gb()` |
| GPU pressure > 75 % + disk exists → switch to disk cache | `AdaptiveCacheController.determine_strategy()` |
