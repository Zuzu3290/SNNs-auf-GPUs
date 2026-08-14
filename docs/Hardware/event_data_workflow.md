# Event Data Workflow — Architecture Reference

Graphical reference for the neuromorphic data pipeline implemented in
`event_data_workflow/`.

The project runs one hardware topology: data loading and caching always
run on CPU, training always runs on GPU. There is no separate CPU-only or
GPU-only deployment path — `AdaptiveCacheController` does not choose
between hardware configurations, it only chooses where one dataset's cache
lives: RAM, disk, or both. VRAM is never a cache storage target — the GPU
is only ever the training device, never a dataset storage location.

---

## File Map

| File | Responsibility |
|------|---------------|
| `system_monitor.py` | RAM / VRAM / disk probing (`SystemResourceMonitor`, `CacheMetrics`) |
| `cache_engine.py` | Strategy selection, `BoundedRecordingCache`, `AdaptiveCacheController` |
| `data_pipeline.py` | End-to-end assembly: raw → cache → slice → DataLoader (`NeuromorphicEncoder`), DataLoader worker sizing (`dataloader_config()`), and stateless temporal windowing (`create_sliced_dataset()`) |
| `prefetch.py` | Background-thread batch prefetching (`AsyncGPUPrefetcher`) |
| `learning/utilities.py` | Per-layer spike recording and activity regularization (`ActivityMonitor`, `DenseTimestepBuffer`) |

---

## Data Path

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
│  cache_engine.py — AdaptiveCacheController.determine_dataset_strategy│
│                                                                      │
│  Strategy decision tree (RAM/disk only — VRAM is never a candidate): │
│                                                                      │
│  GPU pressure > 75% and disk exists?                                 │
│    YES ──► DISK  (CUDA's pinned-memory allocator competes with       │
│                    RAM cache for the same physical RAM)             │
│                                                                      │
│  RAM available and dataset fits (< 70% of free)?                     │
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
│  Nothing fits ──► RuntimeError (system halt)                         │
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
│  │  data_pipeline.py — create_sliced_dataset()               │        │
│  │  Denoise → ToFrame → slice into T-ms windows             │        │
│  └──────────────────────────────────────────────────────────┘        │
│    NO  ────► transforms baked directly into cache wrapper            │
│                                                                      │
│  DataLoader (num_workers/pin_memory/prefetch_factor from             │
│  dataloader_config(), sized from live RAM; drops to num_workers=0    │
│  only when free RAM leaves under 500MB for worker processes)         │
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
│  utilities.py — ActivityMonitor (one per model, owns its buffers)    │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │  push(spk)  ← forward hook per timestep                  │      │
│  │  stack()    ── [T, B, N] dense tensor for loss            │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                      │
│  loss = task_loss                                                    │
│       + model.activity.regularization_loss()  ← dead/saturated      │
│                                                    neurons only      │
│                                                                      │
│  loss.backward()   optimizer.step()                                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Key Invariants

| Rule | Where Enforced |
|------|---------------|
| Cache wraps **raw** recordings, never sliced datasets | `determine_dataset_strategy()` raises `ValueError` if `slice_map` attribute detected |
| Cache budget ÷ `num_workers` in hybrid mode | `determine_dataset_strategy(num_workers=N)` in `cache_engine.py` |
| VRAM is never a cache storage target | `AdaptiveCacheController` only ever returns `MemoryCachedDataset` / `DiskCachedDataset` / `BoundedRecordingCache` |
| GPU pressure > 75% + disk exists → switch to disk cache | `AdaptiveCacheController.determine_dataset_strategy()` |
| DataLoader worker count sized from live RAM, not a fixed constant | `data_pipeline.dataloader_config()` |
