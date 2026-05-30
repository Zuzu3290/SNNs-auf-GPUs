## pipeline_coordinator.py + temporal_slicer.py

These two files handle the runtime memory management and data preparation layers of the neuromorphic pipeline. They sit between raw dataset loading and the training loop.

---

### PipelineMemoryBudget

A simple dataclass that splits the available RAM across three pipeline stages:

- **recording_cache_fraction (40%)** — RAM allocated to BoundedRecordingCache (the hot layer that keeps raw recordings in memory so disk is not re-read every batch)
- **worker_fraction (30%)** — RAM reserved for DataLoader worker subprocesses
- **prefetch_fraction (30%)** — RAM reserved for the async GPU prefetch queue

The total is derived from live system RAM minus a safety margin at startup.

---

### PipelineMemoryCoordinator

The coordinator enforces the budget above and answers questions that the rest of the pipeline asks before it starts running.

**What it does:**

`from_system()` — class method that probes live RAM and GPU state via SystemResourceMonitor, then builds the budget automatically. This is the only constructor you should call in practice.

`effective_cache_gb()` — returns how many GB the recording cache is allowed to use right now. If GPU VRAM usage exceeds 75%, this is halved automatically — because CUDA pinned memory and the recording cache both draw from the same physical RAM, letting both grow unchecked causes OOM.

`dataloader_config()` — recommends `num_workers`, `prefetch_factor`, `pin_memory`, and `persistent_workers` based on the RAM worker budget. If the worker RAM budget is under 500 MB (GPU-only embedded scenario), it returns `num_workers=0` and disables pin_memory since there is no CPU-to-GPU transfer boundary to optimise.

`prefetch_queue_size()` — returns 1 under GPU pressure, 2 otherwise.

**Important:** tracks a single GPU only. For multi-GPU setups, create one coordinator per device.

---

### DenseTimestepBuffer

A per-timestep spike accumulation buffer used during the SNN forward pass.

The SNN processes data timestep by timestep. After each timestep, the spike tensor is pushed into this buffer with `push(spk)`. When the full forward pass is done, `stack()` reconstructs the full `[T, B, ...]` tensor in a single `torch.stack` call — avoiding repeated CUDA syncs in the hot loop.

`clear()` resets the buffer between forward passes.

Useful properties: `num_spikes`, `num_timesteps`, `memory_bytes`, `firing_rate` (fraction of neurons that fired — for diagnostics, note that `firing_rate` forces a CUDA sync per timestep so do not call it in the training hot loop).

---

### temporal_slicer.py

DVS event cameras produce continuous streams of events. A single recording may contain millions of events spanning hundreds of milliseconds. Feeding a full raw recording into the SNN at once is impractical — the network expects fixed-length temporal windows. The temporal slicer breaks each recording into smaller windows before the training loop sees them.

**create_sliced_dataset()** — the main function. Wraps a cached dataset with tonic's `SlicedDataset`. Two slicing modes:

- **Time-based** (default): each slice covers a fixed time window (e.g. 15 ms). Controlled by `slice_duration_ms` and `overlap_ms` in `data_workflow.yaml`.
- **Event-count-based**: each slice contains exactly N events regardless of time. Activated by setting `events_per_slice` in the config.

If `metadata_path` is provided, tonic stores the slice index as HDF5 so it is not recomputed on subsequent runs — important for large datasets.

**AdaptiveTemporalSlicer** — analyses the dataset and recommends a slice duration automatically when `auto_tune: true` is set in `data_workflow.yaml`. It samples 50 recordings, measures their duration and event rate, and suggests a window size that produces roughly 2 slices per recording with at least 50 events per slice. If the suggested window is too short to contain enough events, it adjusts upward automatically.

**Why slicing happens after caching:**
The cache wraps raw recordings. Slicing wraps the cached dataset. This ordering means N slices from the same recording all hit the same single cache entry — the recording is loaded from disk once, stored once in RAM, and served N times. If slicing happened before caching, each slice would be cached independently, multiplying memory usage by the number of slices per recording.

---

### How they connect

```
SystemResourceMonitor
        │
        ▼
PipelineMemoryCoordinator ──► DataLoader config (num_workers, pin_memory, ...)
        │
        ▼
AdaptiveCacheController
        │
        ├── BoundedRecordingCache  (hybrid: RAM hot layer)
        ├── DiskCachedDataset      (disk mode)
        ├── MemoryCachedDataset    (memory mode)
        └── GPURecordingCache      (gpu_memory mode)
                │
                ▼
        create_sliced_dataset  ──► tonic.SlicedDataset  [T windows per recording]
                │
                ▼
           DataLoader  ──►  Training loop
```
