# event_data_workflow

Turns event recordings into the frame tensors the network trains on.

| File | Role |
|---|---|
| `dataset_registry.py` | One entry per dataset: loader, sensor size, number of classes |
| `data_pipeline.py` | `NeuromorphicEncoder`: loads a dataset, denoises it, cuts each recording into T time bins, builds the DataLoaders; `DeviceLoader` hands batches to the GPU |
| `cache_engine.py` | Caches the finished frames in RAM or on disk, chosen from the free resources, so framing runs once |
| `system_monitor.py` | Reads RAM, disk, VRAM and CPU cores for the worker and cache decisions; `PipelineMonitor` samples GPU power and utilisation during a run |
| `fast_preprocessing.py` | A Numba denoising filter from an earlier speed test; not used by the pipeline, which uses Tonic's `Denoise` |

The CPU prepares the next batches while the GPU computes the current one; batches are copied from
pinned memory without blocking.

## The cache engine

**Why it exists.** An event camera records a list of individual events, not images. Before the
network can use a recording, it is denoised and cut into T time bins (frames), which takes about
136 ms per recording. Repeated every epoch, that dominates a run: N-MNIST has 60,000 training
recordings, so roughly 2.3 hours of CPU work per epoch. The cache engine does this work once and
keeps the finished frames, so every later epoch, and with the disk cache every later run, framework
and seed, reads ready-made frames instead.

**What it decides.** At the start of a run, `AdaptiveCacheController` frames a few samples, estimates
the size of the whole framed dataset and picks one place to keep it for the train split and one for
the test split:

- **RAM** (tonic's `MemoryCachedDataset`), if it fits comfortably into the free memory. Each DataLoader
  worker holds its own copy, so the estimate is multiplied by the number of workers. Fastest to read,
  but gone when the run ends.
- **Disk** (tonic's `DiskCachedDataset`), otherwise, as files under `cache/`. Built on first use and
  reused by every later run with the same settings.
- If neither fits, the run stops with the free RAM and disk space it found, instead of failing
  halfway through an epoch.

The choice is made once per run; data does not move between RAM and disk while the run goes on.

**What it guarantees.**

- **Finished frames are cached, not raw events.** Tonic's cache classes on their own would repeat the
  framing on every read; `PreTransformedDataset` frames each sample before it reaches the cache, so
  framing really happens only once.
- **Settings never mix.** Each combination of framing settings gets its own folder, named after them:
  `cache/<dataset>/bins16_dn10000/` holds 16 time bins with a 10,000 µs denoise filter, and changing
  either setting builds a new folder rather than reusing frames made the old way.
- **Augmentation stays live.** When `augmentation.enabled` is on, the random rotation is applied after
  the cache, on every read, so each epoch sees new rotations while the cached frames stay unchanged.
  The test split is never augmented.
- **Empty recordings batch correctly.** `FixedToFrame` corrects a tonic quirk where an empty recording
  comes back with height and width swapped, which would otherwise break batching on non-square
  sensors such as N-Caltech101.

**Tuning and resetting.** The RAM/disk decision uses four keys under `resource_policy:` in
`configuration/data_workflow.yaml`: `memory_safety_margin_gb` (RAM always left free),
`memory_cache_threshold_gb` (minimum free RAM before RAM caching is considered),
`memory_tier_headroom_fraction` (share of the remaining RAM the cache may fill) and
`disk_tier_headroom_multiple` (free disk needed, as a multiple of the dataset size). `cache.path`
moves the disk cache, for example onto Google Drive on Colab. To rebuild a disk cache, delete its
folder; the next run creates it again.

**Where it could go next.** VRAM is never used for the cache, because the GPU memory is reserved for
the model and fitted by the batch-size calibration. Keeping a small dataset in VRAM, as a third choice
next to RAM and disk, would be the natural extension; the calibration would then have to count it
against the GPU budget.
