# event_data_workflow

Pipeline for loading, caching, and slicing neuromorphic event data before training.
See [`docs/event_data_workflow/caching_pipeline_refactor.md`](../docs/event_data_workflow/caching_pipeline_refactor.md)
for the full design report.

## Components

- **cache_engine.py** — `AdaptiveCacheController`. Selects between RAM, disk, hybrid, GPU VRAM, or no-cache strategy based on live system resources.
- **system_monitor.py** — `SystemResourceMonitor`. RAM/disk/VRAM probing, gated by an explicit `cuda_enabled` flag so a GPU that isn't requested never influences decisions.
- **data_pipeline.py** — `NeuromorphicEncoder`. Wires cache and slicing together into a DataLoader-ready pipeline; also holds `dataloader_config()` (worker sizing) and `create_sliced_dataset()` (temporal windowing) — both folded in from since-removed `pipeline_coordinator.py`/`temporal_slicer.py`.
- **prefetch.py** — `AsyncGPUPrefetcher`. Background-thread double-buffering so the GPU doesn't idle waiting on the CPU to prepare the next batch.

## Correct Usage Order

Cache must be applied to raw recordings **before** temporal slicing:

```python
controller = AdaptiveCacheController(device=torch.device(cfg.DEVICE))
cached_raw = controller.determine_dataset_strategy(raw_dataset, transform=frame_tf, split="train")
sliced     = create_sliced_dataset(cached_raw, slice_duration_ms=15.0)
```

Applying the cache after slicing raises a `ValueError`. The reason is efficiency:
one recording produces N temporal slices, so caching the recording once gives N cache
hits from a single stored entry.

## Known Limitation — Single GPU Only

`SystemResourceMonitor` (and everything built on it — `AdaptiveCacheController`,
`dataloader_config()`) tracks memory pressure for **one specific GPU** (whichever
`device_idx` it was initialised with). On a multi-GPU machine, each instance only
sees its own device — no visibility into VRAM usage on other GPUs. If the pipeline
is scaled to multi-GPU training (`DataParallel` or `DistributedDataParallel`), this
would need a separate instance per device, or extending to aggregate pressure
across all device indices.
