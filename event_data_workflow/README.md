# event_data_workflow

Pipeline for loading, caching, and slicing neuromorphic event data before training.
See [`docs/event_data_workflow/caching_pipeline_refactor.md`](../docs/event_data_workflow/caching_pipeline_refactor.md)
for the full design report.

## Components

- **cache_engine.py** — `AdaptiveCacheController`. Selects between RAM or disk cache strategy based on live system resources. VRAM is never a cache target — the GPU is only ever the training device, never a dataset storage location.
- **system_monitor.py** — `SystemResourceMonitor` (shared instance: `monitor`). RAM/disk/VRAM probing, gated by an explicit `cuda_enabled` flag so a GPU that isn't requested never influences decisions; also holds `dataloader_config()` (worker sizing, physical-core-based). Also `PipelineMonitor` — background-thread CPU/GPU utilization/power/memory sampling, used both by the offline diagnostics harness and by `SNNTrainer`/`SNNTester` for per-epoch reporting.
- **data_pipeline.py** — `NeuromorphicEncoder`. Wires cache and slicing together into a DataLoader-ready pipeline; also holds `compute_prefetch_depth()` (live VRAM-based prefetch sizing), `create_sliced_dataset()` (temporal windowing), and `PrefetchedLoader` (background-thread CPU prefetch + CUDA-stream H2D overlap, wrapped around the DataLoaders `create_loaders()` builds).

## Correct Usage Order

Cache must be applied to raw recordings **before** temporal slicing:

```python
controller = AdaptiveCacheController()
cached_raw = controller.determine_dataset_strategy(raw_dataset, transform=frame_tf, split="train")
sliced     = create_sliced_dataset(cached_raw, slice_duration_ms=15.0)
```

Applying the cache after slicing raises a `ValueError`. The reason is efficiency:
one recording produces N temporal slices, so caching the recording once gives N cache
hits from a single stored entry.

## Known Limitation — Single GPU Only

This pipeline only ever addresses GPU 0 — `training.device` in `SNN_module.yaml`
is `cpu | cuda | auto`, never an indexed `cuda:N`, so there's no multi-GPU
selection anywhere in `SystemResourceMonitor`/`PipelineMonitor`/`AdaptiveCacheController`.
If the pipeline is ever scaled to multi-GPU training (`DataParallel` or
`DistributedDataParallel`), this would need a real device-index parameter
reintroduced, plus a monitor instance per device.
