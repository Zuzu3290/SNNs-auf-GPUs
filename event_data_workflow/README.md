# event_data_workflow

Pipeline for loading, caching, and slicing neuromorphic event data before training.

## Components

- **cache_engine.py** — Adaptive cache controller. Selects between RAM, disk, hybrid, GPU VRAM, or no-cache strategy based on live system resources.
- **temporal_slicer.py** — Slices raw recordings into fixed-length temporal windows for SNN input.
- **data_pipeline.py** — Wires the cache and slicer together into a single DataLoader-ready pipeline.

## Correct Usage Order

Cache must be applied to raw recordings **before** temporal slicing:

```python
cached_raw = cache_controller.wrap_dataset(raw_dataset)   # cache first
sliced     = TemporalSlicedDataset(cached_raw, config)     # slice second
```

Applying the cache after slicing will raise a `ValueError`. The reason is efficiency:
one recording produces N temporal slices, so caching the recording once gives N cache
hits from a single stored entry.

## Known Limitation — Single GPU Only

`PipelineMemoryCoordinator` tracks memory pressure for **one specific GPU** (whichever
`device` index it was initialised with). On a multi-GPU machine (e.g. 4 GPUs), each
coordinator only sees its own device — it has no visibility into VRAM usage on other
GPUs. If the pipeline is scaled to multi-GPU training (`DataParallel` or
`DistributedDataParallel`), a separate coordinator is needed per device, or the class
must be extended to aggregate pressure across all device indices.
