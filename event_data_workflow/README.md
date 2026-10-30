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
