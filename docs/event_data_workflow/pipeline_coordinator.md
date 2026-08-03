## pipeline_coordinator.py + temporal_slicer.py (removed)

Both files were removed in the caching pipeline refactor — see
[`caching_pipeline_refactor.md`](./caching_pipeline_refactor.md) for the
full report.

Where their contents live now:

| Was | Now |
|---|---|
| `pipeline_coordinator.dataloader_config()` | `data_pipeline.py` |
| `pipeline_coordinator.DenseTimestepBuffer` | `src/learning/frameworks/activity_reg.py` |
| `pipeline_coordinator.AsyncGPUPrefetcher` | `event_data_workflow/prefetch.py` |
| `temporal_slicer.create_sliced_dataset()` | `data_pipeline.py` |
| `temporal_slicer.AdaptiveTemporalSlicer` | removed entirely (see report — solved a problem this project doesn't have) |
