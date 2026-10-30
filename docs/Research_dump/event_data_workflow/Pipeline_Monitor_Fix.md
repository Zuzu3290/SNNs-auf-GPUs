# Pipeline monitor: false-positive `gpu_idle` warning, and a toggle

## Symptom

```
WARNING:event_data_workflow.system_monitor:[PIPELINE MONITOR] gpu_idle sustained
for 5.1s (phase='unset', started at t=25.4s)
```

## Root cause

`PipelineMonitor` (`event_data_workflow/system_monitor.py`) starts sampling
CPU/GPU utilization in a background thread at `pm.start()`, but the phase
label stays `"unset"` until the first `pm.set_phase(...)` call — which only
happens once the first training epoch (or test run) actually begins
(`learning/training.py`, `learning/inference.py`). Everything before that —
dataset resolution, cache warm-up, `NeuromorphicEncoder` building the
DataLoaders — is CPU-bound with no GPU work scheduled, so the GPU is
correctly idle. The `gpu_idle` sustained-violation check didn't know the
difference, so it fired on this normal startup window and logged it as a
warning.

**The background sampler itself was never the problem.** It only calls
NVML/psutil driver queries (`nvmlDeviceGetUtilizationRates`,
`nvmlDeviceGetPowerUsage`, `psutil.cpu_percent`) — it never touches a CUDA
tensor and never calls `torch.cuda.synchronize()`, so it doesn't block or
serialize against the training loop it's watching. What actually happens is
a lightweight daemon thread waking up every 0.2s to read driver counters; it
doesn't force the GPU to stop and hand anything to the CPU.

## Fix

`sample_once()` now skips the `gpu_idle` check while `phase == "unset"`:

```python
self.check_bound("gpu_idle", self.phase != "unset" and gpu_util is not None
                   and gpu_util < self.thresholds.gpu_idle_pct_below,
                   t, self.thresholds.gpu_idle_sustained_s)
```

Idle time during real setup is no longer flagged. Once a real phase is
entered (`epoch_0`, `test_run`, ...), `gpu_idle` is checked as before —
useful there, since it can catch a genuine data-loading bottleneck
mid-training.

## Toggle

Added `cfg.ENABLE_PIPELINE_MONITOR` (default `True`, `configuration/SNN_module.yaml`
→ `training.enable_pipeline_monitor`), threaded into `PipelineMonitor(enabled=...)`
in both `SNNTrainer` and `SNNTester`. When `False`, `start()` and
`measure_idle_baseline()` no-op — no background thread, no idle-power probe,
no warnings — while the point-in-time GPU memory stats used elsewhere
(`torch.cuda.max_memory_allocated`) are unaffected, since those don't depend
on the background thread.

Set it directly in `learning/main.py`, next to the existing
`RUN_ADVERSARIAL_EVAL` / `RUN_REALTIME_EVAL` toggles:

```python
cfg.ENABLE_PIPELINE_MONITOR = True  # background CPU/GPU utilization + power sampling; set False to disable
```

## Note for the report

With the monitor on (the default), it costs nothing measurable — a 0.2s-interval
read-only NVML/psutil poll on its own thread, never touching CUDA. With it off,
GPU utilization%, power draw, and energy-per-epoch all report as `0.0` instead of
being measured — GPU memory and every training metric (loss, accuracy, spike
rate) stay real and correct either way, since they don't depend on this monitor.
