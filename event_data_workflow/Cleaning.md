1. Confirmed dead — computed but never read by anyone:

CacheMetrics.cpu_percent and .cpu_count (system_monitor.py) — populated on every SystemResourceMonitor.snapshot() call (a real psutil.cpu_percent()/cpu_count() cost paid each time), but no caller anywhere in the repo ever reads metrics.cpu_percent or metrics.cpu_count. cache_engine.py's log_diagnostics() doesn't print CPU at all.
GPUStats.end_epoch()'s "gpu_mem_curr_gb" dict key — computed every epoch (torch.cuda.memory_allocated()), returned in the dict, never read by training.py or inference.py.
dataloader_config()'s batch_bytes parameter — the one call site (data_pipeline.py:307) never passes it, so it's always 0, so the if batch_bytes > 0: max_workers = worker_bytes / (2*batch_bytes) branch never runs in practice. Worker count always falls through to settings.NUM_WORKERS.
2. Scattered across files — your instinct that things should live in system_monitor.py looks right:

There are now three independent SystemResourceMonitor instances, each probing the same live RAM/GPU state separately: one in AdaptiveCacheController.__init__, one built fresh inside every dataloader_config() call, and one I just added inside SNNTrainer/SNNTester for GPUPhaseMonitor last turn. No sharing.
GPUStats and SystemResourceMonitor each independently query torch.cuda.get_device_properties(...).total_memory for total VRAM — same fact, two separate code paths.
PipelineMonitor/BoundThresholds/BoundViolation/PipelineSample (continuous background sampler, phase-tagging, sustained-violation flagging) live inside system_monitor.py, but are only ever used by diagnostics/gpu_utilization_harness.py — a standalone offline script, not the real training/inference pipeline at all. It's a second, mostly self-contained monitoring subsystem bundled into the same file as the pipeline's actual live cache/worker probing.
GPUStats (per-epoch/overall utilization + memory, via NVML/PyTorch counters) and the new GPUPhaseMonitor (free-VRAM-vs-margin, via SystemResourceMonitor) both report GPU memory state, through separate objects, both built independently in SNNTrainer and SNNTester.
3. Inconsistent, not dead — asymmetric usage worth a decision:

GPUStats.measure_idle_baseline()/dynamic_power_w() (idle-power subtraction) is called in inference.py but never in training.py — training always reports raw power draw, inference reports idle-subtracted "dynamic" power.
dataloader_config() has its own hardcoded safety_margin_gb=2.0 default, separate from AdaptiveCacheController's YAML-configurable memory_safety_margin_gb — two independent "reserve 2GB" constants, only one of which is actually tunable via config.
No code touched this round. Your ca