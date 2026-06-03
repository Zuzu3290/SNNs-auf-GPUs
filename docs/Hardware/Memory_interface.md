Contradiction 1 — Two VRAM managers with no shared state
pipeline_coordinator.py manages VRAM for the data pipeline:


GPU_PRESSURE_THRESHOLD = 0.75   # halves cache when VRAM > 75%
memory_management.cu audits VRAM for the kernel:


memory_warn_threshold_mb: 512   # warns kernel when free VRAM < 512MB
Both call cudaMemGetInfo (or the Python equivalent) independently. Neither knows what the other has reserved. The pipeline coordinator can allocate 2GB of GPU recording cache, and the kernel's snn_query_memory will then see only what's left — but won't know why it's low or that the pipeline is responsible.

Contradiction 2 — Spike rate measured twice, from opposite sides
DenseTimestepBuffer.firing_rate in pipeline_coordinator.py:


fired = int(sum(e.sum().item() for e in self.events))  # .item() forces a CUDA sync per timestep
Measured on CPU, causes a GPU sync on every timestep.

__ballot_sync in lif_temporal.cu / lif_warp_oriented.cu:


const uint32_t ballot = __ballot_sync(active_mask, fired);
Measured on GPU, zero overhead, already computed — but immediately discarded. The training loop never sees it.

They measure the same quantity. One blocks the CPU. The other is free but unused.

Contradiction 3 — Phase-aware cache vs phase-blind kernel
cache_engine.py reduces VRAM aggressively during backward:


GPU_PHASE_CAPS = {
    "backward":  0.05,   # 5% cap — most dangerous moment
    "eval":      0.25,   # relaxed
}
The kernel dispatcher (engine.cu) launches with the same target_blocks_per_sm=8 regardless of phase. During backward, when VRAM is at 95% capacity and gradients are competing, the kernel doesn't back off — it launches the same way it does during inference.

Contradiction 4 — Pinned memory allocated by both independently
pipeline_coordinator.py sets pin_memory=True → PyTorch's DataLoader allocates page-locked RAM for async H2D transfers.

memory_management.cu has snn_malloc_pinned → also allocates page-locked RAM for kernel-side staging.

Pinned memory is a finite system resource. Two systems allocating it without coordination can silently cause OOM on the host side, which surfaces as a confusing CUDA error during training.

Contradiction 5 — Event-to-spike conversion happens twice
event_data_workflow/data_pipeline.py → Tonic transforms convert DVS events to frame tensors on CPU.

acceleration/spike_kernel.py → takes those frame tensors and runs a CUDA kernel to threshold/integrate them into spikes.

These are sequential, not redundant — but the boundary between them is implicit. There is no contract specifying what format the EDW hands off to the spike kernel. If someone changes the Tonic transform (e.g., switches from ToFrame to EventCount), the spike kernel receives unexpected input silently.

The fundamental contradiction

event_data_workflow    = software memory manager (Python, RAM-first view)
acceleration/          = hardware memory manager (CUDA, VRAM-first view)
They govern overlapping territory with different thresholds, different instruments, and no shared protocol. The EDW assumes it owns the VRAM budget decision. The acceleration layer assumes it owns the VRAM budget decision.

What resolves it
A single unified memory arbiter that both register with at startup:


MemoryArbiter
  ├── dataset_cache_gb   → issued to PipelineMemoryCoordinator
  ├── kernel_workspace_gb → issued to snn_query_memory / snn_malloc_*
  ├── model_params_gb    → reserved for PyTorch
  └── emergency_gb       → never touched
And a spike rate bus — the ballot mask from the kernel fed back into the training loop so DenseTimestepBuffer and activity regularization read from the same source instead of re-measuring it independently.

Both of these are missing. The two systems are individually well-built but architecturally isolated. 

Concept 1 — Memory Arbiter: Static Partition + Lease Model
Proven in: Linux kernel memory zones, jemalloc arena allocators, NVIDIA CUDA Multi-Process Service (MPS)

The core idea: at startup, before any component allocates anything, a single arbiter queries total VRAM and partitions it into named zones with hard and soft limits. Each consumer requests a lease from the arbiter. The arbiter grants or denies based on remaining budget. No component can allocate outside its zone.


VRAM (e.g. 15.6 GB)
┌─────────────────┬──────────────┬────────────────┬──────────────┐
│  dataset_cache  │ model_params │ kernel_workspace│   emergency  │
│   (soft limit)  │ (hard limit) │  (hard limit)  │  (reserved)  │
│    2.5 GB       │   4.0 GB     │    1.0 GB      │    0.5 GB    │
└─────────────────┴──────────────┴────────────────┴──────────────┘
         ↑                ↑                ↑
  PipelineMemoryCoordinator  PyTorch    snn_query_memory
Soft limit (dataset_cache): can shrink under pressure — the arbiter evicts cache pages when the kernel zone needs headroom. This is exactly how Linux cgroups memory.soft_limit_in_bytes works.

Hard limit (kernel_workspace, model_params): never exceeded. If a consumer requests beyond its hard limit, the arbiter returns an error rather than letting it silently steal from another zone.

Why this resolves the contradiction: both PipelineMemoryCoordinator and snn_query_memory stop querying cudaMemGetInfo independently. They both query the arbiter instead, which holds the one authoritative view of the budget.

Concept 2 — Spike Rate Bus: EWMA Gauge + Publish-Subscribe
Proven in: Prometheus gauge metric pattern, LMAX Disruptor ring buffer, exponential moving average (signal processing)

The ballot mask from __ballot_sync is already computing popcount(mask) / 32 — the warp-level spike rate — on every timestep, at zero cost. Currently that value is discarded. The concept is to route it through a gauge — a single shared scalar that the kernel writes and any observer reads without synchronisation.

The gauge uses EWMA (Exponentially Weighted Moving Average) to smooth across timesteps:


rate_ema(t) = α × popcount(ballot_t)/32 + (1 - α) × rate_ema(t-1)
With α = 0.1 (slow decay), this gives a stable signal that doesn't thrash on individual noisy timesteps. α = 0.1 is the standard value used in PyTorch's batch norm momentum and Adam's β parameters — already calibrated for ML workloads.

Subscribers to the bus:

DenseTimestepBuffer — reads spike rate instead of computing .item() per step (eliminates the CUDA sync)
dispatch_warp_oriented in engine.cu — adapts target_blocks_per_sm from spike density instead of from elapsed_ms alone
Activity regulariser in training.py — reads the bus instead of summing the dense buffer
PipelineMemoryCoordinator — at low spike rate, the GPU is lightly loaded → can expand dataset cache

Kernel (GPU)                    SpikeRateBus (Python singleton)
  ballot_sync every t       →   rate_ema updated after each batch
  popcount → warp spike rate →   consumed by: DenseTimestepBuffer
                                              energy feedback
                                              cache coordinator
                                              activity regulariser
Why this resolves the contradiction: the ballot mask is computed once on the GPU at kernel speed. Every Python-side consumer reads the smoothed scalar — no repeated CUDA syncs, no double measurement, one source of truth.

What I will build
Component	File	Based on
MemoryArbiter	acceleration/memory_arbiter.py	Linux cgroup soft/hard limits
Zone dataclass	same	jemalloc arena model
SpikeRateBus	acceleration/spike_rate_bus.py	Prometheus gauge + EWMA
Arbiter registration hook	src/crsc/engine.cu	MPS lease pattern
Bus write in kernel	lif_warp_oriented.cu	ballot popcount → gauge
Bus reads	pipeline_coordinator.py, training.py	pub-sub observer
