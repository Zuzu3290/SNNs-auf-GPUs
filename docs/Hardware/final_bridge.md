# Final Bridge — Structural Integration Record

Branch: `Operators`  
Date: 04 June 2026  
Scope: All architectural changes implemented across this session — kernel layer, runtime layer, config bridge, and diagnostic infrastructure.

---

## What the Bridge Connects

The project has four structural layers. Before this session they were partially disconnected — each layer had pieces of logic that overlapped or contradicted the others. The work in this session built the connective tissue between them.

```
src/learning/          ← application layer (training, inference)
event_data_workflow/   ← data pipeline (cache, prefetcher, spike buffers)
        ↓  ↑                   ↓  ↑
 ┌─────────────────────────────────────┐
 │           src/runtime/              │  ← NEW coordination layer
 │   MemoryArbiter   SpikeRateBus      │
 │   PhaseManager                      │
 └─────────────────────────────────────┘
        ↓  ↑                   ↓  ↑
src/crsc/              ← kernel dispatcher (engine.cu)
acceleration/          ← hardware execution (kernels, GPU attributes)
```

---

## Changes by Component

### 1. Kernel Layer — `src/crsc/` + `acceleration/`

**What existed:** A basic `lif_basic` kernel with a race condition (B×N×T threads all writing `voltage[b,n]`).

**What was built:**

| Kernel | File | Description |
|---|---|---|
| `lif_temporal_kernel` | `kernels/lif_temporal.cu` | T-in-register, correct sequential LIF. One thread per neuron, voltage held in register across all T steps. Eliminates race condition. Warp ballot per timestep. |
| `lif_warp_oriented_kernel` | `kernels/lif_warp_oriented.cu` | SM-saturating grid-stride. Launches `sm_count × target_bps` blocks regardless of B×N. Device properties cached once. Hot path fully async (no CPU-GPU sync). Separate timed variant for energy feedback sampling. |
| `snn_engine_dispatch` | `src/crsc/engine.cu` | Four-path dispatcher reading `KernelConfig`. Adaptive energy feedback loop adjusts `target_blocks_per_sm` every 50 calls based on sampled elapsed time. |

**On N — kernel layer width vs network structure:**
The kernel processes one LIF layer at a time. The tensor entering the kernel is `[B, N, T]` where `N` is the flattened spatial dimension of the input data, not the total neuron count of the network. The full network structure (`input → hidden layers → output`) is owned by the SNN framework (Norse, SpikingJelly, snn_torch) above the kernel. The kernel provides the LIF compute primitive for one layer; the framework calls it and handles the rest of the forward pass.

---

### 2. Runtime Coordination Layer — `src/runtime/`

**What existed:** Nothing. Two independent systems (pipeline coordinator and kernel workspace) competed for VRAM without coordination.

**What was built:**

#### `MemoryArbiter` (C++ + Python fallback)

- **C++ extension** (`acceleration/GPU_attributes/memory_arbiter.cu` + `src/runtime/runtime_binding.cpp`) — uses `cudaDeviceGetDefaultMemPool` to acquire the CUDA memory pool and sets `cudaMemPoolAttrReleaseThreshold` to pin the emergency zone floor. The pool never releases below that amount to the OS. Zone accounting uses `std::atomic<size_t>` — lock-free reads, CAS rollback on hard-limit breach.
- **Python fallback** (`src/runtime/memory_arbiter.py`) — identical API, no cuMemPool access. Used when `snn_runtime` extension is not compiled.

| Zone | Budget | Hard enforced |
|---|---|---|
| `dataset_cache` | 40% | No — cache eviction tolerable |
| `model_params` | 30% | Yes — must not bleed into kernel workspace |
| `kernel_workspace` | 20% | Yes — temporary kernel buffers |
| `emergency` | 10% | Yes — pinned as pool release threshold |

#### `SpikeRateBus` (`src/runtime/spike_rate_bus.py`)

EWMA singleton (α=0.1). Any component pushes once per batch via `push_dense()` (one `.item()` sync) or `push_count()` (zero sync). Any consumer reads `.rate`. Replaces the previous pattern where multiple subsystems each called `.item()` independently to measure spike rate.

#### `PhaseManager` (`src/runtime/phase_manager.py`)

Single source of truth for execution phase: `IDLE`, `WARMUP`, `TRAIN`, `BACKWARD`, `EVAL`. Set by the training loop; readable by any component without importing from `learning/`.

---

### 3. Config Bridge — `SNN_module.yaml` → `Settings` → kernel routing

**What existed:** `kernel: ON/OFF` in `SNN_module.yaml` controlled whether to use any custom kernel. `accel_config.yaml` existed as a separate file that nothing in the Python stack read. `training.py` always called `kernel.forward()` regardless of config.

**What was built:**

Two new fields in `SNN_module.yaml` under `training`:

```yaml
kernel_mode: warp_oriented    # basic | temporal | warp_oriented
kernel_profile_energy: false  # CUDA event energy timing (requires NVML)
```

`skeleton/snn_config.py` reads both as `cfg.KERNEL_MODE` and `cfg.KERNEL_PROFILE_ENERGY`. No second YAML file — `SNN_module.yaml` remains the single application config.

`training.py` and `inference.py` now route based on `cfg.KERNEL_MODE`:

```python
if self.kernel_mode == "warp_oriented":
    spikes, _, _ = kernel.warp_oriented_forward(...)
elif self.kernel_mode == "temporal":
    spikes = kernel.temporal_forward(...)
else:
    spikes = kernel.forward(...)
```

The 100x kernel is now reachable from the full training loop, not just the Colab benchmark.

---

### 4. Compatibility Reporting — `_report_kernel_compatibility()`

**What existed:** Nothing. The kernel would silently fail or produce wrong output if the architecture was incompatible.

**What was built:** An inspection method that runs at `SNNTrainer` init when `kernel: ON`. It checks the pressed architecture against the kernel's contract and prints a labelled report. Does not disable the kernel — reports only.

Checks performed:
- T = `cfg.TIMESTEPS` > 0
- N = `cfg.INPUT_SIZE` > 0 (kernel layer width, not total neurons — full structure shown alongside)
- CUDA device available
- Model class name against known SNN markers (`snn`, `lif`, `norse`, `spiking`, etc.)
- Required kernel function present in the built module for the selected mode

If a new architecture's class name doesn't match a known SNN marker, the report flags it — the kernel still runs, the user sees it.

---

### 5. Hot-Path Sync Fix — `DenseTimestepBuffer`

**What existed:** `firing_rate` and `num_spikes` properties in `pipeline_coordinator.py` called `.item()` inside a Python loop over all T stored tensors — T separate CPU-GPU syncs per diagnostic read.

**What was changed:** Both replaced with `torch.stack(self.events).sum().item()` — one sync regardless of T.

`PipelineMemoryCoordinator.effective_cache_gb()` now consults `MemoryArbiter` for the `dataset_cache` zone budget instead of independently querying `SystemResourceMonitor` — single authoritative VRAM budget decision.

---

## Benchmark Results (04 June 2026, Colab T4)

| Metric | Value |
|---|---|
| Kernel time (GPU events) | 0.035 ms |
| Warp-oriented wall time | 0.028 ms / forward |
| PyTorch baseline wall time | 2.785 ms / forward |
| Wall-time speedup | 100.22x |
| Spike rate | 0.3% |
| Blocks launched | 8 of 320 possible (under-saturated at B=4, N=512) |
| Estimated energy per forward (custom) | ~0.91 mJ |
| Estimated energy per forward (baseline) | ~44 mJ |
| Energy reduction | ~48x |

Speedup at production scale (B=32, N=4096) estimated at **4–8x** as baseline becomes compute-bound.

---

## What Remains

| Item | Status |
|---|---|
| Cell 8 training run | Fails — `from runtime import ...` not pulled into Colab environment yet |
| `snn_runtime` Colab build | Cell 4b available but optional |
| Sparse downstream matmul (ballot mask → skip silent rows) | Not implemented — next frontier |
| float4 vectorised input loads | Not implemented — memory bandwidth ceiling at production scale |
| NVML energy readings | Unavailable on free T4 tier — guarded by `SNN_HAS_NVML=0` |
