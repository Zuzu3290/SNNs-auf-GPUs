# Kernel & Acceleration — Architecture Reference

Documents the custom CUDA kernel stack and the two-path execution model
implemented across `src/crsc/` and `acceleration/` on the **Operators** branch.
The objective is scalable SNN inference performance through a configurable
acceleration layer that spans energy, memory, and throughput optimisation.

---

## File Map

| File | Responsibility |
|------|----------------|
| `acceleration/kernel_config.h` | Shared `KernelConfig` struct; `KERNEL_CONFIG_DEFAULT` and `KERNEL_CONFIG_ACCELERATED` presets |
| `acceleration/accel_config.yaml` | Python-readable mirror of `KernelConfig`; flip `accelerate: true` to switch paths |
| `src/crsc/engine.cu` | Dispatcher: routes to basic or accelerated path based on `KernelConfig` |
| `src/crsc/kernels/snn_forward.cu` | Standard LIF kernel + profiled variant (uses all GPU attributes) |
| `acceleration/Kernel.cu` | Standalone PyTorch-callable LIF kernel with full GPU-attribute integration |
| `acceleration/GPU_attributes/energy_management.{cu,h}` | NVML-backed `EnergyProfiler`; reports elapsed time, power, and energy per launch |
| `acceleration/GPU_attributes/memory_management.{cu,h}` | `snn_malloc_device`, `snn_malloc_pinned`, `snn_query_memory` |
| `acceleration/GPU_attributes/throughput_optimization.{cu,h}` | `compute_1d_launch` via `cudaOccupancyMaxPotentialBlockSize`; `warp_aligned_block` |
| `acceleration/backend/launch_wrappers.cu` | `ProfiledLaunchResult` — combined energy + launch config result struct |
| `acceleration/spike_kernel.py` | Python-side CuPy / PyTorch kernel backends for event → spike conversion |

---

## Execution Paths

### Path 1 — Standard (accelerate: false)

No GPU-attribute overhead. Fixed 256-thread block. Zero profiling cost.

```
Python / PyTorch extension call
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  src/crsc/engine.cu — snn_engine_dispatch()                          │
│                                                                      │
│  KernelConfig.accelerate == false                                    │
│                                                                      │
│  dispatch_basic()                                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  lif_basic<<<grid, 256, 0, stream>>>                        │    │
│  │                                                             │    │
│  │  grid = ceil(B × N × T / 256)                              │    │
│  │                                                             │    │
│  │  One thread per work item (b, n, t):                        │    │
│  │    v_new = v_prev × (1 − τ⁻¹) + input[b,n,t]              │    │
│  │    spike  = (v_new ≥ v_th) ? 1 : 0                         │    │
│  │    if spike: v_new = 0   (hard reset)                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Returns: spikes [B, N, T]                                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Path 2 — Accelerated (accelerate: true)

Three GPU-attribute modules applied in sequence before and around the launch.

```
Python / PyTorch extension call
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  src/crsc/engine.cu — snn_engine_dispatch()                          │
│                                                                      │
│  KernelConfig.accelerate == true                                     │
│                                                                      │
│  dispatch_accelerated()                                              │
│                                                                      │
│  ┌── 1. MEMORY ────────────────────────────────────────────────┐    │
│  │  optimize_memory == true                                    │    │
│  │  snn_query_memory(&free_bytes, &total_bytes)                │    │
│  │  Warn if free < 2 × sizeof(output spikes tensor)           │    │
│  │  (threshold configurable via memory_warn_threshold_mb)     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                │                                                     │
│                ▼                                                     │
│  ┌── 2. THROUGHPUT ────────────────────────────────────────────┐    │
│  │  optimize_throughput == true                                │    │
│  │  compute_1d_launch(lif_basic, B×N×T)                       │    │
│  │    └─ cudaOccupancyMaxPotentialBlockSize                    │    │
│  │         → block_size that maximises SM occupancy           │    │
│  │         → grid_size = ceil(total / block_size)             │    │
│  │  Reports: block, grid, theoretical_occupancy %             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                │                                                     │
│                ▼                                                     │
│  ┌── 3. ENERGY (start) ────────────────────────────────────────┐    │
│  │  profile_energy == true                                     │    │
│  │  EnergyProfiler.start(stream)                              │    │
│  │    ├─ cudaEventRecord(ev_start)                            │    │
│  │    └─ nvmlDeviceGetPowerUsage → power_before_mw            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                │                                                     │
│                ▼                                                     │
│  ┌── KERNEL LAUNCH ───────────────────────────────────────────┐    │
│  │  lif_basic<<<grid_size, block_size, 0, stream>>>           │    │
│  │  (same LIF logic as Path 1; launch params are tuned)       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                │                                                     │
│                ▼                                                     │
│  ┌── 4. ENERGY (stop) ─────────────────────────────────────────┐    │
│  │  EnergyProfiler.stop(stream)                               │    │
│  │    ├─ cudaEventRecord(ev_stop) → cudaEventSynchronize      │    │
│  │    ├─ cudaEventElapsedTime     → elapsed_ms                │    │
│  │    └─ nvmlDeviceGetPowerUsage  → power_after_mw            │    │
│  │  energy_mj = avg_power_W × elapsed_s × 1000               │    │
│  │  Prints full report to stdout                              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Returns: spikes [B, N, T]                                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### C++ — `acceleration/kernel_config.h`

```cpp
struct KernelConfig {
    bool accelerate;           // master switch
    bool optimize_throughput;  // auto-tune block/grid
    bool optimize_memory;      // audit GPU memory headroom
    bool profile_energy;       // NVML power + timing
};

constexpr KernelConfig KERNEL_CONFIG_DEFAULT     = {false, false, false, false};
constexpr KernelConfig KERNEL_CONFIG_ACCELERATED = {true,  true,  true,  true };
```

Pass `KERNEL_CONFIG_DEFAULT` or `KERNEL_CONFIG_ACCELERATED` directly to
`snn_engine_dispatch()`, or build a custom `KernelConfig` from the YAML below.

### Python — `acceleration/accel_config.yaml`

```yaml
accelerate: false          # ← set true to enable full acceleration stack

optimize_throughput: true
optimize_memory: true
profile_energy: false      # requires NVML; disabled by default

memory_warn_threshold_mb: 512
```

Typical loader pattern:

```python
import yaml
from pathlib import Path

cfg_raw = yaml.safe_load(
    (Path(__file__).parent / "acceleration/accel_config.yaml").read_text()
)
# pass cfg_raw["accelerate"] etc. to the bound C++ extension
```

---

## GPU Attribute Modules

| Module | Header | What it measures / controls |
|--------|--------|-----------------------------|
| `energy_management` | `EnergyProfiler` | Kernel wall time (CUDA events) + GPU power draw (NVML mW) + energy (mJ) |
| `memory_management` | `snn_query_memory` | Free / total VRAM via `cudaMemGetInfo`; pinned host allocation helpers |
| `throughput_optimization` | `compute_1d_launch` | Optimal 1-D block size via occupancy API; theoretical occupancy fraction |

`profile_energy: false` compiles and runs without NVML — timing is always
available; power readings fall back to 0.0 mW automatically.

---

## LIF Kernel — Mathematics

Both paths execute the same Leaky Integrate-and-Fire update per work item `(b, n, t)`:

```
v_new = v_prev × (1 − τ⁻¹) + input[b, n, t]
spike  = 1  if  v_new ≥ v_th  else  0
v_new  = 0  if  spike          (hard reset)
```

| Symbol | Argument | Default |
|--------|----------|---------|
| `v_th` | firing threshold | 1.0 |
| `τ⁻¹` | `tau_inv` — inverse membrane time constant | 0.1 |

---

## Key Invariants

| Rule | Where Enforced |
|------|----------------|
| All tensors must be on CUDA before dispatch | `TORCH_CHECK(input.is_cuda() && voltage.is_cuda())` in `engine.cu` |
| Input must be 3-D `[B, N, T]`; voltage must be 2-D `[B, N]` | `TORCH_CHECK(input.dim() == 3)` in `engine.cu` |
| `profile_energy: false` is safe on hardware without NVML | Stubs in `energy_management.cu` return 0 mW; `nvml_available = false` |
| Memory audit uses 2× headroom: output tensor + internal allocations | `dispatch_accelerated()` in `engine.cu` |
| `accel_config.yaml` and `kernel_config.h` must stay in sync | Shared field names; no auto-generation — update both when adding a flag |
