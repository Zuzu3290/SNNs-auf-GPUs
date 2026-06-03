# Acceleration Backend — Architecture Reference

Documents the execution infrastructure layer in `acceleration/backend/`.
The backend sits between the high-level dispatcher (`src/crsc/engine.cu`) and
the raw CUDA runtime, providing two services: **profiled launch sessions** and
**runtime PTX loading**.

---

## File Map

| File | Responsibility |
|------|----------------|
| `acceleration/backend/launch_wrappers.h` | `ProfiledLaunchResult` struct; `LaunchSession` RAII interface |
| `acceleration/backend/launch_wrappers.cu` | `LaunchSession::begin/end`; `print_profiled_result` |
| `acceleration/backend/ptx_loader.h` | Driver API interface — load, get kernel, launch, unload |
| `acceleration/backend/ptx_loader.cu` | Full CUDA Driver API implementation (`cuModuleLoadData`, `cuLaunchKernel`) |

---

## Service 1 — Profiled Launch Sessions (`launch_wrappers`)

### Purpose

`engine.cu` needs to sequence three operations around every kernel launch when
in accelerated mode: compute the optimal block/grid, arm the energy profiler,
and collect the result afterward. `LaunchSession` wraps this into a clean
begin/end pair so the dispatcher does not repeat the sequencing logic.

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  Caller (engine.cu — dispatch_accelerated)                           │
│                                                                      │
│  LaunchSession sess;                                                 │
│  sess.begin(kernel_fn, n_elements, stream)                           │
│      │                                                               │
│      ├─► compute_1d_launch(kernel_fn, n_elements)                    │
│      │     cudaOccupancyMaxPotentialBlockSize                        │
│      │     → sess.config.block_size, grid_size, occupancy           │
│      │                                                               │
│      └─► EnergyProfiler.start(stream)                               │
│            cudaEventRecord(ev_start)                                 │
│            nvmlDeviceGetPowerUsage → power_before_mw                │
│                                                                      │
│  kernel<<<sess.config.grid_size, sess.config.block_size>>>( ... )   │
│                                                                      │
│  ProfiledLaunchResult r = sess.end(stream)                           │
│      │                                                               │
│      └─► EnergyProfiler.stop(stream)                                │
│            cudaEventRecord(ev_stop)                                  │
│            cudaEventSynchronize(ev_stop)                             │
│            cudaEventElapsedTime → elapsed_ms                         │
│            nvmlDeviceGetPowerUsage → power_after_mw                 │
│            energy_mj = avg_power_W × elapsed_s × 1000              │
│                                                                      │
│  r = ProfiledLaunchResult { KernelEnergyResult, LaunchConfig }       │
└──────────────────────────────────────────────────────────────────────┘
```

### ProfiledLaunchResult

| Field | Type | Content |
|-------|------|---------|
| `energy.elapsed_ms` | `float` | Kernel wall time from CUDA events |
| `energy.power_before_mw` | `float` | GPU power sampled before launch (NVML) |
| `energy.power_after_mw` | `float` | GPU power sampled after sync (NVML) |
| `energy.energy_mj` | `float` | Estimated energy: avg power × time |
| `energy.nvml_available` | `bool` | False → timing-only; power fields are 0 |
| `config.block_size` | `int` | Threads per block chosen by occupancy API |
| `config.grid_size` | `int` | Blocks launched |
| `config.theoretical_occupancy` | `float` | Fraction of max SM warp slots in use |

### Usage Pattern

```cpp
#include "acceleration/backend/launch_wrappers.h"

LaunchSession sess;
sess.begin((const void*)lif_basic, (int)total, stream);

lif_basic<<<sess.config.grid_size, sess.config.block_size, 0, stream>>>(
    input, voltage, spikes, v_th, tau_inv, B, N, T
);

ProfiledLaunchResult r = sess.end(stream);
print_profiled_result(r, "lif_basic");
```

---

## Service 2 — Runtime PTX Loading (`ptx_loader`)

### Purpose

The Runtime API (used everywhere else) bundles CUDA C kernels into the host
binary at compile time. The Driver API alternative — loading PTX at runtime —
enables two things this project needs:

1. **Per-arch kernel selection**: ship separate PTX files compiled for SM 7.5
   (Turing), SM 8.6 (Ampere), SM 8.9 (Ada) and select the right one at
   startup based on `cudaGetDeviceProperties`.
2. **Kernel hot-swapping**: replace the PTX file on disk and reload without
   recompiling or relinking the Python extension.

### CUDA Driver API vs Runtime API

| | Runtime API (`cuda_runtime.h`) | Driver API (`cuda.h`) |
|---|---|---|
| Kernel launch | `kernel<<<grid, block>>>()` | `cuLaunchKernel(fn, ...)` |
| Module load | Bundled at compile time | `cuModuleLoadData(ptx_src)` |
| Flexibility | Fixed at build | Hot-swap at runtime |
| Init required | Automatic | `cuInit(0)` once per process |

### Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  ptx_load_from_file("kernels/lif_sm89.ptx")                          │
│      │                                                               │
│      ├─► ensure_driver_init()   cuInit(0) — once per process        │
│      ├─► read_file()            fopen / fread PTX source into RAM   │
│      └─► cuModuleLoadData()     JIT-compile PTX → cubin on device   │
│                                 returns CUmodule                     │
│                                                                      │
│  ptx_get_kernel(mod, "lif_basic")                                    │
│      └─► cuModuleGetFunction()  returns CUfunction handle           │
│                                                                      │
│  ptx_launch_1d(fn, n_elements, block_size, 0, stream, args)          │
│      │                                                               │
│      ├─► grid = ceil(n_elements / block_size)                        │
│      └─► cuLaunchKernel(fn, grid,1,1, block,1,1, 0, stream, args)   │
│                                                                      │
│  ptx_unload(mod)                                                     │
│      └─► cuModuleUnload()       frees device-side cubin             │
└──────────────────────────────────────────────────────────────────────┘
```

### API Reference

```cpp
// Load from disk (reads file, calls cuModuleLoadData internally)
CUmodule ptx_load_from_file(const char* ptx_path);

// Load from an in-memory PTX string (e.g. embedded via raw string literal)
CUmodule ptx_load_from_string(const char* ptx_source);

// Get a kernel handle by symbol name
CUfunction ptx_get_kernel(CUmodule module, const char* kernel_name);

// Launch a 1-D grid; kernel_args = array of void* to each argument
void ptx_launch_1d(CUfunction fn,
                   int n_elements, int block_size,
                   size_t shared_mem_bytes,
                   CUstream stream,
                   void** kernel_args);

// Release the module and its device memory
void ptx_unload(CUmodule module);
```

### Usage Pattern

```cpp
#include "acceleration/backend/ptx_loader.h"

// At startup — pick arch from device properties
CUmodule   mod = ptx_load_from_file("acceleration/ptx/lif_sm89.ptx");
CUfunction fn  = ptx_get_kernel(mod, "lif_basic");

// Per-inference
void* args[] = { &d_input, &d_voltage, &d_spikes,
                 &v_th, &tau_inv, &B, &N, &T };
ptx_launch_1d(fn, (int)total, 256, 0, stream, args);

// Shutdown
ptx_unload(mod);
```

### Generating PTX from a CUDA kernel

```bash
# Compile lif_basic kernel to PTX for SM 8.9 (RTX 40xx / Ada Lovelace)
nvcc -ptx -arch=sm_89 src/crsc/kernels/snn_forward.cu -o acceleration/ptx/lif_sm89.ptx

# SM 8.6 (RTX 30xx / Ampere)
nvcc -ptx -arch=sm_86 src/crsc/kernels/snn_forward.cu -o acceleration/ptx/lif_sm86.ptx
```

---

## How Backend Connects to the Rest of Acceleration

```
accel_config.yaml
      │  accelerate: true
      ▼
src/crsc/engine.cu — snn_engine_dispatch()
      │
      │  accelerated path
      ▼
┌─────────────────────────────────────┐
│  acceleration/backend/              │
│                                     │
│  Option A — Runtime API path        │
│  ┌─────────────────────────────┐   │
│  │  LaunchSession              │   │
│  │  .begin(lif_basic, total)   │   │
│  │    → compute_1d_launch      │   │
│  │    → EnergyProfiler.start   │   │
│  │  lif_basic<<<...>>>()       │   │
│  │  .end()  → ProfiledResult   │   │
│  └─────────────────────────────┘   │
│                                     │
│  Option B — Driver API (PTX) path   │
│  ┌─────────────────────────────┐   │
│  │  ptx_load_from_file(...)    │   │
│  │  ptx_get_kernel("lif_basic")│   │
│  │  ptx_launch_1d(fn, ...)     │   │
│  │  ptx_unload(mod)            │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
      │
      ▼
GPU — lif_basic kernel executes
      │
      ▼
spikes [B, N, T] returned to Python
```

---

## Key Invariants

| Rule | Where Enforced |
|------|----------------|
| `cuInit(0)` called once before any Driver API use | `ensure_driver_init()` static guard in `ptx_loader.cu` |
| PTX file read fully into RAM before `cuModuleLoadData` | `read_file()` in `ptx_loader.cu`; buffer freed after load |
| `LaunchSession::end()` must be called after every `begin()` | `EnergyProfiler` destructor releases CUDA events on scope exit |
| `ptx_unload()` must be called when module is no longer needed | `cuModuleUnload` releases device-side cubin |
| IntelliSense path warnings (code 1696) on backend `.cu` files are IDE-only | nvcc resolves relative `../GPU_attributes/` paths correctly at build time |
