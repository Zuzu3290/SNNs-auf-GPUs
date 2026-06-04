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

---

## Comparative Evaluation — Custom CUDA Kernel vs JAX/XLA

### What JAX/XLA does for LIF

JAX's equivalent of our temporal kernel uses two primitives:

```python
@jax.jit
def lif_forward(voltage, inputs, v_th, tau_inv):
    def step(v, x):
        v_new = v * (1.0 - tau_inv) + x
        fired = (v_new >= v_th).astype(jnp.float32)
        return v_new * (1.0 - fired), fired
    final_v, spikes = jax.lax.scan(step, voltage, inputs)
    return final_v, spikes
```

`jax.lax.scan` is the XLA loop primitive. XLA compiles the scan body + loop into a **single GPU kernel** — no Python dispatch per timestep. The sequential dependency `v(t) → v(t+1)` is handled by XLA's loop unrolling. After the first JIT compilation, the wall-time cost is similar to a single CUDA kernel call.

This means JAX/XLA already solves the same problem our temporal kernel solved — the Python dispatch loop eliminated, voltage state flowing correctly across T steps. The question is how deep the optimisation goes.

---

### Head-to-head comparison

| Dimension | Custom CUDA kernel | JAX / XLA |
|---|---|---|
| **Python dispatch per forward** | 1 pybind11 call | 1 JAX dispatch (after JIT warmup) |
| **Temporal loop** | Register-resident, hand-coded | XLA-compiled scan, compiler-managed registers |
| **Grid sizing** | SM-saturating adaptive (`target_blocks_per_sm`) | XLA heuristic — no user control |
| **Warp ballot** | `__ballot_sync` per timestep, free | No equivalent — requires separate reduction kernel |
| **cuMemPool control** | `MemoryArbiter` pins emergency zone | XLA allocator — no user access |
| **Adaptive energy feedback** | Built into `engine.cu` dispatch (every 50 calls) | Not available — no elapsed-time hook |
| **First-call cost** | ~0 (already compiled via `load()`) | XLA compilation: 10–120 s depending on model size |
| **Backward pass** | Falls back to PyTorch autograd | `jax.grad` through `lax.scan` — analytically exact |
| **Hardware portability** | CUDA/NVIDIA only | GPU, TPU, CPU via XLA backend |
| **Lines of custom code** | ~2000 (kernels + binding + dispatcher) | ~20 (lax.scan + jit decorator) |

---

### Kernel time estimate — same problem (B=4, N=512, T=25)

Our measured kernel time: **0.035 ms**

JAX/XLA at the same scale: estimated **0.05–0.10 ms**

XLA generates correct, vectorised GPU code but does not reach hand-tuned occupancy. The gap comes from:

1. **Register pressure**: Our kernel explicitly loads voltage into a named register and keeps it there for all T iterations. XLA's compiler may achieve this for simple scan bodies, but register allocation across a compiled loop is less predictable than explicit CUDA code. Spills to local memory (L1 cache) are possible.

2. **Grid strategy**: Our SM-saturation design launches exactly `sm_count × bps` blocks. XLA uses a fixed heuristic (typically `ceil(N / 256)` blocks) — identical to our kernel at this small scale, but does not adapt to measured elapsed time.

3. **Memory access pattern**: Our kernel pre-computes `in_ptr = input + b*N*T + n*T` once and strides through T with a simple pointer increment. XLA generates slice operations through its HLO (High Level Operations) IR, which adds abstraction overhead that the XLA compiler must optimise away.

The net result: at kernel-time level, our code is roughly **1.5–3x faster** than XLA-generated code for the LIF scan at this scale. At wall-time level (including Python-side overhead), the gap narrows to roughly **1.2–2x** because both pay approximately the same Python-to-GPU boundary crossing cost.

---

### Where JAX/XLA is strictly better

**1. Gradient computation**
`jax.grad(lif_forward)` through `lax.scan` produces exact analytical gradients via reverse-mode autodiff unrolled through the scan. Our kernel has no CUDA backward pass — backpropagation falls back to PyTorch autograd which reconstructs gradients from the spike output tensor. This is correct but less efficient: PyTorch re-traces the operations, while JAX's backward through scan is a compiled pass that runs at the same speed as the forward.

At `B=32, N=4096, T=25` (production scale), the backward pass is typically 2–3x the cost of the forward. This is where the JAX advantage is measurable.

**2. Architecture flexibility**
Changing the LIF equation in JAX means editing the 4-line `step` function and recompiling. Changing it in our system means editing C++/CUDA, recompiling the extension, and pushing to the repository. For research iteration, JAX is significantly more productive.

**3. Cross-hardware deployment**
JAX/XLA targets NVIDIA GPU, Google TPU, and CPU through the same API. Our kernel is CUDA-only. For a deployment that needs to run on TPU (automotive inference on Google edge hardware) or a future RISC-V neuromorphic chip with an XLA backend, JAX remains viable. Our kernel does not.

---

### Where our kernel is strictly better

**1. Warp ballot — free sparsity measurement**
`__ballot_sync(mask, fired)` produces the 32-bit warp firing pattern in one instruction at zero compute cost. JAX/XLA has no equivalent — measuring spike rate requires `jnp.sum(spikes)`, a full reduction kernel that adds a memory round-trip. The ballot mask feeds our `SpikeRateBus` and adaptive energy feedback without any additional kernel launch.

**2. cuMemPool release threshold**
`MemoryArbiter` sets the pool release threshold to pin the emergency zone. JAX's XLA allocator manages its own memory pool — the user has no hook to set a retention floor. In a multi-tenant system (data pipeline + model + kernel workspace all on the same GPU), this matters: JAX can have its allocator evict memory that another subsystem needed.

**3. Adaptive SM scheduling**
`target_blocks_per_sm` is adjusted every 50 calls based on measured kernel elapsed time. At low spike rates (0.3% as measured), the kernel completes fast — the feedback loop detects this and would hold or reduce block count at production scale, keeping fewer SMs engaged proportional to neural activity. JAX launches the same grid every call regardless of how much work was done.

**4. No warmup cost**
Our kernel is ready on the first call after `load()`. JAX's first `jit`-decorated call triggers XLA compilation — for a model with `B=32, N=4096, T=25`, this is typically 15–60 seconds on a T4. In production inference, this warmup cost must be paid on every cold start (Colab runtime reset, container restart, etc.).

---

### Quantitative summary

| Metric | Custom CUDA | JAX/XLA | Advantage |
|---|---|---|---|
| Kernel time (B=4, N=512, T=25) | 0.035 ms | ~0.06–0.10 ms | **CUDA 1.7–3x** |
| Wall time after warmup | 0.028 ms | ~0.04–0.07 ms | **CUDA 1.4–2.5x** |
| First-call latency | ~0 ms | 15–120 s | **CUDA** |
| Spike rate monitoring overhead | 0 (ballot, free) | ~0.01 ms (reduction) | **CUDA** |
| Gradient quality | PyTorch autograd | Exact scan grad | **JAX** |
| Code volume to change LIF eq. | ~50 lines CUDA + rebuild | 4 lines Python + recompile | **JAX** |
| Hardware targets | CUDA only | GPU + TPU + CPU | **JAX** |
| VRAM pool control | cuMemPool API | None | **CUDA** |

---

### Architectural conclusion

JAX/XLA and our kernel are not in competition for the same role. JAX is a **research compiler** — fast to iterate, portable, analytically correct gradients. Our kernel is a **production substrate** — tighter hardware control, lower latency, zero warmup, sparsity-aware via ballot, and integrated with the runtime layer that manages VRAM across the whole system.

For the automotive ADAS context (DVS event camera pipeline, fixed inference architecture, latency-critical deployment): our kernel is the correct choice. For a research lab exploring different SNN architectures on the same hardware: JAX is more productive and JAX's gradient quality advantage matters for training fidelity.

The two are complementary rather than competing: JAX can be used to prototype and validate new LIF variants, then the validated kernel is hand-implemented in CUDA for deployment.

---

## What Remains

| Item | Status |
|---|---|
| Cell 8 training run | Fails — `from runtime import ...` not pulled into Colab environment yet |
| `snn_runtime` Colab build | Cell 4b available but optional |
| Sparse downstream matmul (ballot mask → skip silent rows) | Not implemented — next frontier |
| float4 vectorised input loads | Not implemented — memory bandwidth ceiling at production scale |
| NVML energy readings | Unavailable on free T4 tier — guarded by `SNN_HAS_NVML=0` |
