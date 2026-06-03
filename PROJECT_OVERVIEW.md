# Project Overview — SNNs auf GPUs

## Abstract

This project builds a full-stack GPU-accelerated Spiking Neural Network (SNN)
pipeline targeting real-time spatial perception for automotive ADAS systems.
The input is a DVS (Dynamic Vision Sensor) event camera stream — a
neuromorphic sensor that produces asynchronous per-pixel brightness-change
events rather than frames. Events are converted to spike tensors on the GPU
and fed into a recurrent LIF (Leaky Integrate-and-Fire) SNN that classifies
or detects objects in the scene. The entire chain, from raw events to
inference output, is designed to run on a single GPU with energy,
memory, and throughput all explicitly managed and profiled.

The **Operators** branch (current) is focused on kernel scalability: replacing
PyTorch's generic operators with a custom CUDA kernel stack that exposes
per-launch energy, memory headroom, and SM occupancy as first-class concerns.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  DVS Event Camera  (x, y, t, polarity stream)                              │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  event_data_workflow/  — Neuromorphic Data Pipeline                        │
│                                                                            │
│  system_monitor      → probe RAM / VRAM / disk at startup                 │
│  pipeline_coordinator → split budget (40% cache / 30% workers / 30% prefetch) │
│  cache_engine        → select strategy: MEMORY / HYBRID / DISK / GPU      │
│  data_pipeline       → NeuromorphicEncoder: raw → cache → slice → loader  │
│  temporal_slicer     → break recordings into fixed T-ms windows           │
│                                                                            │
│  Output: pinned CPU tensors [T, B, C, H, W]  →  async H2D transfer        │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  acceleration/spike_kernel.py  — Event → Spike Conversion (GPU)            │
│                                                                            │
│  SpikeKernel (CuPy or Torch backend):                                      │
│    event_to_spikes  → hard threshold: spike if events ≥ θ                 │
│    lif_spikes       → per-pixel LIF membrane integration                  │
│    merge_polarities → ON spike +1 / OFF spike −1 signed map               │
│                                                                            │
│  Output: spike tensor [2, T, H, W] on GPU                                 │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  src/compiler/  — SNN Compiler & IR                                        │
│                                                                            │
│  compiler.py     → entry point; wraps model with torch.compile if enabled  │
│  src/ir.py       → intermediate representation for SNN ops                 │
│  src/lowering.py → lowers IR to CUDA-executable ops                       │
│  src/planner.py  → memory and execution planner                            │
│  src/scheduler.py → timestep scheduling across layers                     │
│  passes/         → device_annotation, op_rewrite, fusion                  │
│  kernels/lif_kernel.h → fused LIF forward + backward declarations         │
│                                                                            │
│  Controlled by SNN_module.yaml: compiler.torch_compile flag               │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  src/crsc/  — Custom CUDA Kernel Stack (CRSC)                              │
│                                                                            │
│  binding.cpp          → pybind11: exposes snn_forward to Python            │
│                          SNNCompiler maps HardwareConfig → KernelConfig    │
│  engine.cu            → dispatcher: KernelConfig → basic or accelerated   │
│  kernels/snn_forward.cu → lif_kernel + snn_forward_cuda + snn_forward_profiled │
│  compiler_bridge.cpp  → reserved bridge to compiler IR (placeholder)      │
│                                                                            │
│  Two execution paths selected by KernelConfig:                             │
│    basic:       fixed 256 threads, no overhead                            │
│    accelerated: occupancy-tuned + memory audit + NVML energy profiling    │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  acceleration/  — GPU Attribute & Execution Layer                          │
│                                                                            │
│  kernel_config.h      → KernelConfig struct; DEFAULT and ACCELERATED presets │
│  accel_config.yaml    → Python-readable mirror; flip accelerate: true     │
│  Kernel.cu            → standalone PyTorch-callable LIF kernel             │
│  SNN_mapping.cpp      → SNN↔neuromorphic mapping documentation + binding  │
│  spike_kernel.py      → CuPy / Torch event-to-spike backends              │
│                                                                            │
│  GPU_attributes/                                                           │
│    energy_management   → EnergyProfiler (CUDA events + NVML power)        │
│    memory_management   → snn_malloc_device / pinned / snn_query_memory    │
│    throughput_optimization → compute_1d_launch (occupancy API)            │
│                                                                            │
│  backend/                                                                  │
│    launch_wrappers     → LaunchSession (begin/end brackets any kernel)    │
│    ptx_loader          → CUDA Driver API runtime PTX load + hot-swap      │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  src/learning/  — Training Framework                                       │
│                                                                            │
│  main.py         → training entry point; reads SNN_module.yaml            │
│  training.py     → epoch loop, BPTT, loss assembly                        │
│  inference.py    → eval loop                                               │
│  frameworks/     → snn_norse, snn_spikingjelly, snn_torch adapters        │
│  adversarial_robustness.py → TRADES: clean loss + KL(clean ∥ adversarial) │
│                                                                            │
│  Loss components (all configurable in SNN_module.yaml):                   │
│    CrossEntropy task loss                                                  │
│    TRADES robustness penalty (PGD-based adversarial perturbation)         │
│    Activity regularisation (dead / saturated neuron prevention)           │
│    STDP causal correlation loss                                            │
└──────────────────────────────┬─────────────────────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  viewer/  — Visualisation Frontend (TypeScript / React)                    │
│  skeleton/ — Project-wide config and structured logging utilities          │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Subsystem Summary

| Subsystem | Folder | Role |
|-----------|--------|------|
| Data pipeline | `event_data_workflow/` | DVS event ingestion, adaptive caching, temporal slicing |
| Spike conversion | `acceleration/spike_kernel.py` | Event voxel → spike tensor on GPU |
| Compiler / IR | `src/compiler/` | SNN op representation, lowering, fusion, torch.compile bridge |
| CRSC kernel | `src/crsc/` | Custom LIF CUDA kernel, pybind11 bindings, KernelConfig dispatcher |
| Acceleration layer | `acceleration/` | Energy / memory / throughput GPU attributes; LaunchSession; PTX loader |
| Learning | `src/learning/` | Multi-framework SNN training, TRADES robustness, STDP, activity reg |
| Config | `SNN_module.yaml` | Single top-level config: architecture, training, compiler, dataset |
| Viewer | `viewer/` | Visualisation frontend |

---

## Configuration — SNN_module.yaml

The entire runtime is controlled from one file at the project root.

| Section | Key setting | Effect |
|---------|-------------|--------|
| `architecture` | `device: cuda` | All tensor ops on GPU |
| `training` | `kernel: OFF / ON` | OFF = PyTorch ops; ON = compiled CRSC CUDA kernel |
| `training` | `trades_enabled` | Adds PGD adversarial robustness penalty to loss |
| `training` | `activity_reg_enabled` | Penalises dead (< 1%) and saturated (> 50%) neurons |
| `training` | `stdp_enabled` | Adds STDP causal spike-order correlation loss |
| `compiler` | `torch_compile: false / true` | Wraps model with torch.compile / TorchInductor |

The acceleration path is separately controlled by `acceleration/accel_config.yaml`.

---

## Kernel Execution Paths

```
SNN_module.yaml: kernel: ON
        │
        ▼
src/crsc/binding.cpp  (pybind11)
        │
        ▼
src/crsc/engine.cu — snn_engine_dispatch(KernelConfig, input, voltage)
        │
        ├── accelerate: false ──► dispatch_basic
        │                         lif_basic<<<grid=ceil(B×N×T/256), 256>>>
        │                         fixed block, no overhead
        │
        └── accelerate: true  ──► dispatch_accelerated
                                  1. snn_query_memory  → VRAM headroom check
                                  2. compute_1d_launch → optimal block/grid
                                  3. EnergyProfiler.start
                                  4. lif_basic<<<grid, block>>>
                                  5. EnergyProfiler.stop → KernelEnergyResult
```

The `accelerate` flag is toggled in `acceleration/accel_config.yaml` and maps
to `KernelConfig` in `acceleration/kernel_config.h`.

---

## LIF Neuron Model

All kernel paths implement the same discrete-time Leaky Integrate-and-Fire update:

```
v(t) = v(t−1) × (1 − τ⁻¹) + I(t)       leaky integration
s(t) = 1  if  v(t) ≥ v_th  else  0       threshold comparison
v(t) = 0  if  s(t) = 1                   hard reset on spike
```

| Parameter | Symbol | Default | Config key |
|-----------|--------|---------|------------|
| Firing threshold | `v_th` | 1.0 | `architecture.threshold` |
| Inverse time constant | `τ⁻¹` (`tau_inv`) | 0.1 | — |
| Leak factor | `β` | 0.95 | `training.beta` |

---

## Branch Status — Operators

The Operators branch objective is kernel scalability. What has been completed:

| Item | Status |
|------|--------|
| Basic LIF kernel in `src/crsc/kernels/snn_forward.cu` | Done |
| `KernelConfig` + `accel_config.yaml` config bridge | Done |
| `engine.cu` dispatcher (basic / accelerated paths) | Done |
| `EnergyProfiler` (CUDA events + optional NVML) | Done |
| `memory_management` (device / pinned / query) | Done |
| `throughput_optimization` (occupancy auto-tune) | Done |
| `LaunchSession` backend (begin / end wrapper) | Done |
| PTX runtime loader (Driver API, hot-swap capable) | Done |
| 3-line file headers across `acceleration/` and `src/crsc/` | Done |
| Hardware docs (`kernel_acceleration.md`, `backend.md`) | Done |

Next: wire `accel_config.yaml` into the Python training loop so `kernel: ON`
also reads the YAML and passes the correct `KernelConfig` preset to
`snn_engine_dispatch`.
