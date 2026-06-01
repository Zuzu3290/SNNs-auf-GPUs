# SNNs-auf-GPUs

A research platform for running Spiking Neural Networks on GPU hardware.
The system moves from Python-based SNN frameworks toward a GPU-native runtime
where a custom CUDA kernel owns the full execution path — event ingestion,
neuron dynamics, weight updates, and memory management.

---

## What This Is

Neuromorphic computing on commodity GPUs. The project bridges event-based
sensor data (DVS / DAVIS cameras) with SNN training frameworks, while
progressively replacing Python operations with compiled CUDA kernels.

The codebase is structured in two planes that will converge over time:

| Plane | Location | Role |
|-------|----------|------|
| Python | `src/learning/`, `event_data_workflow/`, `skeleton/` | Configuration, framework wrappers, fallback path |
| CUDA | `src/crsc/`, `acceleration/` | Kernel execution — LIF dynamics, spike ops, memory |

The compiler layer (`src/compiler/`) bridges them, dispatching operations to
the kernel when available and falling back to Python otherwise.

---

## Project Layout

```
src/
  learning/         SNN framework wrappers — SNNTorch, Norse, SpikingJelly
  compiler/         JIT compiler, kernel loader, dispatch bridge
  crsc/             CUDA kernels — membrane, spike, threshold, reset, decode
skeleton/           Configuration and settings (SNN_module.yaml)
event_data_workflow/ Neuromorphic data pipeline — caching, slicing, DataLoader
acceleration/       GPU hardware attributes, PTX loader, SNN hardware mapping
docs/               Architecture references and hardware notes
```

---

## Entry Point

```
python src/learning/main.py
```

Reads `SNN_module.yaml`, loads the neuromorphic dataset via
`event_data_workflow`, builds the SNN model, runs training, then evaluation.

---

## Configuration

All runtime parameters live in `SNN_module.yaml` at the project root:
architecture, training schedule, dataset path, device, compiler flags, and
data pipeline settings. No hardcoded values in source files.

---

## Framework Backends

The trainer and inference pipeline are framework-agnostic at the model boundary.
Any model that satisfies the `ModelInterface` contract can be plugged in — the
pipeline does not care what runs inside.

| Backend | Status | Notes |
|---------|--------|-------|
| SNNTorch | Working | Default |
| Norse | Working | Current default in `main.py` |
| SpikingJelly | Working | |
| JAX + Flax/Haiku | Extension point | Trains via XLA; DLPack bridge to PyTorch at boundary |
| TensorFlow | Extension point | DLPack bridge at boundary |
| Custom / from scratch | Extension point | Return a PyTorch tensor — everything else is your choice |

For details on how each backend cooperates with the training loop, backward pass,
and adversarial evaluation, see [`docs/frameworks/`](docs/frameworks/).

---

## Current Capabilities

- Three SNN backends: SNNTorch, Norse, SpikingJelly — switchable via config
- Adaptive data pipeline: selects memory, disk, hybrid, or GPU-VRAM cache
  strategy automatically based on available system resources
- Activity regularization and STDP as differentiable loss terms alongside BPTT
- Fused LIF CUDA kernel with surrogate gradient for BPTT
- JIT compiler pipeline that lowers SNN models to an IR and schedules
  device-aware execution
- Adversarial robustness evaluation via TRADES

---

## Roadmap

The kernel dispatch layer (`src/compiler/runtime.py`) is the next build target.
When complete, GPU detection at startup routes all operations — event decoding,
tensor caching, neuron dynamics, weight updates — through `src/crsc/` kernels.
Python implementations remain as the CPU fallback and correctness reference.

See `docs/kernel_dispatch_architecture.md` for the full plan.
