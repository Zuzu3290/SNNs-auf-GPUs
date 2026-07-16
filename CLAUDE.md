# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run training
```bash
# Preferred — sets PYTHONPATH correctly
./launch.sh

# Direct
export PYTHONPATH="$PWD:$PWD/src"
python src/learning/main.py
```

### Install dependencies
```bash
pip install -r requirements.txt
# Pick one based on your CUDA version:
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x
# Install PyTorch separately via https://pytorch.org/get-started/locally/
```

### Build the optional CRSC CUDA extension
```bash
# Only needed when training.kernel: ON in SNN_module.yaml
export PYTHONPATH="$PWD:$PWD/src"
python src/learning/setup.py build_ext --inplace
```
This compiles `snn_cuda.snn_forward` from `src/crsc/kernels/snn_forward.cu` plus the `acceleration/GPU_attributes/` kernels.

### Run tests
```bash
# Individual compiler test files (all are runnable directly):
export PYTHONPATH="$PWD:$PWD/src"
python src/compiler/tests/test_ir.py
python src/compiler/tests/test_runtime.py
python src/compiler/tests/test_scheduler.py
python src/compiler/tests/test_cuda_execution.py
```

### Inspect configuration
```bash
python skeleton/snn_config.py   # prints the full Settings display table
```

### Run framework comparison experiments
```bash
export PYTHONPATH="$PWD:$PWD/src"

# Run one phase at a time (recommended on free Colab — one session per phase)
python src/experiments/run_experiments.py --phase 1   # threshold sweep
python src/experiments/run_experiments.py --phase 2   # decay sweep (needs phase 1 done)
python src/experiments/run_experiments.py --phase 3   # head-to-head (needs 1+2 done)

# Or run all phases in one go (long)
python src/experiments/run_experiments.py

# After phases complete:
python src/experiments/aggregate.py    # → outputs/experiments/summary.csv + stats.json
python src/experiments/visualize.py   # → outputs/experiments/plots/*.png
```

Re-running any command is safe — completed runs are skipped automatically.
Change `meta.output_root` in `configuration/experiments.yaml` to a Google Drive path to persist results across Colab sessions.

### Web viewer (optional)
```bash
cd viewer && npm run dev
```

---

## Architecture

The project operates on **two converging planes**:

| Plane | Location | Purpose |
|-------|----------|---------|
| Python | `src/learning/`, `event_data_workflow/`, `skeleton/` | Training loop, framework wrappers, data pipeline, config |
| CUDA | `src/crsc/`, `acceleration/` | Kernel execution — LIF dynamics, spike ops, GPU memory |

The **compiler layer** (`src/compiler/`) bridges them: it lowers the model to an IR, schedules device-aware execution, and dispatches to `src/crsc/` kernels when available, otherwise falling back to the Python path.

### Configuration flow

All runtime values live in YAML — **no hardcoded parameters in source**.

```
configuration/SNN_module.yaml         ← framework, training, TRADES/STDP, dataset
configuration/network_architecture.yaml ← conv layer sizes, neuron types per framework
configuration/data_workflow.yaml      ← cache strategy, temporal slicing
        ↓
skeleton/snn_config.py (Settings)     ← parses both YAMLs, exposes as cfg.ATTR
        ↓
src/learning/main.py                  ← reads cfg, wires everything together
```

`Settings` is the single source of truth passed to every component (`SNNTrainer`, `SNNTester`, `NeuromorphicEncoder`, etc.).

### Model abstraction

Every SNN backend implements `ModelInterface` (`src/learning/frameworks/model_interface.py`). The training and evaluation pipeline never calls framework-specific code directly — only the interface methods.

Key contract points:
- `forward(data)` always receives and returns a **PyTorch tensor**, regardless of what runs inside
- `backward_pass()` is a no-op for JAX/TF backends (gradients computed inside `forward()`)
- `tensor_format()` returns `"TB"` (time-first `[T, B, C, H, W]`) by default; the trainer transposes if needed
- `is_differentiable()` returns `False` for non-PyTorch backends, skipping adversarial attack generation

The three current backends (`snn_norse.py`, `snn_torch.py`, `snn_spikingjelly.py`) all share the same Conv-SNN architecture: `Conv→LIF→MaxPool→Conv→LIF→MaxPool→FC→LIF`. The neuron type per layer is controlled by `network_architecture.yaml`.

### Training loop (`src/learning/training.py`)

`SNNTrainer.train()` composes multiple loss terms in a single backward pass:
1. **Primary loss** — cross-entropy or MSE over spike count (framework-dependent)
2. **TRADES** — KL divergence between clean and PGD-perturbed outputs (when `trades_enabled: true`)
3. **Activity regularization** — two-sided penalty for dead (<1% firing) and saturated (>50% firing) neurons
4. **STDP** — causal correlation loss alongside BPTT

All three optional terms are gated by their `_enabled` flag in `SNN_module.yaml`.

### Data pipeline (`event_data_workflow/`)

`NeuromorphicEncoder` chains three layers:
1. **`AdaptiveCacheController`** — selects RAM / disk / GPU-VRAM / hybrid cache based on live system resources
2. **`TemporalSlicedDataset`** — slices recordings into fixed-duration windows (applied *after* caching so one recording produces N cache hits)
3. **`PipelineMemoryCoordinator`** — monitors VRAM/RAM pressure during training and adjusts worker count/batch size dynamically (single-GPU only)

### Compiler pipeline (`src/compiler/`)

```
model → ir.py (ComputeGraph/IRNode) → passes/ (fusion, device annotation, op rewrite)
     → planner.py (FusedStep plan) → runtime.py (execute, lif_step)
```

`torch.compile()` is an optional additional wrapper (set `compiler.torch_compile: true`). The CRSC kernel path (`src/crsc/`) requires building the C++ extension first.

### Switching frameworks

Change `training.framework` in `configuration/SNN_module.yaml` to `norse`, `torch`, or `sj`. No code changes needed. Per-framework neuron params (threshold, beta, tau) live in the `frameworks:` block of the same file.

**Important threshold note**: `threshold: 1.0` kills Norse (dead neurons, confirmed EXP-004). `threshold: 0.5` is the safe neutral starting point for all three backends.

### Framework comparison experiments (`src/experiments/`)

Three-phase study comparing Norse, SNNTorch, and SpikingJelly on N-MNIST. All experiment parameters live in `configuration/experiments.yaml`.

**Phase 1** — threshold sweep (3 values per framework, different ranges per framework).
**Phase 2** — membrane decay sweep at best threshold from Phase 1.
**Phase 3** — head-to-head at optimal settings: Tier A (150 iters/epoch, 2 seeds) and Tier B (full 937 iters/epoch, 2 seeds).

**What seeds do in this context:** `network_architecture.yaml` defines layer shapes only — never weight values. PyTorch always randomly initialises weight values when a model is constructed (this happens in regular `main.py` too, silently). The seed makes that random initialisation reproducible. Seed 42 and seed 123 = two different starting weight configurations, used in Phase 3 to confirm results aren't due to lucky initialisation.

**Known confound:** Norse uses hard reset (membrane → 0 after spike). SNNTorch and SpikingJelly use soft reset (membrane − threshold). This is documented in every run's JSON and in all plots.

**SpikingJelly spike rate normalization:** SpikingJelly's `forward()` returns `[B, C]` (summed spike counts over T) while Norse/SNNTorch return `[T, B, C]` (per-timestep 0/1). The runner divides SpikingJelly's raw `spk_rec.mean()` by T=25 to produce a comparable fraction.

**Loss function:** Cross-entropy forced for all three frameworks. SNNTorch's default `mse_count` is overridden via `_apply_overrides` before model construction.
