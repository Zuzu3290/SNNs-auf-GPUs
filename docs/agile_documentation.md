# Agile Documentation — SNNs-auf-GPUs
**Date:** 2026-05-31 | **Branch:** 21-Deep-Learning | **Owner:** Muhammad Zuhair

> Companion to [v_model_diagnosis.md](v_model_diagnosis.md). The V-model maps *what* the system must be; this document maps *how* the work is structured and prioritized iteratively.

---

## Product Vision

> **Enable Spiking Neural Networks to run on commodity GPU hardware for real-time neuromorphic inference in automotive ADAS contexts, using DVS event camera data, without requiring specialist neuromorphic chips.**

**Who benefits:** Automotive perception researchers, ADAS system engineers, neuromorphic ML practitioners.
**What it replaces:** Dependence on dedicated neuromorphic hardware (Intel Loihi, SpiNNaker) for SNN execution.
**How we know we succeeded:** Competitive classification accuracy on real DVS datasets, measured energy efficiency, and a reproducible GPU-native SNN runtime.

---

## Roles (Personas)

| Role | Description |
|------|-------------|
| **Researcher** | Muhammad Zuhair — runs experiments, tunes hyperparameters, interprets results |
| **ADAS Stakeholder** | Professor / Mercedes context — defines acceptance criteria for spatial perception |
| **Future Developer** | Anyone extending the codebase (new backend, new dataset, new kernel) |
| **CI System** | Automated pipeline that validates correctness on every commit |

---

## Epics

| ID | Epic | V-Model Level | Status |
|----|------|---------------|--------|
| E1 | Event Data Pipeline | Architecture + Implementation | Done |
| E2 | SNN Learning Framework | Architecture + Implementation | Done |
| E3 | Compiler & GPU Acceleration | Architecture + Implementation | Done |
| E4 | Unit Test Coverage | Unit Tests | In Progress (compiler only) |
| E5 | Integration Test Suite | Integration Tests | Not Started |
| E6 | Requirements Documentation | User + Technical Requirements | Not Started |
| E7 | System Verification | System Verification | Not Started |
| E8 | ADAS Validation | Validation | Not Started |

---

## Product Backlog

Stories are listed under each epic, ordered by priority within the epic. Each story has an ID, acceptance criteria, and a link to the V-model level it satisfies.

---

### Epic E1 — Event Data Pipeline

---

**E1-S1 — Load NMNIST through the full pipeline** ✅ Done
> As a **Researcher**, I want to load NMNIST data through `NeuromorphicEncoder` so that I can run a training loop end-to-end.

*Acceptance Criteria:*
- `NeuromorphicEncoder.get_dataloaders()` returns `(train_loader, test_loader)` without error
- Tensor shape is `[T, B, C, H, W]` where T = time steps, B = batch size
- DataLoader respects `batch_size` from `SNN_module.yaml`

*Implemented in:* [`event_data_workflow/data_pipeline.py`](../event_data_workflow/data_pipeline.py)

---

**E1-S2 — Adaptive cache selects GPU / RAM / disk automatically** ✅ Done
> As a **Researcher**, I want the cache engine to choose the best storage tier based on available VRAM so that memory pressure does not crash training.

*Acceptance Criteria:*
- S3-FIFO policy evicts correctly when cache is full
- Phase-based GPU budget (warmup=5%, train=10%, backward=5%, eval=25%, inference=30%) is enforced
- Falls back to disk cache when GPU VRAM is under pressure

*Implemented in:* [`event_data_workflow/cache_engine.py`](../event_data_workflow/cache_engine.py)

---

**E1-S3 — Temporal slicing is configurable** ✅ Done
> As a **Researcher**, I want to configure event stream slice duration via YAML so that I can experiment with different temporal resolutions.

*Acceptance Criteria:*
- `TemporalSliceConfig` reads from `data_workflow.yaml`
- `AdaptiveTemporalSlicer` recommends slice duration based on dataset statistics
- `create_sliced_dataset()` wraps with `tonic.SlicedDataset` correctly

*Implemented in:* [`event_data_workflow/temporal_slicer.py`](../event_data_workflow/temporal_slicer.py)

---

**E1-S4 — Load a real DVS / DAVIS dataset** 🔴 Not Started
> As an **ADAS Stakeholder**, I want to run the pipeline on DAVIS event camera data from the professor's dataset so that results reflect real sensor inputs, not a proxy benchmark.

*Acceptance Criteria:*
- `NeuromorphicEncoder` loads DAVIS recordings without shape errors
- Spatial labels (depth / object class) are accessible from the DataLoader
- A sample output file or notebook shows the loaded data visually

*V-Model:* Validation (UR-01)

---

### Epic E2 — SNN Learning Framework

---

**E2-S1 — Train with Norse backend on NMNIST** ✅ Done
> As a **Researcher**, I want to run a full training loop using the Norse backend so that I have a working baseline.

*Implemented in:* [`src/learning/training.py`](../src/learning/training.py), [`src/learning/frameworks/snn_norse.py`](../src/learning/frameworks/snn_norse.py)

---

**E2-S2 — Switch framework via config without changing training code** ✅ Done
> As a **Researcher**, I want to change the `simulator` field in `SNN_module.yaml` to swap between Norse, SNNTorch, and SpikingJelly so that I can compare frameworks without touching core code.

*Acceptance Criteria:*
- Setting `simulator: snntorch` / `norse` / `spikingjelly` selects the correct backend
- All three backends satisfy `ModelInterface` (forward, backward_pass, zero_grad, train_mode, eval_mode, get_state)
- Training loss is logged identically regardless of backend

*Implemented in:* [`src/learning/frameworks/model_interface.py`](../src/learning/frameworks/model_interface.py)

---

**E2-S3 — Adversarial robustness evaluation (FGSM / PGD / TRADES)** ✅ Done
> As an **ADAS Stakeholder**, I want adversarial attack evaluation so that I can quantify model robustness for safety-critical use.

*Implemented in:* [`src/learning/adversarial_robustness.py`](../src/learning/adversarial_robustness.py)

---

**E2-S4 — Activity regularization prevents dead / saturated neurons** ✅ Done
> As a **Researcher**, I want activity regularization hooks to keep hidden neuron firing rates within a healthy band so that training does not collapse.

*Implemented in:* [`src/learning/frameworks/activity_reg.py`](../src/learning/frameworks/activity_reg.py)

---

**E2-S5 — Per-class metrics and confusion matrix at evaluation** ✅ Done
> As a **Researcher**, I want per-class precision, recall, and F1 scores plus a confusion matrix CSV so that I can diagnose class-specific failures.

*Implemented in:* [`src/learning/inference.py`](../src/learning/inference.py)

---

**E2-S6 — Benchmark accuracy against published SNN results** 🔴 Not Started
> As an **ADAS Stakeholder**, I want a comparison table of our accuracy vs. published SNN benchmarks on the same dataset so that the system's competitiveness is demonstrable.

*Acceptance Criteria:*
- Accuracy on NMNIST compared against at least two published results
- Comparison table stored in `docs/validation/accuracy_benchmark.md`
- Epsilon sweep for adversarial robustness plotted and saved

*V-Model:* Validation (UR-03, UR-04)

---

### Epic E3 — Compiler & GPU Acceleration

---

**E3-S1 — Compiler IR represents a model as a ComputeGraph** ✅ Done
> As a **Future Developer**, I want SNN operations represented as an IR so that hardware-specific optimizations are decoupled from the framework layer.

*Implemented in:* [`src/compiler/src/ir.py`](../src/compiler/src/ir.py)

---

**E3-S2 — Compiler passes run in order: rewrite → annotate → fuse** ✅ Done
> As a **Future Developer**, I want compiler passes to be composable and ordered so that new passes can be added without breaking existing ones.

*Implemented in:* [`src/compiler/src/scheduler.py`](../src/compiler/src/scheduler.py), [`src/compiler/passes/`](../src/compiler/passes/)

---

**E3-S3 — Runtime dispatches LIF quartet to custom CUDA kernel** ✅ Done
> As a **Researcher**, I want the fused LIF kernel (membrane update + threshold + spike gen + reset) to execute as a single CUDA kernel so that GPU throughput is maximized.

*Implemented in:* [`src/compiler/src/runtime.py`](../src/compiler/src/runtime.py), [`src/compiler/kernels/lif_kernel.cu`](../src/compiler/kernels/lif_kernel.cu)

---

**E3-S4 — Runtime falls back to PyTorch when CUDA is unavailable** ✅ Done
> As a **Future Developer**, I want the runtime to fall back to a pure-PyTorch LIF when the CUDA kernel is not loaded so that CPU development and CI are unblocked.

*Acceptance Criteria:*
- FusedStep tries CUDA kernel, then framework neuron, then pure-Python LIF
- No error is raised on a CPU-only machine

*Implemented in:* [`src/compiler/src/runtime.py`](../src/compiler/src/runtime.py)

---

**E3-S5 — Measure real GPU power via nvidia-ml-py** ✅ Done
> As a **Researcher**, I want per-epoch GPU power readings so that energy consumption is tracked during training.

*Implemented in:* [`event_data_workflow/gpu_stats.py`](../event_data_workflow/gpu_stats.py)

---

**E3-S6 — Measure energy per inference (not estimated)** 🔴 Not Started
> As an **ADAS Stakeholder**, I want measured energy per inference sample (not the hardcoded 3.5 pJ proxy) so that the energy efficiency claim is backed by real data.

*Acceptance Criteria:*
- `nvidia-ml-py` power readings integrated into inference loop
- Energy per sample = (mean power draw during inference run) × (inference latency per sample)
- Results saved to `outputs/energy_profile.csv`
- Compare against GPU baseline (standard ANN on same task)

*V-Model:* Validation (UR-05) | System Verification (TR-11)

---

### Epic E4 — Unit Test Coverage

---

**E4-S1 — Compiler IR unit tests** ✅ Done
> As a **CI System**, I want tests for the ComputeGraph so that IR construction regressions are caught immediately.

*Existing:* [`src/compiler/tests/test_ir.py`](../src/compiler/tests/test_ir.py)

---

**E4-S2 — Compiler scheduler and runtime unit tests** ✅ Done
> As a **CI System**, I want tests for the scheduler pass chain and runtime execution so that compiler regressions are caught.

*Existing:* [`src/compiler/tests/test_scheduler.py`](../src/compiler/tests/test_scheduler.py), [`src/compiler/tests/test_runtime.py`](../src/compiler/tests/test_runtime.py)

---

**E4-S3 — Cache engine unit tests** 🔴 Not Started
> As a **CI System**, I want unit tests for `AdaptiveCacheController` and `BaseS3FIFOCache` so that eviction policy regressions are caught without needing a GPU.

*Acceptance Criteria:*
- S3-FIFO evicts the correct item when cache is full (small / main queue logic)
- `compute_gpu_cache_budget()` returns values within phase caps for mocked VRAM readings
- `AdaptiveCacheController` selects `MemoryCachedDataset` when RAM is sufficient and `DiskCachedDataset` when it is not
- All tests pass on CPU (no GPU required)

*File to create:* `tests/unit/test_cache_engine.py`
*V-Model:* Unit Tests ↔ Detailed Design of `cache_engine.py`

---

**E4-S4 — Activity regularization unit tests** 🔴 Not Started
> As a **CI System**, I want unit tests for `activity_regularization()` and `stdp_regularization()` so that silent gradient regressions are caught.

*Acceptance Criteria:*
- `activity_regularization()` returns zero loss when firing rate is within `[min_rate, max_rate]`
- Returns positive loss when rate is outside bounds
- `stdp_regularization()` returns a scalar tensor with a valid gradient
- Hook registration and removal do not leave dangling state

*File to create:* `tests/unit/test_activity_reg.py`
*V-Model:* Unit Tests ↔ Detailed Design of `activity_reg.py`

---

**E4-S5 — Temporal slicer unit tests** 🔴 Not Started
> As a **CI System**, I want unit tests for `AdaptiveTemporalSlicer` so that slice duration recommendation logic is verifiable.

*Acceptance Criteria:*
- Slicer recommends a longer duration for sparse event streams
- `create_sliced_dataset()` wraps a mock dataset without error
- Edge case: zero-length stream handled gracefully

*File to create:* `tests/unit/test_temporal_slicer.py`
*V-Model:* Unit Tests ↔ Detailed Design of `temporal_slicer.py`

---

**E4-S6 — Config parsing unit tests** 🟡 Low Priority
> As a **CI System**, I want unit tests for `Settings` (YAML parsing, architecture generation) so that a broken config fails fast.

*File to create:* `tests/unit/test_snn_config.py`

---

### Epic E5 — Integration Test Suite

---

**E5-S1 — End-to-end smoke test (CPU-only, CI-friendly)** 🔴 Not Started | **Highest Priority**
> As a **CI System**, I want a single test that runs `main.py` for 1 epoch on a tiny synthetic dataset (CPU-only) so that the full pipeline is validated on every commit.

*Acceptance Criteria:*
- Test completes in under 60 seconds on CPU
- Uses a synthetic dataset (no file download needed)
- Covers: data loading → model construction → training step → evaluation → metrics saved
- Test passes for all three backends (Norse, SNNTorch, SpikingJelly)

*File to create:* `tests/integration/test_pipeline_e2e.py`
*V-Model:* Integration Tests ↔ Architecture Design

---

**E5-S2 — Data pipeline → learning module boundary test** 🔴 Not Started
> As a **CI System**, I want a test confirming that the tensor shape and dtype produced by `NeuromorphicEncoder` is consumed correctly by `SNNTrainer` without reshape errors.

*Acceptance Criteria:*
- Tensor shape `[T, B, C, H, W]` flows through without manual reshaping
- dtype is `float32` at model input
- Test fails if shape convention breaks (catches future refactor regressions)

*File to create:* `tests/integration/test_data_to_training_boundary.py`
*V-Model:* Integration Tests (IT-01)

---

**E5-S3 — Compiler CUDA / PyTorch parity test** 🔴 Not Started
> As a **Future Developer**, I want a test that runs the same input through the FusedStep (CUDA) and AtomicStep (PyTorch) paths and confirms numerical equivalence so that kernel correctness is verifiable.

*Acceptance Criteria:*
- Output spike tensors agree to within `1e-4` absolute tolerance
- Membrane voltage at each timestep agrees to within `1e-4`
- Test is skipped if CUDA is unavailable (not failed)

*File to create:* `tests/integration/test_compiler_cuda_parity.py`
*V-Model:* Integration Tests (IT-03)

---

**E5-S4 — Framework swap produces equivalent output shapes** 🔴 Not Started
> As a **CI System**, I want a test that instantiates all three backends with identical config and confirms that output tensor shapes and loss computation paths are equivalent.

*File to create:* `tests/integration/test_framework_swap.py`
*V-Model:* Integration Tests (IT-02)

---

**E5-S5 — Config propagation integration test** 🟡 Medium Priority
> As a **CI System**, I want a test that loads `SNN_module.yaml` via `Settings` and confirms that all downstream modules (trainer, tester, encoder, compiler) receive the correct values.

*File to create:* `tests/integration/test_config_propagation.py`
*V-Model:* Integration Tests (IT-04)

---

### Epic E6 — Requirements Documentation

---

**E6-S1 — Write User Requirements Specification** 🔴 Not Started
> As an **ADAS Stakeholder**, I want a formal document listing what the system must achieve in plain language so that research progress can be evaluated against stated goals.

*Acceptance Criteria:*
- Each UR has a unique ID (UR-01..UR-N), a rationale, and a verifiability note
- Document is reviewed and acknowledged by supervisor / professor
- Stored at `docs/requirements/user_requirements.md`

---

**E6-S2 — Write System Requirements Specification with pass criteria** 🔴 Not Started
> As a **Future Developer**, I want a document listing technical requirements with measurable pass/fail criteria so that system verification is unambiguous.

*Acceptance Criteria:*
- Each TR has a unique ID, a metric (e.g., "GPU VRAM usage during train phase ≤ 10% of total VRAM"), and a verification method
- Traceability column links each TR to one or more URs
- Stored at `docs/requirements/system_requirements.md`

---

### Epic E7 — System Verification

---

**E7-S1 — GPU budget enforcement test** 🔴 Not Started
> As a **Researcher**, I want a test that simulates GPU memory pressure and confirms the phase-based caps are enforced so that TR-03 is verified.

*File to create:* `tests/system/test_gpu_budget_enforcement.py`

---

**E7-S2 — S3-FIFO hit-rate benchmark** 🟡 Medium Priority
> As a **Researcher**, I want a benchmark comparing S3-FIFO hit rate against LRU on a representative access pattern so that the cache policy choice is justified.

*File to create:* `tests/system/test_cache_policy_benchmark.py`

---

**E7-S3 — Metrics correctness cross-validation** 🟡 Medium Priority
> As a **CI System**, I want per-class F1 / precision / recall from `SNNTester` to be validated against `sklearn.metrics` on a known dataset so that metric computation bugs are ruled out.

*File to create:* `tests/system/test_metrics_correctness.py`

---

### Epic E8 — ADAS Validation

---

**E8-S1 — Run on DAVIS dataset, record results** 🔴 Not Started
> As an **ADAS Stakeholder**, I want a documented run on the professor's DAVIS dataset with spatial labels so that UR-01 (DVS event processing) is validated.

*Acceptance Criteria:*
- Results (accuracy, latency, energy) saved in `docs/validation/davis_results.md`
- Sample event frames visualised and committed to `docs/validation/`

---

**E8-S2 — Publish accuracy comparison table** 🔴 Not Started
> Satisfies UR-03. Results table in `docs/validation/accuracy_benchmark.md`.

---

**E8-S3 — Measured energy efficiency vs. ANN baseline** 🔴 Not Started
> Satisfies UR-05. Depends on E3-S6 (real power measurement).

---

## Sprint Plan

Suggested 2-week sprints, ordered to unblock CI first, then close V-model gaps from the bottom up.

| Sprint | Goal | Stories |
|--------|------|---------|
| **Sprint 1** | Green CI on every commit | E5-S1 (e2e smoke test), E4-S3 (cache unit tests), E4-S4 (activity reg unit tests) |
| **Sprint 2** | Close unit test gaps | E4-S5 (temporal slicer), E4-S6 (config), E5-S2 (data→training boundary) |
| **Sprint 3** | Compiler & kernel verification | E5-S3 (CUDA parity), E5-S4 (framework swap), E5-S5 (config propagation) |
| **Sprint 4** | Requirements docs | E6-S1 (URS), E6-S2 (SRS with pass criteria) |
| **Sprint 5** | System verification | E7-S1 (GPU budget), E7-S3 (metrics correctness), E3-S6 (real energy measurement) |
| **Sprint 6** | ADAS validation | E1-S4 (DAVIS dataset), E8-S1, E8-S2, E8-S3 |

---

## Definition of Done

A story is **Done** when all of the following are true:

- [ ] Code is committed on the feature branch and passes `git status` clean
- [ ] All acceptance criteria are met and manually verified
- [ ] For test stories: tests pass locally (`pytest`) and no existing test is broken
- [ ] For feature stories: at least one unit or integration test covers the new behaviour
- [ ] For documentation stories: document reviewed and saved in `docs/`
- [ ] No TODO comments left in committed code
- [ ] `SNN_module.yaml` or `data_workflow.yaml` updated if new config keys were added

---

## Backlog Health

| Category | Total Stories | Done | In Progress | Not Started |
|----------|--------------|------|-------------|-------------|
| E1 — Data Pipeline | 4 | 3 | 0 | 1 |
| E2 — Learning | 6 | 5 | 0 | 1 |
| E3 — Compiler & GPU | 6 | 5 | 0 | 1 |
| E4 — Unit Tests | 6 | 2 | 0 | 4 |
| E5 — Integration Tests | 5 | 0 | 0 | 5 |
| E6 — Requirements Docs | 2 | 0 | 0 | 2 |
| E7 — System Verification | 3 | 0 | 0 | 3 |
| E8 — ADAS Validation | 3 | 0 | 0 | 3 |
| **Total** | **35** | **15** | **0** | **20** |

**Completion: 15 / 35 stories (43%)**

The implementation plane is ~90% done. The verification and validation plane is almost entirely open — which is the expected state for a research project that prioritised getting the system working before hardening it.
