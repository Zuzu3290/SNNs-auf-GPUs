# V-Model Diagnosis — SNNs-auf-GPUs
**Date:** 2026-05-31 | **Branch:** 21-Deep-Learning | **Author:** Muhammad Zuhair

---

## Overview

This document maps the current project state onto the V-model — a systems engineering framework that pairs every specification level on the left arm with a corresponding verification or validation activity on the right arm. Each level is assessed for what **already exists** in the project and what is **missing or incomplete**.

```
User Requirements        ◄──────────────────────► Validation
       │                                                ▲
       ▼                                                │
 Technical Requirements  ◄──────────────────────► System Verification
       │                                                ▲
       ▼                                                │
 Architecture Design     ◄──────────────────────► Integration Tests
       │                                                ▲
       ▼                                                │
 Detailed Design         ◄──────────────────────► Unit Tests
       │                                                ▲
       └──────────────► Implementation ─────────────────┘
```

---

## Left Arm — Specification

---

### Level 1 — User Requirements

**Definition:** What the end stakeholder needs the system to achieve, independent of how it is built.

**Context for this project:**
- Automotive ADAS application (Mercedes spatial perception context)
- DVS (Dynamic Vision Sensor) event camera input pipeline
- Dataset provided by the university professor
- Goal: prove that SNNs can run efficiently on commodity GPU hardware for real-time neuromorphic inference

| # | Requirement | Status |
|---|-------------|--------|
| UR-01 | Process event-camera (DVS) data streams for spatial perception | **Implicit** — implemented but not formally stated |
| UR-02 | Demonstrate SNN inference on GPU without specialist neuromorphic hardware | **Implicit** — core thesis of the project |
| UR-03 | Achieve competitive classification accuracy on standard neuromorphic datasets | **Implicit** — NMNIST used as proxy benchmark |
| UR-04 | Support adversarial robustness evaluation for safety-critical automotive use | **Implicit** — TRADES/FGSM/PGD implemented |
| UR-05 | Provide energy efficiency estimates per inference sample | **Implicit** — 3.5 pJ/spike model in inference.py |
| UR-06 | System must be extensible to multiple SNN frameworks | **Implicit** — three backends integrated |

**Existing artifacts:**
- [`README.md`](../README.md) — project motivation and two-plane architecture overview
- [`docs/NN/NN.md`](NN/NN.md) — AI research philosophy and framing
- [`docs/Haseeb-open-items.md`](Haseeb-open-items.md) — open items list

**Gap:** No formal User Requirements Specification (URS) document exists. Requirements are scattered across READMEs and implied by implementation choices. No stakeholder sign-off record.

---

### Level 2 — Technical Requirements

**Definition:** What the system must do technically to satisfy user requirements — measurable, testable system-level properties.

| # | Requirement | Source | Status |
|---|-------------|--------|--------|
| TR-01 | Must support NMNIST, DAVIS, and custom event datasets | `event_data_workflow/data_pipeline.py` | **Implemented** |
| TR-02 | Must support Norse, SNNTorch, SpikingJelly, Sinabs backends interchangeably | `src/learning/frameworks/` | **Implemented** |
| TR-03 | GPU VRAM usage must stay within configurable phase-based budgets | `event_data_workflow/cache_engine.py` | **Implemented** |
| TR-05 | Adaptive cache must evict correctly under a bounded RAM/VRAM budget | `event_data_workflow/cache_engine.py` | **Implemented** — plain FIFO, not S3-FIFO (see below) |
| TR-07 | System must produce per-class precision, recall, F1 at evaluation | `src/learning/inference.py` | **Implemented** |
| TR-08 | Adversarial attacks (FGSM, PGD, TRADES) must be framework-agnostic | `src/learning/adversarial_robustness.py` | **Implemented** |
| TR-09 | Configuration must be driven by a single YAML file | `SNN_module.yaml`, `skeleton/snn_config.py` | **Implemented** |
| TR-10 | System must operate on Windows 11 / CUDA GPU environment | Environment config, `requirements.txt` | **Implemented** |
| TR-11 | Real-time GPU power readings via nvidia-ml-py | `event_data_workflow/gpu_stats.py` | **Implemented** |
| TR-12 | Temporal slicing of event streams must be configurable | `event_data_workflow/data_pipeline.py` | **Implemented** |

**Since revised:** TR-04 ("LIF dynamics must be computable via custom CUDA kernel")
is removed — `src/crsc/` no longer exists in this repo (removed in a prior cleanup
pass alongside `src/compiler/` and `acceleration/`; git history has it if that
work resumes). All backends currently run their neuron dynamics through their own
framework (Norse/SNNTorch/SpikingJelly/Sinabs), not a custom kernel. TR-05 was
originally written assuming S3-FIFO specifically; the requirement itself (correct
eviction under a bounded budget) still holds, just via plain FIFO now — see
[`event_data_workflow/caching_pipeline_refactor.md`](event_data_workflow/caching_pipeline_refactor.md)
for why.

**Existing artifacts:**
- [`SNN_module.yaml`](../SNN_module.yaml) — quantitative thresholds (LR, threshold voltage, TRADES ε, activity reg bounds)
- [`event_data_workflow/data_workflow.yaml`](../event_data_workflow/data_workflow.yaml) — cache memory thresholds
- [`requirements.txt`](../requirements.txt) — dependency constraints with versions

**Gap:** No dedicated System Requirements Specification (SRS) document with traceability IDs. Parameter bounds (e.g., maximum acceptable latency, minimum classification accuracy threshold) are not formally defined as pass/fail criteria.

---

### Level 3 — Architecture Design

**Definition:** How the system is decomposed into major subsystems and how they interact.

**Existing artifacts (well-covered):**
- [`src/README.md`](../src/README.md) — overall architecture and data flow narrative
- [`src/learning/README.md`](../src/learning/README.md) — learning module structure
- [`event_data_workflow/README.md`](../event_data_workflow/README.md) — pipeline usage and limitations
- [`skeleton/README.md`](../skeleton/README.md) — configuration bridge role

**Architecture subsystems identified:**

```
┌─────────────────────────────────────────────────────┐
│                   SNNs-auf-GPUs                     │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │  Event Data  │    │      SNN Learning         │  │
│  │  Workflow    │───►│  (training + inference)   │  │
│  │              │    │                            │  │
│  │ data_pipeline│    │ SNNTrainer / SNNTester     │  │
│  │ cache_engine │    │ AdversarialEvaluator       │  │
│  │ prefetch     │    │ Frameworks: Norse /        │  │
│  │              │    │  SNNTorch / SpikingJelly /  │  │
│  │              │    │  Sinabs                    │  │
│  └──────────────┘    └───────────┬────────────────┘  │
│                                  │                   │
│  ┌──────────────┐                                    │
│  │  Skeleton    │                                    │
│  │  (config)    │                                    │
│  │              │                                    │
│  │ snn_config   │                                    │
│  │ snn_logging  │                                    │
│  └──────────────┘                                    │
└─────────────────────────────────────────────────────┘
```

**Since revised:** `src/crsc/` (the standalone CUDA LIF kernel this diagram
previously showed) no longer exists in this repo — removed in a prior cleanup
pass, along with `src/compiler/` and `acceleration/`. All backends currently
compute neuron dynamics through their own framework, not a custom kernel.
`temporal_slicer.py` was folded into `data_pipeline.py` in a later session (see
[`event_data_workflow/caching_pipeline_refactor.md`](event_data_workflow/caching_pipeline_refactor.md)).

**Gap:** No formal Architecture Description Document (ADD) with interface control tables. Inter-subsystem boundaries are described narratively but not specified with explicit API contracts or data format specifications.

---

### Level 4 — Detailed Design

**Definition:** Module-level design — class diagrams, data flows within each subsystem, interface specifications.

**Existing artifacts:**
- [`docs/event_data_workflow/caching_pipeline_refactor.md`](event_data_workflow/caching_pipeline_refactor.md) — cache engine design, current as of the refactor that removed `pipeline_coordinator.py`/`temporal_slicer.py`
- [`docs/event_data_workflow/Dataset_workflow.md`](event_data_workflow/Dataset_workflow.md)
- [`docs/frameworks/README.md`](frameworks/README.md)
- [`docs/Hardware/`](Hardware/) — GPU, CPU, TPU resource notes
- [`docs/Camera/`](Camera/) — Event-based camera documentation
- [`src/learning/frameworks/model_interface.py`](../src/learning/frameworks/model_interface.py) — formal ABC contract for all SNN backends (closest thing to a detailed design spec for the framework boundary)

**Modules with documented design:**
| Module | Documentation | Quality |
|--------|---------------|---------|
| `event_data_workflow/cache_engine.py` | README + refactor report | Moderate |
| `event_data_workflow/data_pipeline.py` | Refactor report | Moderate |
| `src/learning/frameworks/` | model_interface.py ABC | Good |

**Gap:** No class diagrams or formal UML for the cache engine. No data format specification for the event tensor shape conventions (T × B × C × H × W) used at the boundary between the data pipeline and the learning module.

---

## Bottom — Implementation

**Status: Primary artifact — largely complete**

| Component | Files | Completeness |
|-----------|-------|--------------|
| SNN Backends | `src/learning/frameworks/` (Norse, SNNTorch, SpikingJelly, Sinabs) | Complete |
| Training Loop | `src/learning/training.py` | Complete |
| Inference & Metrics | `src/learning/inference.py` | Complete |
| Adversarial Robustness | `src/learning/adversarial_robustness.py` | Complete |
| Activity Regularization | `src/learning/frameworks/activity_reg.py` | Complete |
| STDP Loss | `src/learning/frameworks/activity_reg.py` | Complete |
| Data Pipeline | `event_data_workflow/data_pipeline.py` | Complete |
| Adaptive Cache | `event_data_workflow/cache_engine.py` | Complete |
| Async Prefetch | `event_data_workflow/prefetch.py` | Complete |
| System Monitor | `event_data_workflow/system_monitor.py` | Complete |
| Configuration Bridge | `skeleton/snn_config.py` | Complete |
| Entry Point | `src/learning/main.py` | Complete |

**Since revised:** the CUDA LIF kernel (`src/crsc/`) and its build system
(`CMakeLists.txt`) no longer exist — removed as unused. `temporal_slicer.py` and
`pipeline_coordinator.py` were split up in a later session (see the refactor
report); their responsibilities now live in `data_pipeline.py`, `prefetch.py`,
and `src/learning/frameworks/activity_reg.py`. BindsNET and Spyx backends were
also explored and later dropped — see `docs/frameworks/additional_frameworks.md`.

---

## Right Arm — Verification & Validation

---

### Level 4 — Unit Tests

**Definition:** Tests that verify each module in isolation against its detailed design.

**Existing unit tests: none.** The only unit tests the project had lived in `src/compiler/tests/`, testing the compiler IR/scheduler/runtime subsystem. That subsystem was removed (unused — nothing in the four-framework comparison depended on it), and its tests were removed with it. No replacement test suite exists yet for anything currently in the codebase.

**Missing unit tests (not present):**

| Module | Priority | What to Test |
|--------|----------|-------------|
| `event_data_workflow/cache_engine.py` | High | FIFO eviction order, GPU budget computation, phase caps, transform/live_transform freshness split |
| `event_data_workflow/data_pipeline.py` | High | dataloader_config() worker sizing, temporal slicing boundary conditions |
| `event_data_workflow/prefetch.py` | Medium | AsyncGPUPrefetcher stays N batches ahead, propagates loader exceptions |
| `src/learning/frameworks/activity_reg.py` | High | Activity penalty values, dead neuron detection, STDP loss gradient |
| `src/learning/training.py` | Medium | Aggregate spike output shape normalization, checkpoint save/load |
| `src/learning/inference.py` | Medium | Per-class metric computation, confusion matrix |
| `skeleton/snn_config.py` | Medium | YAML parsing, architecture generation correctness |
| `src/learning/adversarial_robustness.py` | Medium | FGSM perturbation bounds, PGD convergence |

**Coverage summary:** 0 unit test files — 0% subsystem coverage.

---

### Level 3 — Integration Tests

**Definition:** Tests that verify subsystem interactions — does the data pipeline correctly feed the learning module, do framework swaps preserve behavior, does the hardware-configuration picker resolve correctly?

**Existing integration tests:** **None** as an automated suite. A synthetic-data sample test for the cache engine (FIFO eviction, transform freshness, CUDA gating, end-to-end strategy resolution — 18 checks) was run manually this session; not yet checked into the repo as a real test file.

**Missing integration tests (not present):**

| Test ID | Scope | What to Verify |
|---------|-------|---------------|
| IT-01 | Data pipeline → Learning module | Event tensor shape and dtype produced by `NeuromorphicEncoder` matches what `SNNTrainer` expects (T × B × C convention) |
| IT-02 | Framework swap | Training run with Norse vs SNNTorch vs SpikingJelly vs Sinabs produces equivalent output shapes and loss trajectories |
| IT-04 | Skeleton config → All modules | `Settings` loaded from `SNN_module.yaml` correctly propagates to trainer, tester, and encoder |
| IT-05 | Cache engine → GPU pressure | Under simulated GPU pressure, `AdaptiveCacheController` correctly demotes from memory/hybrid to disk (VRAM is never a candidate) |
| IT-06 | Adversarial evaluator → Framework | `AdversarialEvaluator` produces valid perturbations for each of the four SNN backends |
| IT-07 | End-to-end pipeline | Full run of `main.py` from data load to evaluation completes without error on CPU (CI-friendly smoke test) |
| IT-08 | Hardware resolution | `training.device: auto` resolves to `cuda` with adaptive `force_mode`, matching the project's single CPU-loads/GPU-trains topology |

---

### Level 2 — System Verification

**Definition:** Verifying that the integrated system satisfies the technical requirements — does the system as a whole behave within the specified parameters?

**Existing system verification activities (informal):**
- Training loss and accuracy are logged to CSV in `outputs/`
- GPU VRAM stats are tracked per epoch via `event_data_workflow/gpu_stats.py`
- Energy per sample estimated at 3.5 pJ/spike (hardcoded, not verified against hardware)
- `SNN_module.yaml` parameters are validated implicitly by Settings class
- Cache engine behavior (FIFO eviction, transform freshness) manually verified via synthetic-data test this session — not yet automated/repeatable as CI

**Missing formal verification:**

| Requirement | Verification Method Needed | Status |
|-------------|---------------------------|--------|
| TR-03: GPU budget stays within phase caps | Automated test that drives GPU near limit and checks cap enforcement | Missing |
| TR-05: FIFO eviction correctness | Checked in this session manually — needs checking into a real automated test | Partial |
| TR-07: Per-class metrics accuracy | Cross-validate against sklearn metrics on known dataset | Missing |
| TR-11: Real power readings via nvidia-ml-py | Hardware-in-loop test with known power profile | Missing |

**Gap:** No System Verification Plan (SVP). No acceptance criteria are formally written down — there is no document stating what result constitutes a passing system verification.

---

### Level 1 — Validation

**Definition:** Confirming the final system meets the original user requirements — does the product do what the stakeholder actually needed?

**Existing validation activities (informal):**
- Adversarial robustness evaluation (FGSM, PGD, TRADES) via `AdversarialEvaluator`
- Per-class F1 / precision / recall computed for NMNIST classification
- Confusion matrix generated and saved

**Missing formal validation:**

| User Requirement | Validation Evidence Needed | Status |
|-----------------|---------------------------|--------|
| UR-01: DVS event data processing | End-to-end DAVIS or real DVS dataset run with spatial labels | Partial (NMNIST only) |
| UR-02: SNN on GPU without neuromorphic hardware | Latency and throughput benchmark vs. Intel Loihi or SpiNNaker reference | Missing |
| UR-03: Competitive accuracy | Comparison table against published SNN benchmarks on same dataset | Missing |
| UR-04: Adversarial robustness for safety | Clean vs. robust accuracy table with epsilon sweep | Partial (implemented, no report template) |
| UR-05: Energy efficiency | Measured energy per inference vs. GPU baseline (not estimated) | Missing |
| UR-06: Framework extensibility | Demonstration of adding a 4th backend with no core changes | **Demonstrated** — Sinabs added as a 4th backend; BindsNET/Spyx were also added and later dropped, see `docs/frameworks/additional_frameworks.md` |

**Gap:** No Validation Plan or Acceptance Test Procedure (ATP). The adversarial evaluator and inference metrics are implemented but there is no document tying results back to stated user requirements.

---

## Summary Diagnosis

### What Exists

| V-Model Level | Artifact Quality | Coverage |
|---------------|-----------------|---------|
| User Requirements | Implicit / scattered | ~60% of requirements discoverable |
| Technical Requirements | YAML config + READMEs | ~70% specified informally |
| Architecture Design | Multiple READMEs + refactor reports | Good — ~80% covered |
| Detailed Design | Module READMEs + ABC interface | Moderate — ~60% covered |
| **Implementation** | **Full codebase** | **~95% — primary artifact** |
| Unit Tests | None | 0% subsystem coverage |
| Integration Tests | None | 0% |
| System Verification | Informal metrics logging | ~20% |
| Validation | Partial (metrics, adversarial eval) | ~30% |

### Critical Gaps (Prioritized)

1. **Integration Tests (IT-01, IT-02, IT-04 to IT-07)** — Zero coverage. The data pipeline → learning module boundary is the highest-risk point and is completely untested at the system boundary level.

2. **Unit Tests for event_data_workflow and learning** — The cache engine (FIFO eviction, transform freshness), activity regularization, and STDP loss have no *automated* tests (the cache engine has a manually-run synthetic-data sample test, not yet checked in). These are complex, stateful components where a regression would be silent.

3. **Formal System Requirements Specification** — All TR entries above exist only as behavior in code. A single SRS document with IDs, measurable pass criteria, and traceability to URs would make verification tractable.

4. **Validation plan tied to user requirements** — The Mercedes/ADAS context requires demonstrating that the system works on real event camera data (DAVIS, not just NMNIST), and that energy efficiency is measured rather than estimated.

---

## Recommended Next Steps

| Priority | Action | Maps To |
|----------|--------|---------|
| 1 | Write `tests/integration/test_pipeline_e2e.py` (IT-07 smoke test, CPU-only) | Integration Tests |
| 2 | Write `tests/unit/test_cache_engine.py` (FIFO eviction, transform freshness — formalize the synthetic sample test from this session) | Unit Tests |
| 3 | Write `tests/unit/test_activity_reg.py` (dead neuron penalty, STDP gradient) | Unit Tests |
| 4 | Create `docs/requirements/system_requirements.md` with TR IDs and pass criteria | Technical Requirements |
| 5 | Create `docs/requirements/user_requirements.md` with UR IDs and rationale | User Requirements |
| 6 | Run on DAVIS dataset and record validation results in `docs/validation/` | Validation |
| 7 | Add `tests/system/test_gpu_budget_enforcement.py` (TR-03 verification) | System Verification |
