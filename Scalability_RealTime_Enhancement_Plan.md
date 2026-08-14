# Scalability & Real-Time-Suitability Enhancement Plan — Event Data Workflow

## 1. Problem definition

**The field-level problem** (what Prof. Bauer's brief is actually probing at,
task 5 — "Bewertung der Eignung für Echtzeitszenarien"): SNN training
pipelines are routinely benchmarked on small, fixed-resolution toy datasets
(N-MNIST, 34×34). Real deployments — robotics, automotive, security — use
higher-resolution event cameras and longer time horizons. Whether a pipeline
that works on the toy case survives that jump is rarely tested, because it's
rarely hit in academic benchmarks that stop at T=4-8 timesteps and small
sensors.

**What this project exposed, concretely, not hypothetically:** switching
from N-MNIST (34×34) to N-Caltech101 (240×180, ~46x more pixels per frame) at
the same training config produced a real `torch.OutOfMemoryError`. Root
cause, confirmed by matching hand-derived memory math against the empirical
crash point (`event_data_workflow/vram_batch_scaling_task.md`): BPTT has to
hold every timestep's activations in memory simultaneously for the backward
pass, so memory scales with sensor resolution × timesteps × batch size — not
with neuron count alone, which is the more commonly assumed scaling factor.

## 2. Purpose / goal / objective

- **Purpose:** make this project's scalability and real-time-suitability
  claims defensible under a realistic dataset, not just the toy case.
- **Goal:** any dataset in `DATASET_REGISTRY` trains without an OOM crash,
  with a fix that is tested and documented, not asserted.
- **Objective (measurable):** N-Caltech101 trains end-to-end at a stated
  effective batch size, verified via `diagnostics/verify_dataset_load.py`,
  with peak VRAM logged before/after.

## 3. Naive baseline resolution (already validated, not yet applied)

The simplest fix, already empirically confirmed in
`vram_batch_scaling_task.md`: shrink the per-dataset batch size.
`batch_size=16` completes real forward+backward passes on this 8GB card;
`batch_size=128` (the current global default) OOMs. This is a one-line
registry change (`dataset_registry.py`'s N-Caltech101 entry, currently
`"batch_size": None`), not a design problem. Two already-wired (not
hypothetical) mechanisms pair with it to keep training dynamics sane at a
smaller physical batch:

- `GRAD_ACCUM_STEPS` — real, used in `training.py:202`. Lets a smaller
  physical batch simulate a larger effective batch size via accumulation.
- `USE_AMP` — real, full `autocast`+`GradScaler` wiring in
  `training.py:118-119,206`. Already halves memory for the ops it covers.

**This tier is hours of work, not days — infrastructure already exists,
it just hasn't been pointed at this specific dataset yet.**

## 4. Grounding in real research — two tiers, correctly separated

Earlier in this project's exploration, a different AI's response fabricated
a claim that this bottleneck is secretly "solved" by unpublished industry IP.
That was false and has been discarded. The real literature, checked just
now, is public, real, and directly on-topic:

**Tier A — tactical, minimal architecture change, works within existing BPTT:**
- **Gradient checkpointing** (Chen et al. 2016, "Training Deep Nets with
  Sublinear Memory Cost") — don't store most intermediate activations;
  recompute them during backward instead. PyTorch-native
  (`torch.utils.checkpoint`). Trades backward-pass compute time for memory.
  Not yet implemented anywhere in this codebase (verified: no
  `torch.utils.checkpoint` usage in `learning/training.py`).

**Tier B — architectural, replaces the learning rule itself, constant memory
regardless of timestep count:**
- **OTTT (Online Training Through Time)** — forward-in-time learning via
  presynaptic activity tracking; memory cost is constant, not linear in T.
- **E-prop (eligibility propagation)** — biologically-motivated online rule,
  alternative to BPTT; also constant memory vs. BPTT's linear growth.
- **FPTT (Forward Propagation Through Time)** — reports 4-5x lower memory
  and 3-4x faster training than BPTT in its own published results.

Tier A is a bounded engineering task on top of the current design. Tier B is
a genuine research-grade change — it doesn't extend the current BPTT+SG
training loop, it replaces the credit-assignment method entirely, and would
need per-framework compatibility investigation (SNNTorch/Norse/SpikingJelly/
Sinabs each have different internal state handling, as the existing
CUDA-graph/`Leaky`-neuron incompatibility already found in this project
shows). **Scope Tier B as a stretch goal for later, not this semester's
committed deliverable** — directly in line with the professor's own caution
about not conflicting with thesis timing.

## 5. Where this plugs into the existing architecture

Good news, found while reviewing `ModelInterface`: every backend already
reports a `credit_assignment()` value (confirmed in
`outputs/data/training_results.csv`'s `credit_assignment` column — currently
always `"BPTT+SG"`). The interface already has a slot for describing which
training/credit-assignment strategy a given model uses. A checkpointed-BPTT
or (later) OTTT-based model could report a different value there without the
trainer/tester code needing to know which strategy backs any given model —
the extension point already exists, it's just only ever been populated with
one value so far.

## 6. Test-before-code strategy (as requested — test space before production code)

Before touching `learning/training.py`:
1. Build a small standalone harness (same shape as
   `diagnostics/verify_dataset_load.py`) that runs one backend, one dataset,
   at increasing sensor resolution, logging peak VRAM and final-epoch
   accuracy for: (a) current plain BPTT, (b) BPTT + gradient checkpointing.
2. Confirm accuracy is unchanged (checkpointing must be numerically
   equivalent, just recomputed — a real risk to verify, not assume) and
   measure the actual memory reduction and backward-pass time cost.
3. Only after that comparison exists, wire the verified-working version into
   `training.py` for real.

## 7. Phased implementation plan

| Phase | Scope | Estimated effort |
|---|---|---|
| 1 | Set `batch_size`/`grad_accum_steps` for N-Caltech101 in the registry, verify via `diagnostics/verify_dataset_load.py` | Hours |
| 2 | Build the Tier-A test harness (§6), implement gradient checkpointing, verify per-backend (start with one backend, expand once proven) | 1-2 days |
| 3 (stretch, not committed) | Investigate OTTT/e-prop/FPTT adoption feasibility against this project's 4 backends | Multi-week research scope — separate from this semester's timeline |

## 8. Confirming the two claims raised in discussion

**"Someone who wants to bring a new dataset can easily configure it" — true,
with a calibrated caveat.** Verified: `WindowedRecordingDataset` is a real,
generic wrapper already reused for two different datasets (DSEC, DAVIS Camera
Pose) via a small per-dataset loader function
(`dataset_registry.py`'s `_load_dsec`/`_load_davis_pose`). For a dataset
`tonic` already supports natively, adding it really is a few registry-entry
lines. For a dataset needing custom raw-format parsing, it's real but
bounded work — and this project has an honest example of declining that work
when it wasn't worth it: DHP19 was investigated and explicitly **not**
added, because it required translating a MATLAB toolbox
(`event_data_workflow/dataset_registry_changes.md`). So: "easily
configurable" is true for tonic-supported formats, "possible but real work"
for custom ones — don't claim uniform ease without that distinction.

**"Anyone with a PyTorch setup... can simply utilize our setup" — true for
PyTorch backends, true with a real caveat beyond that.**
`learning/frameworks/README.md` confirms: PyTorch backends (SNNTorch, Norse,
SpikingJelly, Sinabs) work natively, full gradient flow, full adversarial
robustness support. Non-PyTorch backends (JAX, TensorFlow) are also
documented as pluggable via DLPack zero-copy tensor exchange — real, not
aspirational — but adversarial robustness (FGSM/PGD) drops to clean-eval-only
for those, since PyTorch's autograd can't trace through XLA/TF's graph. So:
"PyTorch → just works" is fully true; "any other framework → just works" is
true for the forward/training path, with one named, documented limitation
for adversarial evaluation specifically.

## Presentation note

This document is planning material — the two-slide diagnostics/memory-scaling
deck already built (`Presentation_EventDataWorkflow.pptx`) stays as-is for
this week. Tier B (OTTT/e-prop/FPTT) belongs with the Brian2CUDA plan as
later-phase material, not this week's talk.
