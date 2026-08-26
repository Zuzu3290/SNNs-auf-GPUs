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

## 3. Naive baseline resolution — APPLIED AND VERIFIED (2026-08-18)

The simplest fix, already empirically confirmed in
`vram_batch_scaling_task.md`: shrink the per-dataset batch size.
`batch_size=16` completes real forward+backward passes on this 8GB card;
`batch_size=128` (the previous global default) OOMs. Applied as a registry
change (`dataset_registry.py`'s N-Caltech101 entry: `batch_size: 16,
grad_accum_steps: 8`, effective batch size 128, matching the global
default's training dynamics). `apply_dataset_hyperparams()` extended to
apply `grad_accum_steps` from the registry (it previously only handled
`epochs`/`batch_size`/`iterations` — `grad_accum_steps` wasn't threaded
through despite `GRAD_ACCUM_STEPS` already existing as a config field).

**Two real, previously-undiscovered bugs found and fixed while verifying
this end-to-end** (`diagnostics/verify_ncaltech101_batch_fix.py`), neither
hypothetical — both blocked N-Caltech101 from ever actually completing a
real training run before, independent of the batch-size question:

1. **Disk cache collision across datasets.** `AdaptiveCacheController`'s
   cache directory was keyed only by split name (`cache/train`,
   `cache/test`), not by dataset — any two datasets both using the
   `"train"`/`"test"` split labels silently shared the same cache
   directory. Confirmed as live, not theoretical: `cache/train/0_0.hdf5`
   held a `(16, 2, 34, 34)` tensor — N-MNIST's shape, from a run days
   earlier — and `DiskCachedDataset` returned it as a cache hit when
   N-Caltech101 (240x180 sensor) asked for the same index, instead of ever
   decoding the real recording. Surfaced as a shape-mismatch crash deep
   inside `measure_dense_macs()`, nowhere near the actual bug. Fixed:
   `NeuromorphicEncoder.apply_pipeline()` now namespaces every cache and
   slicing-metadata path by dataset name
   (`event_data_workflow/data_pipeline.py`). Stale un-namespaced
   `cache/train`/`cache/test` deleted (regeneratable, confirmed
   wrong-shaped).
2. **Gradient-accumulation flush double-backward.** `SNNTrainer.train()`'s
   "flush a partial final accumulation batch" branch called
   `backward_pass()` a second time on `loss_val`, whose graph had already
   been consumed by that same iteration's in-loop call —
   `RuntimeError: Trying to backward through the graph a second time`.
   Only reachable when `GRAD_ACCUM_STEPS > 1`, which defaults to `1`
   everywhere else in this project, so the branch had apparently never
   fired in a real run before. Fixed: the flush now steps the optimizer
   directly using gradients already accumulated in `.grad`, instead of
   re-invoking `.backward()` (`learning/training.py`).

**Verified, real run** (`cfg.ITERA=3`, 1 epoch, through the actual
`SNNTrainer.train()` path, not a synthetic probe): peak VRAM 6.18GB/7.96GB
(77.7%), no OOM, real per-epoch metrics reported (loss, accuracy, spike
rate, SynOps energy). `diagnostics/verify_ncaltech101_batch_fix.py` kept as
a permanent regression check.

- `USE_AMP` — real, full `autocast`+`GradScaler` wiring in
  `training.py:118-119,206`. Already halves memory for the ops it covers;
  active in the verified run above.

**This tier is done.** What looked like "hours of work, infrastructure
already exists" going in was accurate for the batch-size change itself,
but verifying it end-to-end (rather than trusting the one-line config
change in isolation) surfaced two real bugs that would have silently
produced wrong results — the cache collision in particular would have
affected any future dataset switch, not just this one.

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

| Phase | Scope | Estimated effort | Status |
|---|---|---|---|
| 1 | Set `batch_size`/`grad_accum_steps` for N-Caltech101 in the registry, verify via a real training run | Hours | **DONE** (2026-08-18) — §3 above |
| 2 | Build the Tier-A test harness (§6), implement gradient checkpointing, verify per-backend (start with one backend, expand once proven) | 1-2 days | **DONE for SNNTorch — DROPPED.** `Case_Study_Evaluation_Report.md` Case B: architectural incompatibility (`init_hidden=True` state breaks `torch.utils.checkpoint`'s recompute assumption), confirmed two independent ways via PyTorch's own error diagnostics. Norse left as an untested lead (explicit state-threading, the natural next candidate) — not yet built. |
| 3 (stretch, not committed) | Investigate OTTT/e-prop/FPTT adoption feasibility against this project's 4 backends | Multi-week research scope — separate from this semester's timeline | Not started, still out of scope |
| 4 (new) | Network dimensionality sweep — capacity vs. criticality/fragility trade-off (§9) | Multi-week, follow-on to Phase 1-2 instrumentation | Not started |

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

## 9. Network dimensionality — capacity/fragility trade-off (new objective)

> Network dimensionaility implies the capabilities of the network and its exclusive structure integirty on soliving complex problems. SNNs presnet the feature of being capable of extracting sptail and temporal features, from event-driven applictaions. Modelling a larger network to evalaute what makes SNNs to extrcat the highest amount of features and retain emergence behaviour under a growing arhcietcture. Dissecting the bottleneck of a small network and the structural fragility of a large network that is shaped to resolve SNN complexity on a varing setup. Elavating the capabiltes of each structure and its purpose relavent to event-driven applications and inclusive of hardware constriants, mapping the trade-off curve, pinpointing the exact transition point where a small network's information bottleneck yields to a large network's criticality and structural fragility.

**Operationalizing the claim** (the prose above is unfalsifiable as-is; needs measurable proxies before it's an experiment):

- **Information bottleneck (small network):** task accuracy / mutual information between spike trains and labels plateauing despite added training budget — the small net can't extract enough features regardless of how long it trains.
- **Emergence / feature retention (growing network):** does accuracy or spike-train MI keep increasing with width/depth, or saturate/decline past a point?
- **Criticality / structural fragility (large network):** branching ratio or avalanche statistics of spike counts as a criticality proxy; accuracy or spike-rate collapse under weight/input perturbation, quantization, or eval-time dropout as a fragility proxy.
- **Hardware constraint axis:** peak VRAM and step time per architecture size, reusing the memory-scaling instrumentation from §3 (`vram_batch_scaling_task.md`, the `diagnostics/verify_dataset_load.py`-style harness) — same axis as the resolution×T×batch sweep already done, just swept over width/depth/neuron-count instead.

**Experiment design (draft):**
1. Fix dataset (start with N-Caltech101, since VRAM instrumentation already exists for it) and one backend.
2. Sweep network size along one axis at a time (width, then depth) across a small/medium/large grid.
3. Log per config: task accuracy, spike-train MI (or a cheaper proxy if MI estimation is too costly), branching-ratio/avalanche stat, peak VRAM, step time, and fragility under a fixed perturbation.
4. Plot the trade-off curve; the "transition point" is read off where the fragility/criticality metric crosses the point where the bottleneck metric stops improving.

This plugs into the phased plan as **Phase 4** in §7, added without altering the status or scope of Phases 1-3.

## Presentation note

This document is planning material — the two-slide diagnostics/memory-scaling
deck already built (`Presentation_EventDataWorkflow.pptx`) stays as-is for
this week. Tier B (OTTT/e-prop/FPTT) belongs with the Brian2CUDA plan as
later-phase material, not this week's talk.
