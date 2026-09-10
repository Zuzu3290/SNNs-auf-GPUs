# Scalability Study — Experiment Plan (Final)

**Status:** high-level strategy, locked. Shapes only — no per-run configurations yet;
those get written one experiment at a time, the way `experiments/ex6/README.md` did for
the first one, when that experiment is about to start.

**Study title:** *Boundaries of Emergence — Identifying Capacity Thresholds in
Small-Scale SNNs*

**Note on this document:** the original finalized plan was written on another machine
and never committed to this repo. This file reconstructs it from: the colleague's source
docs (`Scalability.pdf`, `scalability.md`), the 10 clarifying questions and answers
exchanged with the colleague, the first draft plan, the actually-implemented `ex6`
experiment (code + README), and a fresh round of decisions made while rebuilding this
document. Where the original content was genuinely unrecoverable, a new decision was
made explicitly rather than guessed — see [§9 Decisions made while rebuilding this
plan](#9-decisions-made-while-rebuilding-this-plan).

---

## 1. The question, in plain English

We keep growing an SNN in two directions and watch what happens:

- **Wider** — more filters (feature-detectors) per convolution layer.
- **Deeper** — more layers.

At every size we measure: how accurate is it, how much GPU memory does it use, how long
does it take to train, and is it still learning properly (or has it started to break —
unstable training, or just memorizing instead of learning). The goal is to find the
point where growing the network stops being worth it, and to name three landmark sizes:
one that's clearly **too small** (unstable / underpowered), one that's **just right**
("factory correct"), and to understand the cost of going bigger than necessary.

We do this once, in detail, on one framework — then take just the sizes that turned out
to matter and check whether the other three frameworks behave the same way at those
sizes. That second pass is the actual "how does scalability compare across frameworks"
answer.

### Glossary

| term | plain meaning |
|---|---|
| filter / conv layer | a feature-detector that slides over the image; more filters = the layer can notice more different patterns at once |
| pooling | a step right after a filter layer that shrinks the image, keeping only the strongest signal in each small patch |
| width | how many filters per layer (and later, how wide the hidden layers are) |
| depth | how many layers the network has |
| VRAM | GPU memory. Bigger networks need more of it; running out crashes the run |
| seed | the random starting point for a run. Same config + same seed = same result; running the same config with different seeds tells you whether a result is real or luck |
| ceiling | the size beyond which growing further stops paying off |
| gradient norm | a signal-strength reading of whether the learning signal is still reaching the earliest layers of a deep network, or has fizzled out |
| stable / unstable / factory-correct | our three named landmark sizes — see §7 |

---

## 2. Datasets

| dataset | role |
|---|---|
| **N-MNIST** | Pilot only. Fast, cheap, used once (ex6) to sanity-check the instrumentation and settle two global dials before any real GPU time is spent. |
| **N-Caltech101** | Primary. Every substantive experiment (ex7–ex10) runs here — 101 classes on roughly 7k samples is where a too-small network genuinely fails, which is the whole point of the study. |

A third dataset (DVS128 Gesture) appeared in the first draft as a held-back
generalization test. **It is out of scope for this study** — see §9. The
"does it generalize or just memorize" question is instead answered by the **train −
test accuracy gap** on N-Caltech101 itself — not a stored column, but a two-number
subtraction from the final epoch's `train_accuracy_pct` / `test_accuracy_pct`, which
`epochs.csv` already logs per epoch, per run.

## 3. Frameworks

Four backends exist in this pipeline: **SpikingJelly, SNNTorch, Norse, Sinabs**.

The width/depth ladder (ex7–ex9) runs on **SpikingJelly only**, on purpose — the point
of that part of the study is to isolate network *size* as the only variable, and
changing frameworks at the same time would confound that. Once the ladder has found the
sizes that matter, **ex10** replays just those sizes on the other three frameworks, to
see whether each framework's scalability curve bends at the same place or not.

## 4. What gets locked before anything else runs — ex6

**Status: complete.** ex6's three arms ran on 2026-09-02; full results and conclusions
are in `experiments/ex6/README.md` §12. The settings every later run must share are now
decided, not pending:

| setting | value | decided by |
|---|---|---|
| `pool_kernel` | **2** | ex6 — won on all three criteria (structural share, VRAM, and accuracy), not a close call |
| batch size / grad-accum | **64 / 2** (effective 128) for N-MNIST | ex6 arm C — fits N-MNIST's top-of-ladder network with ~26% headroom. **ex7 onward (N-Caltech101) will calibrate their own batch size per the `calibrate_batch_size()` occupancy policy (30-35% VRAM), which will differ substantially by dataset** — N-Caltech101's much larger images will likely result in single-digit batch sizes. See design doc §6a: this is correct, dataset-dependent behavior, not a confound to control for. |
| timesteps `T`, backend, LR schedule (`none`, constant) | fixed in `ex6/config.yaml` | held fixed for the whole study |

**Carried forward as a general caution from ex6's results:** GPU utilization was low
(22-37%) on the smaller arms, meaning wall-clock/epoch-time comparisons were measuring
the data loader, not the network, on this hardware. ex7's smaller rungs will likely hit
the same regime — check `gpu_util_avg_pct` before trusting any timing comparison there,
the same way ex6 did.

## 5. Metrics

Read directly from `Scalability.pdf` (the brief) rather than only its `.md` summary,
cross-checked against the codebase. The brief itself names **5 subjects** — this is
what the colleague's Q&A answer meant by "there is a thorough proposal of 5 subjects
and within those 5 subjects few metrics are already in place in the code base." The
status column below was verified against the code directly, not assumed from
documentation.

### The 5 subjects, in plain English

| # | subject | plain English | built already? |
|---|---|---|---|
| 1 | **Mutual Information** I(Z;Y) | how much the network's internal spike patterns still retain about the task labels | ✅ **built** — `learning/capacity_metrics.py`, opt-in behind `training.compute_capacity_metrics`. **Revised:** I(X;Z) is unreliable at these sample sizes and is dropped; only I(Z;Y) remains, computed over the full test set (not one probe batch), reported raw. See design doc §6e for rationale. |
| 2 | **Participation Ratio** (effective dimension, via PCA on spike rates) | are the extra neurons doing genuinely different work, or all copying each other | ✅ **built** — same module and flag. **Revised:** now computed over the full test set (not one probe batch), measured per-channel (not per-pixel), and reported both raw and normalized (PR/N). See design doc §6b-6d for rationale and the sample-size fix. |
| 3 | **Spike-train entropy** (population coding entropy) | too orderly = the network is ignoring the input; too chaotic = noise, not information | ✅ **built** — same module and flag (distinct from CV-ISI, which measures firing *regularity*, not population entropy). **Revised:** now computed over the full test set (not one probe batch), measured per-channel (not per-pixel), and reported both raw and normalized (H/log₂N). See design doc §6b-6d. |
| 4 | **Task Performance Decoherence** — accuracy vs. scale | does accuracy fall off a cliff or decline gently as the network shrinks / the task gets harder | ✅ **no new code needed** — a plot over accuracy data every run already produces |
| 5 | **Hardware Overhead Proxy** — memory, wall-clock, spike count as an energy proxy | what does this size cost, on this GPU | ✅ **fully built** — `measure_dense_macs`/SynOps energy and VRAM calibration in `learning/utilities.py` and `learning/inference.py`; this is the "few metrics already in place" the colleague meant |

### Named separately in the brief, not part of the 5 subjects

| item | status |
|---|---|
| **Train − test accuracy gap as a scaling axis** | the brief says outright "you're not currently tracking that" — still true today; derivable from `epochs.csv`'s `train_accuracy_pct`/`test_accuracy_pct`, not a stored column (see §2) |
| **Run-to-run variance across seeds** | not a code gap — covered by the seed policy in §9/§6 |

### Named in `scalability.md` / the Q&A, not in the brief

| item | status |
|---|---|
| **Per-layer gradient norms** (depth-stability diagnostic) | ✅ **built** — deferred-sync tracking in `learning/training.py`, same opt-in flag |

### Bottom line

**All four previously-missing metrics are now built** — Mutual Information,
Participation Ratio, spike-train entropy, and per-layer gradient norms — behind the
`training.compute_capacity_metrics` config flag, plus a correctness fix (the network's
output layer, `lif_out`, was never measured for anything before; layer naming/hooking
is now dynamic and depth-safe, always on). Full design and implementation record:
`docs/superpowers/specs/2026-09-04-capacity-metrics-design.md` and
`docs/superpowers/plans/2026-09-04-capacity-metrics.md`.

**Resolved in revision 2:** a final review found that with a single probe batch,
Participation Ratio was mathematically capped near batch_size − 1 and spike entropy
tracked `log2(neuron count)` closely. This was addressed by: accumulating over the
full test set (removing the cap entirely, §6b), measuring per-channel instead of
per-pixel (focusing on the feature-dimensionality axis the width sweep varies, §6c),
and reporting both raw and normalized versions of both metrics (PR/N and H/log₂N, §6d).
See design doc §6b-6d for the full rationale.

### References (for the write-up, per the colleague's request to document and cite this properly)

- Information bottleneck framework: Tishby, Pereira & Bialek, *The Information
  Bottleneck Method* (2000).
- SNN-specific application: Schulz et al., *Information Bottleneck Optimization for
  Spiking Neural Networks* (2020); Bialek, *Spikes: Exploring the Neural Code*.
- Participation Ratio vs. task complexity: Mazzucato et al. (2016), *The Dimensionality
  of Neural Activity in Prefrontal Cortex Underflows Task Complexity*; Rigotti et al.
  (2013), *Dimensionality and Dynamics of Cognitive Control in Prefrontal Cortex*.
- PR-generalization link: Gao et al. (2017), *Geometry of Neural Computations Unifies
  Performance and Generalization in Deep Networks*.

> Neurons and parameters do not grow together — width sweeps scale neuron count roughly
> linearly and parameter count roughly quadratically. `total_neurons` is the axis this
> study reports on; every size claim says which count it means.

Two study-level outputs, built across runs rather than inside one:
- **Confusion matrix per size rung** (ex7/ex8) — makes the information-bottleneck failure visible as specific classes being confused, and shows those blocks break apart as width/depth increases. The underlying confusion-matrix code already exists (`learning/inference.py`); using it per rung across a sweep is new wiring, not new measurement.
- **Correlation matrix across all metrics, all runs** — checks which of the ~15 tracked metrics are actually independent evidence rather than restating each other. No code exists for this yet; it's a new analysis script run once enough rungs have results, not a per-run instrumentation gap.

## 6. The experiments

```
   ex6   PILOT + POOLING        settle pool_kernel, batch size, sanity-check meters
          ↓
   ex7   WIDTH LADDER           where is the FILTER CEILING?          ← the main event
          ↓
   ex8   DEPTH LADDER           where is the DEPTH CEILING?
          ↓
   ex9   COMBINE + STRESS       does the winner survive noise and confirm it's learning, not memorizing?
          ↓
   ex10  CROSS-FRAMEWORK REPLAY do the other 3 frameworks scale the same way at the sizes that mattered?
```

| # | in plain English | dataset | framework(s) | seeds |
|---|---|---|---|---|
| **ex6** | Settle pooling (1 vs 2) and confirm the biggest planned network fits in VRAM | N-MNIST | SpikingJelly | 1 |
| **ex7** | Add filters step by step, keep everything else fixed, find where it stops being worth it | N-Caltech101 | SpikingJelly | 1 per rung; **3 seeds on the winning rung** |
| **ex8** | Lock in the best width, add layers step by step instead, find where *that* stops being worth it | N-Caltech101 | SpikingJelly | 1 per rung; **3 seeds on the winning rung** |
| **ex9** | Combine the best width + best depth. Confirm it works, then stress it with corrupted input, then check the train−test gap | N-Caltech101 | SpikingJelly | 3 seeds throughout (this is what backs the "stable = variance <2%" claim) |
| **ex10** | Re-run just the sizes that mattered (smallest, width winner, depth winner, combined winner) on the other 3 frameworks | N-Caltech101 | SNNTorch, Norse, Sinabs | 1 per config (or 3, if ex9's variance turns out to matter for the comparison — decide when ex10 is scoped) |

### What each is for, and its stopping rule

**Quick reference — don't mix these up:**

| experiment | varies | stopping signal |
|---|---|---|
| ex7 (width) | filters | accuracy vs. **VRAM** |
| ex8 (depth) | layers | accuracy vs. **time/params**, gradient norms watched only as a diagnostic |

- **ex7 — Width ladder.** Stop a rung early if it gives **<1% accuracy gain for >10% VRAM increase** — that's the Filter Ceiling. Includes one extended-epoch arm on the *smallest* rung (train it far past the normal budget) to confirm that rung's ceiling is structural rather than "it just needed more training" — this is the first objection anyone reviewing the result will raise, so it's cheap insurance folded into ex7 rather than its own experiment.
- **ex8 — Depth ladder.** Requires adding a configurable FC hidden layer to `frameworks/spiking_net.py` first — right now the network goes straight `Flatten → Linear(classes) → lif_out`, with no hidden FC layer to grow. (The results schema already has `fc_hidden_layers` / `fc_hidden_size` columns waiting for this.) Stop a rung early at **<1% accuracy gain for >10% time/parameter increase** — the Depth Ceiling. Gradient norms are recorded per layer throughout, but as a diagnostic, not a trigger (see §9 for why this differs from the original scalability.md design).
- **ex9 — Combine + stress.** Three arms: (1) baseline confirmation of best-width+best-depth on N-Caltech101, (2) the same config on corrupted/perturbed input, as the robustness boundary, (3) read off the train−test gap as the generalization signal. This is also where the three named landmark configs (Stable / Unstable / Factory-Correct, §7) get their final multi-seed confirmation.
- **ex10 — Cross-framework replay.** No new ceiling-finding — the sizes are already fixed by ex7-ex9. Purely: does SNNTorch/Norse/Sinabs's cost/accuracy curve at these same sizes match SpikingJelly's, or does one framework hit its own wall earlier?

## 7. The three landmark configurations

Not chosen on accuracy alone — same three criteria ex6 already uses for its own
pooling decision (structural share of the network → cost → accuracy), reapplied at
study scale:

| name | roughly | how we'll know |
|---|---|---|
| **Unstable** | small end of the ladder | accuracy variance across seeds >2%, or gradient norms spike/collapse, or VRAM grows non-linearly |
| **Factory-Correct** | the winning rung(s) from ex7+ex8 | <1% accuracy gain from going further, resources are efficiently used, generalizes (train−test gap stays small) |
| **Stable but not optimal** | between the two | consistent gradient norms, predictable VRAM, but not yet at the accuracy ceiling |

## 8. Deliverables

| deliverable | comes from |
|---|---|
| Filter Ceiling — a number | ex7 |
| Depth Ceiling — a number | ex8 |
| Trade-off curve — quality vs. cost, every rung | ex7 + ex8 |
| Stable / Unstable / Factory-Correct — three named configs | ex7 + ex8 + ex9 |
| Confusion matrices across the ladder | ex7, ex8 |
| Correlation matrix across all tracked metrics | all runs |
| Per-framework comparison at the sizes that mattered | ex10 |

## 9. Decisions made while rebuilding this plan

Carried over unchanged from the colleague's answers, restated here so they aren't
re-litigated later:

- **Depth is varied via FC hidden layers**, not additional conv stages (per the answer
  to Q3, and confirmed by the `fc_hidden_layers`/`fc_hidden_size` columns already
  reserved in the results schema).
- **Gradient-norm collapse is a diagnostic, not a hard stopping rule for depth** — the
  original `scalability.md` design made it the overriding stop condition, but the
  colleague's answer scaled that back ("a simple approach doesn't come to mind... no
  need to execute [it as a stopping mechanism]"), matching `ex6/README.md`'s own
  "diagnostic only — not a stopping rule" wording.
- **No global-average-pooling rework of the classifier layer.** The concern (classifier
  params dominating the conv stack) is instead handled by `pool_kernel` choice in ex6,
  which is exactly what ex6's criterion 1 measures.
- **tdBN / Batch Renormalization is not adopted.** Flagged as optional by the colleague
  ("if this seems a lot, no need to execute"); nothing in the schema or codebase
  reserves space for it, so it's treated as skipped rather than pending.
- **Learning rate held fixed across every size**, stated as a limitation rather than
  tuned per rung.
- **Mutual information is computed both ways** — I(X;Z) and I(Z;Y) — plotted together as
  the information plane, resolving the brief vs. the Plan disagreeing on which one.
  **Superseded by revision 2:** I(X;Z) was dropped as unreliable at these dimensions
  and sample sizes; only I(Z;Y) is computed. See design doc §6e.

New decisions made in this conversation, where the lost file's content could not be
recovered and a fresh call was needed:

- **DVS128 Gesture is dropped from the study.** Only N-MNIST (pilot) and N-Caltech101
  (primary) remain in scope. The generalization question it used to answer is instead
  covered by the train−test gap metric on N-Caltech101.
- **E5 (complexity axis) is dropped** — it required a second full dataset on the same
  axis as N-MNIST, which no longer applies once DVS128 is out of scope.
- **Seed policy:** single seed for every sweep rung (matches ex6's own single-seed
  design), with **3 seeds at the points where a real decision rests on the result**: the
  width-ladder winner, the depth-ladder winner, and throughout ex9. This resolves the
  contradiction between "no need for seed runs anymore" (Q6 answer) and ex6's own
  reference to "three points where multi-seed IS required."
- **Experiment numbering:** ex7 = width ladder, ex8 = depth ladder, ex9 = combine +
  stress, matching ex6's pattern of one experiment folder per phase.
- **Cross-framework comparison gets its own experiment, ex10**, rather than repeating
  the full ladder on all four frameworks (~4x the run count). It replays only the
  handful of configs ex7-ex9 identified as meaningful.

---

## 10. Before ex7 can start

1. ✅ **Done.** `experiments/ex6`'s three arms ran, results table filled in,
   `pool_kernel: 2` locked for the whole study; `batch_size: 64`/`grad_accum: 2`
   confirmed for the N-MNIST pilot specifically (ex7 onward calibrates its own, see §4).
   See `ex6/README.md` §12.
2. ✅ **Done.** The 4 missing metrics — Mutual Information (I(Z;Y) only, revised —
   see below), Participation Ratio, spike-train entropy, and per-layer gradient-norm
   logging — are implemented in `learning/capacity_metrics.py` and wired into the
   training pipeline behind the opt-in `training.compute_capacity_metrics` flag. Also
   fixed along the way: `lif_out` (the network's output layer) was never measured for
   anything before this — layer hooking and naming are now dynamic and depth-safe.
   **Revised per the study designer's feedback:** metrics now accumulate over the full
   test set (not one probe batch), measure per-channel (not per-pixel), report raw and
   normalized values, and I(X;Z) is dropped. See
   `docs/superpowers/specs/2026-09-04-capacity-metrics-design.md` (§6 for the
   revision) and
   `docs/superpowers/plans/2026-09-04-capacity-metrics.md` for the full design and
   implementation record.
3. **Remaining:** add a configurable FC hidden layer to `frameworks/spiking_net.py`
   (needed for ex8).
4. **Remaining:** write `experiments/ex7/README.md` and configs the way `ex6/README.md`
   did — concrete filter counts, epoch counts, and the extended-epoch arm — now that
   ex6's results are in hand to base the ladder's starting point on.
