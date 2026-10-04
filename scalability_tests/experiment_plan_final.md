# Scalability Study — Experiment Plan (Final)

**Status:** high-level strategy, locked. Shapes only — no per-run configurations yet;
those get written one experiment at a time, the way `experiments/ex6/README.md` did for
the first one, when that experiment is about to start.

**Study title:** *Boundaries of Emergence — Identifying Capacity Thresholds in
Small-Scale SNNs*

**Note on this document:** the original finalized plan was written on another machine
and never committed to this repo. This file reconstructs it from: the colleague's source
docs (`Scalability.pdf`, `scalability.md`), the 10 clarifying questions and answers
exchanged with the colleague, the first draft plan (`experiment_plan.md`, 2026-09-02 —
its still-useful explanations were merged into §11 on 2026-10-04 and the draft deleted,
so this file is the single source), the actually-implemented `ex6`
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
subtraction: `train_accuracy_pct` from the **last row of `epochs.csv`**, minus
`test_accuracy_pct` from **`runs.csv`**.

> **Corrected 2026-09-29.** This section previously said `epochs.csv` logs
> `test_accuracy_pct` per epoch. It does not — those columns are always empty, because
> testing runs once in `SNNTester` after training completes, by design. The consequence
> is one test number per run, so a test-accuracy peak occurring mid-training is
> invisible. Report the gap with that limitation stated.

## 3. Frameworks

Four backends exist in this pipeline: **SpikingJelly, SNNTorch, Norse, Sinabs**.

**Revised 2026-09-29.** The **width ladder (ex7) now runs on all four frameworks**, at
every rung, per direct instruction from the study designer. **ex9 remains
SpikingJelly-only** — there, isolating network *size* as the single variable still
holds, and changing frameworks at the same time would confound it.

**Revised 2026-10-04 (final).** The **depth ladder (ex8) also runs on all four
frameworks**, every rung — same as ex7.

The original design ran ex7-ex9 on SpikingJelly alone and deferred all cross-framework
work to **ex10**, a replay of just the sizes that mattered. Since ex7 and ex8 now both
cover all four frameworks, **ex10 is dropped** — nothing is left for it to replay.

Cost consequence: ex7 becomes 4 rungs × 4 frameworks = **16 runs**, roughly 60-100
GPU-hours. Run the SpikingJelly rungs first so the ladder's shape is known before the
other three frameworks are committed to.

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
| 1 | **Mutual Information** I(Z;Y) | how much the network's internal spike patterns still retain about the task labels | ✅ **built, I(X;Z) deliberately excluded** — `learning/capacity_metrics.py` implements `mutual_information_zy` only, opt-in behind `training.compute_capacity_metrics`, computed over the full test set (not one probe batch), reported raw. I(X;Z) was evaluated and rejected on 2026-09-29 — see the note below this table. |
| 2 | **Participation Ratio** (effective dimension, via PCA on spike rates) | are the extra neurons doing genuinely different work, or all copying each other | ✅ **built** — same module and flag. **Revised:** now computed over the full test set (not one probe batch), measured per-channel (not per-pixel), and reported both raw and normalized (PR/N). See design doc §6b-6d for rationale and the sample-size fix. |
| 3 | **Spike-train entropy** (population coding entropy) | too orderly = the network is ignoring the input; too chaotic = noise, not information | ✅ **built** — same module and flag (distinct from CV-ISI, which measures firing *regularity*, not population entropy). **Revised:** now computed over the full test set (not one probe batch), measured per-channel (not per-pixel), and reported both raw and normalized (H/log₂N). See design doc §6b-6d. |
| 4 | **Task Performance Decoherence** — accuracy vs. scale | does accuracy fall off a cliff or decline gently as the network shrinks / the task gets harder | ✅ **no new code needed** — a plot over accuracy data every run already produces |
| 5 | **Hardware Overhead Proxy** — memory, wall-clock, spike count as an energy proxy | what does this size cost, on this GPU | ✅ **fully built** — `measure_dense_macs`/SynOps energy and VRAM calibration in `learning/utilities.py` and `learning/inference.py`; this is the "few metrics already in place" the colleague meant |

> **Resolved 2026-09-29: I(X;Z) stays out, on purpose.** `Scalability.pdf` p.3 names
> I(X;Z) as the ceiling test ("a hard ceiling when I(X;Z) plateaus"), but checked against
> this experiment's real numbers rather than the general design-doc argument: one
> N-Caltech101 sample is ~1.38M raw values (180×240×2 channels ×16 timesteps) against a
> 1,742-sample test set, and the estimator (PCA to k≤3 dims, quantile-binned into up to
> 216 symbols per side) would need a 216×216 joint table filled from 1,742 points —
> badly undersampled, where a saturated estimator and a real bottleneck are
> indistinguishable. I(Z;Y) doesn't have this problem (Y is 101 exact class labels, not a
> lossy PCA summary), and is arguably the more direct signal for this research question
> anyway: it asks whether *task-relevant* information survives, which is what "the
> network can't represent the task" actually means. Evidence for the bottleneck claim:
> **PR + entropy + I(Z;Y) together**. Full reasoning: `experiments/ex7/README.md` §9,
> gap 1.

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
   ex7   WIDTH LADDER           where are the TWO EQUILIBRIUM POINTS?   ← the main event
          f8/f16/f32/f64 x 4 frameworks
          ↓
   ex8   DEPTH LADDER           where is the DEPTH CEILING?
          width fixed at f12, d0/d1/d2/d4 x 128 x 4 frameworks
          ↓
   ex9   COMBINE + STRESS       does the winner survive noise and confirm it's learning, not memorizing?
          ↓
   ex10  (DROPPED)              ex7 + ex8 already cover all 4 frameworks, see §3
```

| # | in plain English | dataset | framework(s) | seeds |
|---|---|---|---|---|
| **ex6** | Settle pooling (1 vs 2) and confirm the biggest planned network fits in VRAM | N-MNIST | SpikingJelly | 1 |
| **ex7** | Add filters step by step (8/16/32/64), keep everything else fixed, find where each regime settles | N-Caltech101 | **all four** | 1 per rung (multi-seed pass dropped 2026-10-04 — width fixed at f12, seeded in ex8/ex9) |
| **ex8** | Fix width at **f12**, add hidden FC layers step by step (0/1/2/4 × 128), find where *that* stops being worth it | N-Caltech101 | **all four** | 1 per rung; **3 seeds on the winning rung (SpikingJelly)** |
| **ex9** | Combine f12 + the ex8 depth winner. Confirm it works, then stress it with corrupted input, then check the train−test gap | N-Caltech101 | SpikingJelly | 3 seeds throughout (this is what backs the "stable = variance <2%" claim) |
| **ex10** | **Dropped (2026-10-04)** — ex7 and ex8 both run all four frameworks (§3), nothing left to replay. | – | – | – |

### What each is for, and its stopping rule

**Quick reference — don't mix these up:**

| experiment | varies | signal |
|---|---|---|
| ex7 (width) | filters | **two** equilibrium points: capacity metrics plateauing (bottleneck side) and cost rising without return (fragility side) |
| ex8 (depth) | hidden FC layers (count only, size fixed at 128) | accuracy vs. **time/params**, gradient norms watched only as a diagnostic — see "ex8 design" below |

- **ex7 — Width ladder.** **Revised 2026-09-29** — the target is no longer the single
  highest-accuracy rung but the **two equilibrium points** the brief asks for
  (`Scalability.pdf` p.2, Strategy item 4): the size at which the small network stops
  gaining representational capacity, and the size past which the large network's cost
  grows without return. These are reported as **two separate trade-off tables** and are
  not required to be the same number. Every rung runs — the old "<1% accuracy gain for
  >10% VRAM" rule becomes an analysis criterion applied afterward, not an abort rule.
  Epochs: set to 50 on 2026-09-29, **reverted to 15 on 2026-09-30** (fixed early
  checkpoint for every rung — comparing width, not peak accuracy; convergence-speed
  confound accepted, see `experiments/ex7/config.yaml`). Full design:
  `experiments/ex7/README.md` §0 and §7.
  **Outcome (2026-10-04):** f8 gave the highest accuracy. For ex8/ex9 the width is fixed
  at **f12** (`conv1_out = conv2_out = 12`) — between f8 and f16, and the 12-filter
  size the pipeline used originally.
- **ex8 — Depth ladder.** Full design in "ex8 design" below. Requires adding a configurable FC hidden layer to `frameworks/spiking_net.py` first — right now the network goes straight `Flatten → Linear(classes) → lif_out`, with no hidden FC layer to grow. (The results schema already has `fc_hidden_layers` / `fc_hidden_size` columns waiting for this.) Stop a rung early at **<1% accuracy gain for >10% time/parameter increase** — the Depth Ceiling. Gradient norms are recorded per layer throughout, but as a diagnostic, not a trigger (see §9 for why this differs from the original scalability.md design).
- **ex9 — Combine + stress.** Three arms: (1) baseline confirmation of f12 + the ex8 depth winner on N-Caltech101, (2) the same config on corrupted/perturbed input, as the robustness boundary, (3) read off the train−test gap as the generalization signal. This is also where the three named landmark configs (Stable / Unstable / Factory-Correct, §7) get their final multi-seed confirmation.
- **ex10 — Dropped (2026-10-04).** Was a cross-framework replay of the ex7-ex9 sizes; ex7 and ex8 now run all four frameworks directly.

### ex8 background — what "depth" means (plain-English, for the write-up)

Added 2026-10-04 as reference material for the report.

#### 1. What "depth" means, as a story

Picture a **mail-sorting office** that gets a messy pile of letters and must put each
one into one of 101 bins (the 101 classes of N-Caltech101).

- **Width** is how many workers stand at one desk. More workers can notice more things
  at once: one spots stamps, one spots handwriting, one spots colours. That is the ex7
  filters (8/16/32/64).
- **Depth** is how many desks the letters pass through, one after another. Desk 1
  notices simple things ("there's a curve here"). Desk 2 combines those ("curve plus
  straight line, maybe a wheel"). Desk 3 combines again ("two wheels plus a frame, so a
  bicycle"). Each extra desk lets the office reason one step more abstractly.

So:

- **Width** = how much the network sees *at one step*.
- **Depth** = how many steps of combining it gets to do.

#### 2. The network today, and what "deeper" means here

```
events → Conv → Pool → Conv → Pool → Flatten → Linear(101) → lif_out
         └──── "seeing" part ────┘            └─ decision ─┘
```

ex7 made the **conv part** wider. Depth in this study does **not** add more conv layers.
This plan (§9) decided to add **hidden FC layers** in the decision part instead:

```
... Flatten → [Linear(256) → LIF] → [Linear(256) → LIF] → Linear(101) → lif_out
              └──── hidden layer 1 ──┘ └── hidden layer 2 ─┘
```

Each `[Linear → LIF]` is one more desk. A Linear layer is "fully connected" (FC): every
input is connected to every output. There are two dials:

- **Number of hidden layers:** this is the depth.
- **Size of each hidden layer:** the number of neurons in it, for example 128 or 256.

#### 3. Why depth gets its own experiment

Depth has one failure mode that width doesn't: **the learning signal fades as it travels
backwards.**

- Training works by passing a correction backwards from the output to the first layer,
  like the head of the office sending "you sorted that wrong" back down the line of
  desks.
- Every desk the message passes through, it gets a bit fainter. With too many desks, the
  first desks hear almost nothing and stop learning. This is called a **vanishing
  gradient**.
- For SNNs it is worse. A spike is an on/off jump, which can't be differentiated
  properly, so training uses an approximation called the **surrogate gradient**. Each
  extra layer adds a little more of that approximation error.

That is why the plan records **per-layer gradient norms** (how loud the correction still
is at each layer). This is already built, and it is a diagnostic only, not a stop rule.

Depth is also expensive in a different way from width. Layers run **one after another**,
so every layer adds to the time of each pass, and FC layers add many parameters.

### ex8 design — final (decided 2026-10-04)

**Fixed:** width **f12** (`conv1_out = conv2_out = 12`), N-Caltech101, optimizer, LR,
BPTT surrogate gradient, 15 epochs (same as ex7). **Varied:** number of hidden FC layers
only — hidden size fixed at **128**. **Frameworks:** all four.

#### The f12 network, layer by layer (180×240 sensor, kernel 5, pool 2)

| layer | output shape | parameters |
|---|---|---|
| **input** (one timestep of events) | 2 × 180 × 240 = 86,400 values | – |
| Conv1 (2→12, k5) + LIF | 12 × 176 × 236 | 612 |
| Pool 2 | 12 × 88 × 118 | 0 |
| Conv2 (12→12, k5) + LIF | 12 × 84 × 114 | 3,612 |
| Pool 2 | 12 × 42 × 57 | 0 |
| **Flatten** | **28,728** values | 0 |
| **Linear (28,728 → 101) + lif_out** = output layer | **101** (one per class) | **2,901,629** |

- **Input to the hidden layers = 28,728.** That is the flattened conv output, not the
  raw sensor size.
- **Output = 101 neurons**, one per class, in `lif_out`. This never changes.
- The hidden layers go **between** those two.

#### The ladder (4 rungs × 4 frameworks = 16 runs)

| rung | hidden layers | total parameters |
|---|---|---|
| d0 | 0 (today's network, the baseline) | 2.91M |
| d1 | 1 × 128 | 3.69M |
| d2 | 2 × 128 | 3.71M |
| d4 | 4 × 128 | 3.74M |

f12 wasn't an ex7 rung, so d0 needs its own runs.

#### How the network code was extended for depth (built 2026-10-04)

- **Config:** a new `fc_hidden: {layers, size}` section in
  `configuration/network_architecture.yaml`.
- **Builder:** `frameworks/spiking_net.py` inserts N × `Linear → LIF` blocks between
  `Flatten` and the classifier:

  ```
  Flatten → [Linear(128) → LIF] × N → Linear(101) → lif_out
  ```

- **`layers: 0` is the default** and builds exactly the old network, so ex6 and ex7 are
  unaffected.
- **Naming:** hidden LIF layers are `lif3`, `lif4`, …; the output stays `lif_out`.
- **Neuron type:** a new `lif_hidden` entry per framework under `neuron_types`.
- Capacity metrics, gradient norms, SynOps and the `fc_hidden_*` results columns already
  handle any depth — no change needed there.
- Rung configs and run instructions: `experiments/ex8/` (README §9).

#### Size vs. parameters — not the same thing

- **Size** is how many values flow through a layer: 28,728 in, 128 out.
- **Parameters** are the weights *connecting* two layers: inputs × outputs + biases.
- So `28,728 × 128 + 128 = 3.68M` parameters for the first hidden layer, but only
  `128 × 128 + 128 = 16.5k` for each hidden layer after it.

#### Why the hidden size is 128 — the rules

The "half or a quarter of the input" rule doesn't work here. It is meant for small
inputs: half of 28,728 is 14,364 neurons, about 412M parameters for that layer alone.
That would never fit in memory and would memorise 7k training samples instantly.
Use these three rules instead:

1. **At least the number of classes (101).** If it's narrower, the hidden layer itself
   becomes a bottleneck and spoils the depth test.
2. **A power of 2** (128 or 256). This is convention and efficient on a GPU.
3. **As small as rule 1 allows.** About 7k samples, and the first hidden layer already
   adds millions of parameters, so a bigger layer just adds overfitting risk.

That gives **128**. Counting layers while keeping their size fixed changes only one
variable, which is a cleaner depth test than the original `scalability.md` ladder (that
one changed both count and size, mixing two effects).

#### Stopping mechanism — the Depth Ceiling

- **Rule:** less than 1% accuracy gain in exchange for more than 10% more time or
  parameters means the **Depth Ceiling** has been hit.
- **Applied afterwards, not as an abort.** Every rung runs, the same as ex7.
- **Gradient norms are a diagnostic, not a trigger** (see §9).
- **Caveat on the "params" half of the rule.** Almost all of the parameter cost comes
  from the **first** hidden layer (+27%). Every layer after that adds only about 0.5%.
  So the "more than 10% params" condition fires once, at d0→d1, and never again.
  From d1 onward, the real cost of depth is **training difficulty**, not hardware:
  fading gradients, slower convergence and overfitting. Time and VRAM will barely move,
  because the FC layers are cheap next to convolution on a 180×240 image.

#### What results to expect, in theory

These are working hypotheses to compare against, **not acceptance criteria**. A result
that differs from them is still a result.

- **Accuracy:** probably a small gain at d0→d1 (one extra combining step), then flat at
  d2, and likely *worse* at d4. Two reasons for the drop at d4:
  - surrogate-gradient error adds up layer by layer;
  - with a fixed 15 epochs and a fixed learning rate, deeper networks learn more slowly.
    That's the same "depth vs. speed of convergence" confound as with width, so state it
    in the report.
- **Gradient norms:** these should shrink in conv1 and conv2 as depth grows. This is the
  vanishing-gradient signature, and the clearest evidence depth has to offer.
- **Firing:** deeper hidden layers may go **quiet** (few spikes, nearly dead) or
  **saturate** (everything fires). Watch the per-layer firing rates.
- **Capacity metrics:** a hidden layer's PR is capped at 128. I(Z;Y) should rise from
  layer to layer towards the output, which is the network "compressing towards the
  answer".
- **Train − test gap:** expect it to jump at d1, from the extra 0.8M parameters.
- **Frameworks:** each framework may use a different default surrogate-gradient
  function, so they could tolerate depth differently. If so, that difference is itself a
  result worth reporting.

## 7. The three landmark configurations

Not chosen on accuracy alone — same three criteria ex6 already uses for its own
pooling decision (structural share of the network → cost → accuracy), reapplied at
study scale:

| name | roughly | how we'll know |
|---|---|---|
| **Unstable** | small end of the ladder | accuracy variance across seeds >2%, or gradient norms spike/collapse, or VRAM grows non-linearly |
| **Factory-Correct** | the equilibrium rung(s) from ex7+ex8 | <1% accuracy gain from going further, resources are efficiently used, generalizes (train−test gap stays small) |
| **Stable but not optimal** | between the two | consistent gradient norms, predictable VRAM, but not yet at the accuracy ceiling |

## 8. Deliverables

| deliverable | comes from |
|---|---|
| **Small-network equilibrium** — a filter count (bottleneck side) | ex7 |
| **Large-network equilibrium** — a filter count (fragility side) | ex7 |
| **Two trade-off tables** — pros/cons per regime, kept separate | ex7 |
| Depth Ceiling — a number of hidden layers (at f12, size 128), per framework | ex8 |
| Trade-off curve — quality vs. cost, every rung | ex7 + ex8 |
| Stable / Unstable / Factory-Correct — three named configs | ex7 + ex8 + ex9 |
| Confusion matrices across the ladder | ex7, ex8 |
| Correlation matrix across all tracked metrics | all runs |
| Per-framework comparison, all 4 rungs | **ex7** (width) and **ex8** (depth) — ex10 dropped, see §3 |

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
  **Revised 2026-10-04:** the width-ladder 3-seed point is dropped — width is fixed at
  f12 (not an ex7 rung), and f12 is seeded via the ex8 winning rung and ex9. ex7's rung
  ordering is therefore single-seed; state that as a caveat in the report.
- **Experiment numbering:** ex7 = width ladder, ex8 = depth ladder, ex9 = combine +
  stress, matching ex6's pattern of one experiment folder per phase.
- **Cross-framework comparison gets its own experiment, ex10**, rather than repeating
  the full ladder on all four frameworks (~4x the run count). It replays only the
  handful of configs ex7-ex9 identified as meaningful. **Superseded 2026-10-04:** ex7
  and ex8 both run all four frameworks; ex10 is dropped.

Final decisions made 2026-10-04 (ex8 depth design — details in §6, "ex8 design"):

- **ex8 runs on all four frameworks**, not SpikingJelly only.
- **Width fixed at f12** (`conv1_out = conv2_out = 12`) for ex8 and ex9. ex7's
  highest-accuracy rung was f8; f12 sits between f8 and f16 and is the pipeline's
  original 12-filter size.
- **Depth = number of hidden FC layers only**, size fixed at **128** (≥101 classes,
  power of 2, smallest that qualifies). Ladder: **d0 / d1 / d2 / d4**.
- **Stop rule applied afterwards**, not as an abort; every rung runs.
- **Theoretical expectations are hypotheses, not acceptance criteria.**

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
3. ✅ **Done (2026-10-04).** Configurable hidden FC layers: new `fc_hidden:` section
   (`layers`, `size`) in `configuration/network_architecture.yaml`, built in
   `frameworks/spiking_net.py` as N × `Linear → LIF` between `Flatten` and the
   classifier; neuron type per framework via `neuron_types.<fw>.lif_hidden`. `layers: 0`
   (the base default) is the original architecture. `experiments/ex8/` written
   (`config.yaml` + `d0/d1/d2/d4.yaml` + `README.md`). Tests in
   `tests/unit_layer_naming.py`.
4. ✅ **Done.** `experiments/ex7/README.md` and the rung configs are written
   (`f8/f16/f32/f64.yaml` over a shared `config.yaml`). Revised 2026-09-29 for the
   four-framework scope, the 8-64 rung range, and the 50-epoch fixed budget — see that
   file's §0 for the full change record.

5. **Remaining, before the ex7 write-up:** settle the **I(X;Z) disagreement** flagged in
   §5 — either implement it or justify its absence in the report.

6. **Remaining, nice-to-have:** SynOps (`synops_energy_pj`) is computed but lives only in
   each run's own `training_results.csv`, not in `runs.csv`/`layers.csv`. The brief's
   "spikes rise without accuracy" diagnostic needs it compared across runs, so either
   bump the schema to v5 with a SynOps column or aggregate it in the analysis script.
   See `experiments/ex7/README.md` §9, gap 2.

---

## 11. Background explanations for the report

Merged 2026-10-04 from the first draft plan (`experiment_plan.md`, 2026-09-02, now
deleted). Its *decisions* were superseded (N-MNIST primary, E0–E5 naming, pinned batch
size, DVS128); only the explanations that still hold are kept here.

### 11a. The pooling decision, explained (`pool_kernel: 2`, locked by ex6)

| value | what it does |
|---|---|
| `pool_kernel: 1` | a 1×1 window — **no downsampling**, the layer is an identity |
| `pool_kernel: 2` | a 2×2 window keeping the **largest** value — halves each dimension |

So it is **no downsampling vs. 4× downsampling**, not "one max vs. two."

Why 2 — pooling sets the flatten width, which sets the classifier's size. On N-MNIST
(12/32 filters):

| | `pool=1` | `pool=2` |
|---|---|---|
| flatten width | 21,632 | 800 |
| classifier params | 216,330 | 8,010 |
| **classifier's share of the network** | **95.5%** | **43.9%** |

At `pool=1` the study would mostly be measuring the classifier, not the feature extractor.

Supporting reasons:
- **VRAM:** no pooling keeps layer 2 ~5.6× larger, and BPTT stores every timestep.
- **Cited baseline:** `12C5-MP2-32C5-MP2-FC` (snnTorch/Tonic tutorial) keeps `MP2`.
- **Pooling on spikes means something:** it sits after the LIF, so it asks "did anything
  in this 2×2 patch fire?" — tolerance to noisy event positions.

### 11b. Neurons vs. parameters — width and depth buy different things

**Filters contain neurons:** each filter produces one feature map, and every cell of that
map is one LIF neuron.

At f12 on N-Caltech101 (180×240):

| layer | neurons |
|---|---|
| `lif1` — 12 maps × 176 × 236 | 498,432 |
| `lif2` — 12 maps × 84 × 114 | 114,912 |
| `lif_out` | 101 |
| **total (d0)** | **613,445** |

| change | neurons added | parameters added |
|---|---|---|
| **Width** (more filters) | many — every filter adds a full feature map | few (conv weights are small) |
| **Depth** — one hidden layer @128 (d0→d1) | **+128** (+0.02%) | **+0.79M** (+27%) |

**Width buys neurons; depth buys parameters.** This is why the two ladders use different
cost currencies (VRAM for width, time/parameters for depth), and why every size claim must
say which count it means.

### 11c. What each metric tells you

| group | metric | what it says |
|---|---|---|
| Quality | test accuracy | the headline result |
| | train accuracy | can it fit the data at all? (bottleneck signal) |
| | **train − test gap** | learning or memorising? (overfitting signal) |
| | per-class precision / recall / F1 | which classes it fails on |
| Capacity | **Participation Ratio** | how many genuinely different things a layer does, vs. how many neurons it has |
| | Mutual information I(Z;Y) | how much of the answer survives into a layer |
| | spike-train entropy | how varied the population code is — read with MI, never alone |
| | task performance decoherence | accuracy vs. scale — graceful decline or cliff |
| Activity | spike rate (per neuron per timestep) | sparsity; target <10%, ideally 1–5% |
| | CV-ISI | how regular vs. bursty firing is |
| | **SynOps energy** | neuromorphic-hardware cost proxy |
| Cost | peak VRAM | width's cost currency |
| | training time per epoch | depth's cost currency |
| | inference latency (bs=1, median/p90/p99) | real-time suitability |
| | GPU energy, utilisation, idle episodes | measured cost; is the GPU working or waiting for data? |
| Stability | **per-layer gradient norms** | is the learning signal reaching the front layers? |
| | front-layer / final-layer norm ratio | vanishing-gradient indicator |

Gradient norms are **logged, not acted on** — nearly free, zero effect on training, and
without them a flat depth result can't be told apart from "the deep layers never trained."

### 11d. Why the two matrices earn their place

- **Confusion matrix per rung:** the information bottleneck says a too-small network
  gives two different inputs the *same* internal description, and no later layer can
  undo that. The confusion matrix shows it directly — specific class pairs swapped,
  related classes clumping into blocks. Those blocks should break apart as the network
  grows. Output: one matrix per rung, plus smallest-vs-largest comparison.
- **Correlation matrix across all metrics, all runs:** ~15 metrics are tracked and some
  measure the same thing. If PR and spike rate correlate at 0.95, they are one piece of
  evidence, not two. It also tests the study's assumptions — "does effective dimension
  predict accuracy?" is a coefficient, not an opinion. An uncorrelated PR is itself a
  finding.

### 11e. Hardware caveat for any timing claim

Every timing, memory and energy number is a property of the machine (Tesla T4, recorded
per run in `runs.csv`: `gpu_name`, driver, CUDA, torch version).

With few CPU cores (Colab/Kaggle), a starved DataLoader makes **wall-clock time measure
the data pipeline, not the network**. Accuracy, neuron counts and **VRAM stay
trustworthy**; epoch time, throughput and GPU energy only if `gpu_util_avg_pct` is high
and `gpu_idle_episodes` near zero. This matters for ex8: the depth stop rule's "more
time" half needs that check before it can be believed.

### 11f. Single seed — what it costs, to state in the report

Outside the multi-seed points (§9), a gain smaller than typical seed noise cannot be told
apart from noise. That is acceptable where differences are large (the ends of a ladder)
and is why the multi-seed points sit where they are not (near a ceiling, where the
stop rule fires on a <1% difference). The "Stable = variance <2%" criterion (§7) is only
measurable at a multi-seed point.

### 11g. Two operating points, not one winner

The two comparison tables stay separate on purpose: the small network wins on latency
and energy, the large one on capacity. They answer different questions, so the study
reports **two stable operating points**, not one "optimal size."
