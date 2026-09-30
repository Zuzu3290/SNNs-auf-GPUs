# Experiment 7 — Width ladder (Factor A)

**Study:** Boundaries of Emergence — Identifying Capacity Thresholds in Small-Scale SNNs
**Position:** the main event of the scalability study. Runs after ex6 (pilot, complete)
and before ex8 (depth ladder).
**Dataset:** N-Caltech101 · **Backends:** all four (`sj`, `snntorch`, `norse`, `sinabs`)
· **Rungs:** 4 (f8, f16, f32, f64) · **Epochs:** 50, fixed · **Seeds:** 1 per rung now,
3 on the equilibrium rung once known

> **Revised 2026-09-29.** Three things changed after the study designer re-read the
> brief (`scalability_tests/Scalability.pdf`). They are recorded in §0 so the earlier
> design is not mistaken for the current one.

---

## The ex7 flow, start to finish

*(Plain-language walkthrough, kept verbatim for the report write-up — 2026-09-30.)*

**1. What you're varying, what you're holding still.** Everything about the network and
training is frozen — dataset (N-Caltech101), timesteps, learning rate, optimizer,
pooling — except one thing: how many filters each conv layer has. You run that at four
sizes: 8, 16, 32, 64. That's "width scaling" — nothing else is allowed to move, or you
can't tell whether a result came from size or from some other change riding along with
it.

**2. Why bother varying it at all.** A too-small network can't represent the task — it
physically doesn't have enough neurons to encode what the data needs (an "information
bottleneck"). A too-large network gets unstable and wasteful instead — more filters,
more VRAM, more training time, more spikes, but accuracy stops moving. Somewhere in
between is a size that's "just right" for this task. You're trying to find both edges —
where "too small" stops being too small, and where "too large" starts being wasteful —
not the single size with the best accuracy number.

**3. What you measure at each size, and what each thing tells you:**

- **Accuracy** (train and test) — the obvious one, but read alongside the others, not
  alone.
- **Participation Ratio (PR)** — of the neurons in a layer, how many are actually doing
  different work versus just copying each other. Low PR with lots of filters means most
  of them are redundant — a real sign the network isn't using its size.
- **Spike entropy** — how varied the firing pattern is. Too low = the network is
  basically ignoring the input. Too high = it's just firing noise, not structure.
- **I(Z;Y)** — how much of what the spikes carry is actually useful for telling classes
  apart. This is the direct "does it retain what it needs" check.
- **Cost:** VRAM, wall-clock per epoch, total spikes (SynOps) — what each size costs you
  on real hardware.

**4. The procedure, per run.** Pick one rung (e.g. f16) and one framework (e.g.
SpikingJelly). Train a fixed epoch budget, the same for every run so nobody's just "had
more time to learn." Test once at the end. Every metric above gets written to a shared
CSV (`runs.csv`/`layers.csv`), one row per run, so all 16 runs (4 sizes × 4 frameworks)
end up on the same table.

**5. Why the epoch budget matters.** An undertrained network gives you undertrained
readings on everything else too — PR, entropy, I(Z;Y) describe "where training happens
to be," not "what the architecture can do." The budget needs to be enough for that gap
to close, whatever the number turns out to be. (See §5 below for how this got resolved
in practice.)

**6. Why four frameworks, not one.** The original plan isolated SpikingJelly first to
keep things clean, then the study designer asked to widen the net — same four sizes, but
on all of Norse/SNNTorch/Sinabs/SpikingJelly too, to see whether they all hit the same
wall at the same width or each has its own.

**7. Reading the result.** Plot each metric against filter count. You're looking for two
crossing points, not a peak:

- **Small end:** the size where PR/entropy/I(Z;Y) stop climbing — below that point the
  network is genuinely bottlenecked; above it, adding width stops buying representation.
- **Large end:** the size where cost (VRAM, time, spikes) keeps climbing but accuracy
  barely moves — past that point you're paying for nothing.

---

## 0. What changed on 2026-09-29, and why

| # | before | now | why |
|---|---|---|---|
| 1 | Find the **highest** accuracy the ladder reaches | Find each regime's **equilibrium point** — the sweet spot where capacity gained stops paying for cost incurred | Brief p.2, Strategy item 4: locate "its own equilibrium /stability point… rather than making a case that they have a shared midpoint" |
| 2 | Rungs **16, 32, 64, 128** | Rungs **8, 16, 32, 64** | The research question (brief p.1) is where a *small* network's bottleneck gives way to a large network's fragility. The ladder was top-heavy for that; it shifts down one step. `f128.yaml` is retired, `f8.yaml` added |
| 3 | **SpikingJelly only** | **All four frameworks**, same 4 rungs each | Direct instruction from the study designer. This absorbs ex10 into ex7 — see §13 |
| 4 | Epochs 15, plus a 100-epoch extended arm on f16 | **Epochs 50 for every run**, no separate extended arm | The 15-epoch budget was measured as too short (§5). 50 was the extended arm's own figure, so the two collapsed — see §6 |

**Unchanged, and deliberately so:** dataset (N-Caltech101), `pool_kernel: 2`, timesteps
`T=16`, optimizer `nadam`, `lr: 0.002` with `lr_scheduler: none`, `use_amp: false`,
per-rung batch-size calibration, `compute_capacity_metrics: true`,
`resource_policy.worker_count_fallback: 1`, seed policy, and BPTT with surrogate
gradients (brief p.2 names the last one as a fixed constraint).

## 1. What this experiment decides

Grow the network **wider** — more filters per conv layer — one step at a time, on a
fixed task and a fixed training budget, and map what each step buys and what it costs.

| # | question | answer form |
|---|---|---|
| 1 | Where does added width stop buying representational capacity? | one filter count — the small-network **equilibrium point** |
| 2 | Where does added width start costing more (VRAM, time, spikes) than it returns? | one filter count — the large-network **equilibrium point** |
| 3 | How do Participation Ratio, spike entropy and mutual information move as the network widens? | one reading per metric, per layer, per rung |
| 4 | Do all four frameworks bend at the same width, or does each have its own wall? | four curves on one axis |

**The two answers in questions 1 and 2 are not required to be the same number.** The
brief is explicit that the small-network and large-network regimes stabilise on their
own terms and are reported as two separate trade-off tables, not forced into one
midpoint. If they coincide, that is a finding; if they don't, that is also a finding.

## 2. Why this runs on N-Caltech101, not N-MNIST

Per `experiment_plan_final.md` §2: N-MNIST was pilot-only, used up in ex6. Every
substantive experiment from here on — ex7, ex8, ex9 — runs on N-Caltech101: 101 classes
on ~7k samples is small and hard enough that a too-small network genuinely fails, which
is the whole point of the study. **ex6's settings still apply**: `pool_kernel: 2`,
timesteps `T=16`, constant LR — all locked for the whole study regardless of dataset.
**ex6's batch size does not apply** — see `config.yaml`'s comments and §4 below.

## 3. A caveat, decided before this experiment was designed

Checking the real numbers before writing this doc surfaced something ex6 could not have
caught (it never tested N-Caltech101): **the classifier dominates the network's
parameter count at every filter size in this sweep**, not just at the small end the way
it did on N-MNIST before pooling fixed it.

| filters | flatten width | classifier params | conv params | classifier's share |
|---|---|---|---|---|
| 8 | 19,152 | 1,934,453 | 2,016 | 99.9% |
| 16 | 38,304 | 3,868,805 | 7,232 | 99.8% |
| 32 | 76,608 | 7,737,509 | 27,264 | 99.6% |
| 64 | 153,216 | 15,474,917 | 105,728 | 99.3% |

Pooling helped on N-MNIST's 34×34 sensor, but N-Caltech101's 180×240 sensor leaves a
flatten width of `filters × 2,394` even after two pooling stages — big enough that the
classifier stays essentially the whole network regardless of filter count. Widening
barely moves this ratio (99.9% → 99.3% across the entire sweep).

**Decision (per discussion, 2026-09-10): proceed as-is, documented rather than fixed.**
Two things keep this from invalidating the experiment:

- **Neuron count, not parameter count, is this study's size axis** (`experiment_plan_final.md`
  §5's own "neurons and parameters do not grow together" note already anticipated a
  mismatch like this one, though not one this extreme).
- **Participation Ratio, spike entropy, and mutual information are measured on the conv
  layers' spike activity (`lif1`, `lif2`), not on the classifier.** The classifier's
  size doesn't touch these readings.

What this means in practice: **read every "network got bigger" claim in this
experiment as "the feature extractor got bigger, and so did the classifier reading
it" — not a clean isolation of feature-extractor capacity alone.** Total parameter
count is not a reliable capacity signal here; total neuron count and the capacity
metrics are.

The first f16 run makes this concrete: a 24.6pp train−test gap at epoch 15 (59.6% train
vs. 35.0% test) on 6,967 training samples against 3,868,805 classifier parameters —
roughly 555 classifier parameters per training sample. Expect the train−test delta
(§9) to be driven substantially by the classifier, not by conv width.

## 4. Batch size: recalibrated per rung, not pinned

This is a deliberate reversal of ex6's own approach — see `config.yaml`'s comments and
`docs/superpowers/specs/2026-09-04-capacity-metrics-design.md` §6a for the full
reasoning. In short: `calibrate_batch_size()` is an occupancy policy (fill 30-35% of
VRAM), not an experimental variable to hold fixed. A bigger network needs more memory
per sample, so it earns a smaller batch — that's correct behavior, not a confound.
Every rung's actual calibrated batch size gets recorded in `runs.csv`; check it before
comparing wall-clock/throughput figures across rungs, the same way ex6 had to check
GPU utilization before trusting its own epoch-time numbers.

**New consequence of the four-framework change (§0 item 3):** calibration now varies by
*framework* as well as by rung, since each backend has its own memory profile. A
cross-framework cost comparison at a fixed rung is therefore comparing two things at
once. Record `batch_size` alongside every cost claim and say so explicitly.

**Hardware must be held fixed too.** The first f16 run used a **Tesla T4**. Because
calibration targets a percentage of *available* VRAM, running a rung on a different card
changes its batch size and therefore its peak-VRAM number. Every run in this ladder must
be on the same GPU model, and `gpu_name` in `runs.csv` must be checked before the rungs
are compared.

## 5. The rungs

| file | filters (both stages) | epochs | framework(s) |
|---|---|---|---|
| `f8.yaml` | 8 | 50 | all four |
| `f16.yaml` | 16 | 50 | all four |
| `f32.yaml` | 32 | 50 | all four |
| `f64.yaml` | 64 | 50 | all four |

All four `extend` a shared `config.yaml` (dataset, timesteps, LR schedule, epochs,
`compute_capacity_metrics: true`) and change only the filter count.

**Retired files**, kept on disk but out of the ladder:
- `f128.yaml` — top rung of the old ladder, dropped by §0 item 2. Run it only if f64
  shows no sign of the fragility regime and the ladder needs extending upward.
- `f16_extended.yaml` — see §6.

**The budget was 15, and measurement proved it too short.** The first f16 run
(`20260929_155737_sj_seed0`, 2026-09-29) ended with train accuracy still climbing —
28.6% at epoch 1 to 59.6% at epoch 15, gaining ~1.5pp/epoch across epochs 10-15, train
loss falling monotonically 3.777 → 1.854. Nothing had plateaued. That raises the budget
for **every** rung, not just f16: `config.yaml` now sets `epochs: 50`. **That run is
superseded** and must not be compared against 50-epoch rungs.

**Convergence still matters even though peak accuracy is not the goal.** §0 item 1 moved
the target from "highest accuracy" to "equilibrium point," but Participation Ratio,
entropy and mutual information measured on an under-trained network describe its
*training state*, not its *architecture*. Each rung does not need to be tuned for
maximum accuracy; it does need to be converged enough, at the same fixed budget, or the
capacity readings compare training progress instead of network size.

## 6. The extended-epoch check — retired

Original design: train the *smallest* rung ~3.3x past the standard budget, to prove its
ceiling is **structural** — the network genuinely cannot represent more, not "it just
needed more training." That is the first objection anyone reviewing a capacity claim
will raise.

What happened: the 15-epoch budget failed (§5), and the extended arm's own 3.3x figure —
50 epochs — was adopted as the *new standard* budget for every rung. `f16_extended.yaml`
and `f16.yaml` now describe the same run, so the arm no longer exists as a separate
check.

**The structural question is still owed an answer**, and the 50-epoch f16 run is what
settles it:
- If train accuracy has clearly plateaued by epoch 50, the structural claim is proven by
  the standard run and no separate extended arm is needed. Delete `f16_extended.yaml`
  and record the plateau epoch here.
- If it is still climbing at 50, the budget is *still* too short: raise it for every
  rung and re-create the extended arm at ~3.3x whatever the new budget becomes.

Either way, f8 — now the smallest rung — inherits this question. Whichever rung ends up
smallest is the one that has to survive the "you just didn't train it long enough"
objection.

## 7. Reading the result — the two equilibrium points

The old rule ("stop early if a rung gives <1% accuracy gain for >10% VRAM increase") was
an *abort* rule for a ladder being climbed until it stopped paying. That no longer fits:
with only four rungs and the sweet spot as the target, **every rung runs regardless**,
and the rule becomes an analysis criterion applied afterward.

**The small-network equilibrium — the bottleneck side.** The lowest filter count at
which the capacity metrics stop rising with width. Signals, per brief p.3:
- **Participation Ratio plateaus.** PR is the effective number of independent dimensions
  the layer uses. If PR saturates at 5 while width keeps growing, the extra filters are
  redundant and the structural ceiling is reached.
- **Spike entropy plateaus or degrades into undifferentiated firing.** Flat entropy means
  the network is ignoring new input; chaotic high entropy means noise, not information.
- **Mutual information plateaus.**
- **Accuracy falls off a cliff rather than declining gently** as the network shrinks.

**The large-network equilibrium — the fragility side.** The filter count past which cost
grows without a matching return. Signals, per brief p.4:
- **Spike count / SynOps rises steeply with no accuracy gain** — the explicit diagnostic
  in the brief ("exponential increase in total spikes without an increase in accuracy").
- **Peak VRAM and wall-clock per epoch rise** while accuracy moves <1%.
- **Train−test delta widens** — capacity going into memorisation, not generalisation.
- **Seed variance exceeds the accuracy gain** — the gain isn't real if it's within noise.

These are reported as **two separate comparison tables** (§12), with pros and cons per
regime, exactly as brief p.2 Strategy item 4 specifies. They are not averaged into one
number.

## 8. Seeds

Per `experiment_plan_final.md` §9: single seed (0) for every rung in this first pass —
that's what `config.yaml` sets. Once results are in and the equilibrium rungs are
identified, re-run **those rungs** at seeds 1 and 2 to back the "accuracy variance <2%"
part of the Stable/Factory-Correct naming in `experiment_plan_final.md` §7. Do not add
seeds to every rung — that's the seed-budget blowup the plan's §9 deliberately avoids.

With four frameworks now in scope (§0 item 3), apply the multi-seed pass to the
equilibrium rung **on SpikingJelly only** unless a cross-framework difference turns out
to sit inside seed noise, in which case the seeds are what settle it.

## 9. What gets recorded — every metric in the brief, and where it lands

Verified against the code on 2026-09-29. Three shared CSVs at
`experiments/ex7/results/`: `runs.csv` (one row per run), `epochs.csv` (one row per
epoch), `layers.csv` (one row per layer per run), all at `schema_version: 4`. Each run
also writes its own folder with finer-grained files.

| brief's metric | status | column → file |
|---|---|---|
| **Mutual Information I(X;Z)** | ❌ **not implemented** — see gap 1 | — |
| Mutual Information I(Z;Y) | ✅ | `mutual_info_zy` → `layers.csv` |
| Participation Ratio (PCA/PR) | ✅ | `participation_ratio`, `participation_ratio_normalized` → `layers.csv` |
| Spike Train Entropy | ✅ | `spike_entropy`, `spike_entropy_normalized` → `layers.csv` |
| Task Performance Decoherence | ✅ | `test_accuracy_pct` → `runs.csv`; `train_accuracy_pct` per epoch → `epochs.csv` |
| Hardware: peak memory | ✅ | `peak_memory_train_mb`, `peak_reserved_train_mb`, `peak_memory_infer_mb` → `runs.csv` |
| Hardware: wall-clock per epoch | ✅ | `train_time_per_epoch_s` → `runs.csv`; `epoch_train_time_s` → `epochs.csv` |
| Hardware: total spikes / spike density | ✅ | `total_spikes`, `opportunities`, `spike_rate_pct` → `layers.csv` |
| **SOPs / SynOps** | ⚠️ computed, not cross-run comparable — see gap 2 | `synops_energy_pj` → per-run `training_results.csv`, `batch_metrics.csv`, `test.csv` |
| Real latency / VRAM (brief p.4 prefers these over the SOP proxy) | ✅ | `inference_latency_bs1_ms`, `_mean_ms`, `_p90_ms`, `inference_throughput_samples_per_s`, `batch_size` → `runs.csv` |
| **Train − test accuracy delta** | ⚠️ derivable, not stored | `train_accuracy_pct` (last row of `epochs.csv`) − `test_accuracy_pct` (`runs.csv`) |
| Run-to-run variance across seeds | ✅ no code needed | `seed` → `runs.csv`, compared across runs |
| Per-layer gradient norms (from `scalability.md`, not the brief) | ✅ | `grad_norm_mean` → `layers.csv` |

### Gap 1 — I(X;Z) is not computed — RESOLVED 2026-09-29, stays dropped

Brief p.3 makes I(X;Z) the primary bottleneck test: *"A small network hits a hard ceiling
when I(X;Z) plateaus."* `learning/capacity_metrics.py` implements `mutual_information_zy`
only. Re-checked against this experiment's actual numbers (not just the general §6e
argument) on 2026-09-29:

- **The estimator needs the joint distribution of two discretized variables, from the
  test set (1,742 samples).** Both X and Z get PCA-reduced to k≤3 components and
  quantile-binned into 6 bins each (`quantile_discretize()`), giving up to 6³ = 216
  symbols per side — so I(X;Z) needs a 216×216 joint table (up to 46,656 cells) filled
  from 1,742 samples. I(Z;Y) needs only 216×101, and Y's 101 symbols are exact class
  labels, not a lossy PCA summary of anything.
- **Raw X has no natural low-dimensional form to begin with.** One N-Caltech101 sample is
  180×240×2 channels × 16 timesteps ≈ 1.38 million raw values. Collapsing that to 3 PCA
  components before it can even enter the estimator is a bigger compression than the
  network itself performs — at that point the estimate mostly reflects the PCA
  projection's own information loss, not the network's.
- **A saturated estimator and a real bottleneck look identical from the outside**, and at
  this sample-size-to-bin-count ratio the estimator saturates on its own regardless of
  what the network is doing — exactly the §6e concern, now confirmed against real
  dimensions rather than argued in the abstract.

**Decision: I(X;Z) stays dropped.** Beyond the reliability problem, I(Z;Y) is arguably the
more direct bottleneck signal for *this* research question anyway — it asks whether the
task-relevant information survives, which is exactly what "the network can't represent
the task" means, rather than how much of the raw input (relevant or not) got compressed.
Evidence for the bottleneck claim in the write-up: **PR + entropy + I(Z;Y) together** —
PR for how many independent dimensions the layer uses, entropy for whether firing is
structured or degenerate, I(Z;Y) for whether what survives is actually useful for the
task. State this reasoning in the report rather than silently omitting I(X;Z).

### Gap 2 — SynOps is not in the cross-run schema

`synops_layer_map()` / `measure_dense_macs()` compute it (brief p.4 correctly notes this
is already built), and it is stored per epoch, per batch and per test batch inside each
run's own folder. But no SynOps column exists in `runs.csv` or `layers.csv`, so the
"spikes rise without accuracy" diagnostic in §7 means opening 16 separate
`training_results.csv` files by hand. Either add a column (schema bump to v5) or write
the aggregation into the analysis script.

### Note on per-epoch test accuracy

`epochs.csv`'s `test_accuracy_pct` and `test_loss` are always empty: testing runs once,
in `SNNTester`, after training completes — by design. Consequence for this experiment:
**one test number per run**, so a test-accuracy peak occurring mid-training is invisible.
Given §3's overfitting pressure, state this as a limitation when reporting the train−test
delta. (`experiment_plan_final.md` §2 previously claimed test accuracy was logged per
epoch; that claim has been corrected.)

### Sanity-check the meters on the first rung before trusting the ladder

The same way ex6 checked its own instrumentation:
- `cv_isi_mean` and `synops_energy_pj` should be **nonzero**.
- `total_neurons` should scale linearly with filter count (it is `filters × 51,112 + 101`
  on this sensor — f16 measured 817,893, so f8 should read 408,997).
- `gpu_util_avg_pct` should be checked before trusting any timing comparison across
  rungs (§4). ex6 saw 22-37% on its smaller arms, meaning it was timing the data loader
  rather than the network.

## 10. Deliverables

| # | deliverable | destination |
|---|---|---|
| 1 | **Small-network equilibrium point** — a filter count | `experiment_plan_final.md` §8 |
| 2 | **Large-network equilibrium point** — a filter count | `experiment_plan_final.md` §8 |
| 3 | Two trade-off tables — pros/cons per regime (brief p.2, item 4) | §12 |
| 4 | Trade-off curve — capacity metrics and cost vs. filter count | all 4 rungs |
| 5 | Per-framework comparison at all 4 rungs | absorbs the old ex10, see §13 |
| 6 | Confusion matrix per rung | the confusion-matrix data each run already produces |
| 7 | First real capacity-metric readings for the study | `layers.csv`, all rungs |

## 11. How to run it

CPU-only sanity checks first — seconds each, and they exist so the expensive step is
never started against a config that was already wrong:

```bash
python check_env.py
python check_network.py --config experiments/ex7/f8.yaml --all
python equivalence_check.py --config experiments/ex7/f8.yaml --experiment ex7
```

Then 4 rungs × 4 frameworks = **16 runs**, each its own process:

```bash
for FW in sj snntorch norse sinabs; do
  for RUNG in f8 f16 f32 f64; do
    python learning/main.py --config experiments/ex7/$RUNG.yaml --experiment ex7 \
        --framework $FW --seed 0 --inference stats \
        --results-root <results path> --cache-root <cache path>
  done
done
```

In practice run them one per session rather than as a loop, so one crash doesn't take
the rest with it. On Kaggle specifically:

- Use **Save Version → Save & Run All (Commit)**, not an interactive session. Interactive
  kernels are tied to the browser connection; a commit runs server-side.
- `runs.csv`/`epochs.csv`/`layers.csv` are written **only after training *and* inference
  finish** (`skeleton/results.py:263`). A session that hits the 12h cap loses the entire
  run — no partial results. Project the total from epoch 1's iteration timings before
  committing to a long rung.
- Point `--cache-root` away from `/kaggle/working`, or delete the cache before the
  commit ends: the frame cache is ~18GB and `/kaggle/working` becomes the saved output.
- Pin every session to the same GPU model (§4).

Pull results together and plot (no GPU needed for either):

```bash
python collect_results.py --from <folder> --experiment ex7    # merges, never overwrites
python make_plots.py --experiment ex7
```

**Check after the first rung, before running the other 15:** `cv_isi_mean` nonzero, the
calibrated batch size sane, `gpu_name` as expected, and the train curve's shape (§5).
Catching a problem here costs one run; catching it at the end costs sixteen.

## 12. Results

*(to be filled in once the rungs have run — the 15-epoch f16 run is superseded, see §5)*

### Per-rung readings, SpikingJelly

| rung | filters | batch | neurons | test acc | train acc (ep50) | train−test | peak VRAM | PR (lif1/lif2) | entropy (lif1/lif2) | I(Z;Y) (lif2) | SynOps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | | 408,997 | | | | | | | | |
| 2 | 16 | | 817,893 | | | | | | | | |
| 3 | 32 | | 1,635,685 | | | | | | | | |
| 4 | 64 | | 3,271,269 | | | | | | | | |

*(repeat this table per framework: `snntorch`, `norse`, `sinabs`)*

### Table A — the small-network regime (bottleneck)

| | |
|---|---|
| equilibrium filter count | _pending_ |
| pros | _e.g. low latency, energy efficiency, analysable_ |
| cons | _e.g. representation bottleneck, low memory retention_ |
| evidence | _PR / entropy / I(Z;Y) plateau, accuracy cliff_ |

### Table B — the large-network regime (fragility)

| | |
|---|---|
| equilibrium filter count | _pending_ |
| pros | _e.g. higher effective dimensionality, multi-timescale memory_ |
| cons | _e.g. training cost, spike inflation, over-synchronisation_ |
| evidence | _SynOps/VRAM/time growth vs. flat accuracy, widening train−test delta_ |

**Structural or under-trained (§6):** _pending_

## 13. Relationship to ex10

`experiment_plan_final.md` §6 scoped **ex10** as a cross-framework replay: re-run only
the sizes ex7-ex9 identified as meaningful, on SNNTorch, Norse and Sinabs. §0 item 3
moves that work into ex7, which now runs all four frameworks at all four rungs from the
start.

**ex10 is therefore redundant as originally scoped and should be either dropped or
re-scoped** once ex8 (depth) is designed — most plausibly as the cross-framework replay
of the *depth* winner only, since ex8 remains SpikingJelly-only for now. That decision
is deferred, not made here.

Cost note: 16 runs at 3-10h each is roughly 60-100 GPU-hours. Kaggle allows ~30 GPU-hours
per account per week, so this is a multi-account, multi-week schedule — plan it as one,
and run the SpikingJelly rungs first so the ladder's shape is known before the other
three frameworks are committed to.
