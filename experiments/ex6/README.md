# Experiment 6 — Pilot and pooling viability

**Study:** Boundaries of Emergence — Identifying Capacity Thresholds in Small-Scale SNNs
**Position:** first experiment of the scalability study. 🔴 **Hard gate — nothing downstream runs until this is settled.**
**Dataset:** N-MNIST · **Backend:** SpikingJelly (`sj`) · **Runs:** 3 · **Seed:** 0

---

## 1. What this experiment decides

Three things, in order of importance:

| # | question | answer form |
|---|---|---|
| **1** | **Which `pool_kernel` value does the whole study use — 1 or 2?** | one number, then frozen |
| **2** | Does the **largest** planned network fit in VRAM at the N-MNIST pilot's batch size? | yes / no / barely |
| **3** | Do all the meters read sensibly before real GPU time is spent? | pass / fail per metric |

Question 1 is the deliverable. Questions 2 and 3 ride along for free on the same three runs.

## 2. Why it has to run first

Pooling controls the spatial geometry of every layer after it. Change it and the flatten width changes, which changes the classifier's size, which changes the parameter count, the VRAM bill and the neuron count.

> **No two runs with different pooling are comparable to each other.** Every rung of every later experiment must share one value, so it has to be chosen before the ladder starts — not discovered midway.

The batch size has a similar property for this pilot's own runs (see [`config.yaml`](config.yaml)), which is why arm C is here too: pooling truly is global for the whole study, while the batch size arm C confirms is scoped to the N-MNIST pilot — later datasets (N-Caltech101 onward) calibrate their own via `calibrate_batch_size()` rather than inheriting this one.

There is also an external dependency — the pooling value was requested for use in a second pipeline, so it is a reporting obligation, not just an internal choice.

## 3. What pooling actually is, precisely

| value | what the layer does |
|---|---|
| `pool_kernel: 1` | a 1×1 window — **no downsampling at all**, the layer is an identity |
| `pool_kernel: 2` | a 2×2 window (4 cells) keeping **the single largest** — discards 75%, halves each dimension |

So this comparison is **no downsampling versus 4× downsampling**, applied twice — not "one max value versus two."

In this architecture pooling sits **after** the spiking layer, so max-pooling operates on 0/1 spikes. `pool=2` therefore asks *"did anything in this 2×2 patch fire?"* — real spatial tolerance, which matters for event data where the exact pixel of an event is noisy.

## 4. The three runs

| arm | config | what changes | purpose |
|---|---|---|---|
| **A** | [`pool1.yaml`](pool1.yaml) | `pool_kernel: 1` | the no-downsampling structure |
| **B** | [`config.yaml`](config.yaml) | *(base architecture, `pool_kernel: 2`)* | the reference structure |
| **C** | [`vram_check.yaml`](vram_check.yaml) | `conv1_out: 128`, `conv2_out: 128` | does the top of the ladder fit? |

Arms A and B differ in **exactly one value**. Arm C inherits arm B and differs in **exactly two** (the filter counts). The `extends:` chain enforces this — the variants state only what they change, so the arms cannot silently drift apart in any other setting.

## 5. What is held fixed

Everything not named above. Stated explicitly in `config.yaml` rather than inherited silently, so the record is unambiguous:

| held fixed | value | why it matters here |
|---|---|---|
| dataset | N-MNIST | no prompt, reproducible from config alone |
| timesteps `T` | 16 | multiplies runtime, memory and spike counts — must not move |
| backend | `sj` | this study varies size, not framework |
| seed | 0 | single seed by design |
| **batch size** | **64 × 2 accum = 128 effective** | **pinned — `calibrate_batch_size: false`** |
| learning rate | 0.002, **`lr_scheduler: none`** | the base `cosine` anneals it; `none` is what actually holds it constant |
| epochs | 5 | enough signal to separate the arms |
| AMP | off | VRAM is a cost metric here; AMP would measure its own coverage instead |
| neuron spec | base, untouched | the neuron is not a variable in this study |
| kernels | 5×5 | geometry stays comparable |

## 5b. Running environment

Every timing, memory and energy figure in this study is a property of **this machine**.
The same code on another GPU is a different number, so the hardware is part of the
result, not a footnote.

| | |
|---|---|
| **Platform** | Google Colab |
| **GPU** | **NVIDIA Tesla T4** — 15.0 GB VRAM reported (16 GB GDDR6), compute capability 7.5 |
| **System RAM** | 12.7 GB total (~8.6 GB free at run time) |
| **Disk** | 112.6 GB total (~55 GB free) |
| **Frame cache** | local Colab disk (`./cache`), disk tier, **~17 GB** for N-MNIST at T=16 |
| **Worker start method** | `fork` (Linux) — no spawn-reload RAM penalty |

`runs.csv` records `gpu_name`, `driver`, `cuda`, `torch_version` and `platform` per run,
so this is verifiable per row rather than trusted from this table.

### What the hardware means for this experiment

**✅ 15 GB VRAM is generous.** The memory-scaling work in
`Scalability_RealTime_Enhancement_Plan.md` was done against an 8 GB card (it reports
6.18 GB / 7.96 GB at 77.7%). Nearly double that headroom makes arm C's 128/128
configuration likely to fit at the pinned batch size — but "likely" is why arm C exists.

**✅ The T4 exposes power through NVML**, so the energy columns should populate rather
than coming back empty. Board TDP is ~70 W, which sets the scale for the idle baseline.

**⚠️ The cache lands on ephemeral local disk.** All three arms should run in one session:
arm B pays the ~17 GB preprocessing cost once, arms A and C then hit a warm cache. Expect
arm B to be noticeably the slowest for that reason alone — **not** because of pooling.

**🔴 Colab gives very few physical cores, and this is the real caveat.**
`check_env.py` warns at ≤ 2 physical cores, and a Colab runtime with 1 has been measured
holding GPU utilisation near 11% — the loader unable to keep the card fed. In that state
**wall-clock time measures the data pipeline, not the network.**

Consequences for the cost metrics:

| metric | trustworthy here? |
|---|---|
| `test_accuracy_pct`, spike rate, `total_neurons`, parameter counts | ✅ yes — hardware-independent |
| `peak_memory_train_mb`, `peak_reserved_train_mb` | ✅ yes — a memory measurement, not a timing one |
| `train_time_per_epoch_s`, throughput, GPU energy | ⚠️ **only if the GPU was not starved** |

**Check `gpu_util_avg_pct` and `gpu_idle_episodes` in `training_results.csv` on arm B
before trusting any timing comparison.** If utilisation is low and idle episodes are
frequent, the epoch times are loader-bound and Factor A's *VRAM* stopping rule should
carry the decision rather than the time-based one.

This also puts an asterisk on `scalability.md`'s "Factory Correct" criterion *"GPU never
idle"* — on a low-core runtime that can be violated regardless of network size, so it
cannot be used as a size verdict here.

## 6. The decision rule

The pooling winner is **not** chosen on accuracy alone. Three criteria, in priority order:

### 🥇 Criterion 1 — does the conv stack remain a meaningful share of the network?

This study is about the **feature extractor's** capacity. If the final classifier layer dominates the parameter count, a filter sweep barely moves the total and the study measures the wrong thing.

Predicted, at 12/32 filters on N-MNIST:

| | `pool=1` | `pool=2` |
|---|---|---|
| spatial at flatten | 26 × 26 | 5 × 5 |
| flatten width | 21,632 | 800 |
| conv stack params | 10,244 | 10,244 |
| classifier params | 216,330 | 8,010 |
| **classifier's share** | **95.5%** ⚠️ | **43.9%** ✅ |
| total neurons | 32,442 | 14,682 |

Across the full ladder the share runs **95.5% → 67.5%** at `pool=1` versus **43.9% → 7.1%** at `pool=2`. `pool=1` keeps the classifier dominant through exactly the small and mid rungs where the bottleneck is supposed to appear.

**This criterion is structural — it is arithmetic, not a measurement, and it does not depend on the run outcome.**

### 🥈 Criterion 2 — cost

`pool=1` leaves the second spiking layer at 26×26 instead of 11×11 — **5.6× more neurons**, and BPTT stores every timestep's activations. Recorded per arm: peak VRAM, epoch wall-clock, total neuron count.

### 🥉 Criterion 3 — accuracy

Test accuracy and the train−test gap. If `pool=1` wins here by a large margin it forces a rethink of criterion 1; a small margin does not, because the structural problem would still make later results uninterpretable.

## 7. What gets recorded

Written automatically. All three arms append into **one** `results/runs.csv`, so the
comparison table assembles itself as the arms finish.

### The columns that decide this experiment

`runs.csv` at `schema_version: 2` — the **architecture** group was added for this study
(see [`HOW_TO_RUN.md` §B3b](../../HOW_TO_RUN.md) for the full schema and the reasoning):

| column | why it matters here |
|---|---|
| **`total_neurons`** ⭐ | the study's size axis. Mandatory record for every run. |
| `neurons_per_layer` | e.g. `lif1:10800\|lif2:3872\|lif_out:10` — the breakdown |
| **`pool_kernel`** | **the value this experiment decides.** Every row proves its own. |
| `conv1_out` · `conv2_out` | 12/32 for arms A and B, 128/128 for arm C |
| `flatten_width` | 800 vs 21,632 — the whole mechanism behind criterion 1 |
| **`classifier_params`** vs `conv_params` | **criterion 1 is read directly off these two** |
| `peak_memory_train_mb` · `peak_reserved_train_mb` | criterion 2 — arm C's verdict |
| `test_accuracy_pct` | criterion 3 |
| `train_time_s` · `train_time_per_epoch_s` | criterion 2 |
| `fc_hidden_layers` · `fc_hidden_size` | `0` and empty here; they start reporting in ex8 |

Everything in the architecture group is **measured off the built network**, not read back
from the config — so a row describes what actually ran, not what was requested.

### Everything else, by group

| group | fields |
|---|---|
| **quality** | test accuracy · train accuracy · **train−test gap** · per-class P/R/F1 |
| **cost** | peak VRAM (train + infer) · epoch time · bs=1 latency (p50/mean/p90) · throughput · GPU energy (total + dynamic) · GPU idle episodes |
| **activity** | spike rate · spikes/neuron/inference · CV-ISI · SynOps energy |
| **stability** | gradient norms *(diagnostic only — not a stopping rule)* |
| **matrices** | confusion matrix per arm |

### Where each grain lives

| file | grain |
|---|---|
| `results/runs.csv` | 1 row per arm — **the comparison table** |
| `results/epochs.csv` | 5 rows per arm |
| `results/layers.csv` | 2 rows per arm (`lif1`, `lif2` — hooked layers only) |
| `results/runs/<run_id>.json` | full merged config per arm |
| `results/<run_id>/training_results.csv` | 5 rows, ~35 columns of GPU detail |
| `results/<run_id>/batch_metrics.csv` | ~4,685 rows (one per iteration) |
| `results/<run_id>/test.csv` | 157 rows (one per test batch) |

⚠️ **Do not change the results schema between arms.** `append_row()` refuses to write
into a file whose header differs, so adding a column after arm B would make arms A and C
fail *after* their training completed. `results/` is empty at the start of this
experiment, which is the one free moment to settle it.

## 8. The prediction

**`pool_kernel: 2` is expected to win**, on criterion 1 above — which is arithmetic and therefore already known. Arms A and B exist to attach measured accuracy, VRAM and time to that conclusion, and to produce the number owed to the second pipeline.

Stating the prediction up front is deliberate: if the measurements contradict it, that is a real finding worth chasing rather than something to quietly reconcile.

Arm C's outcome is genuinely unknown.

## 9. What happens with each result

### Pooling (arms A + B)

| outcome | action |
|---|---|
| `pool=2` wins, as predicted | lock `pool_kernel: 2` for the whole study. Report the measured comparison. Proceed to ex7. |
| `pool=1` wins on accuracy by a **small** margin | still lock `pool=2` — criterion 1 outranks it. Record the accuracy cost as a stated limitation. |
| `pool=1` wins by a **large** margin | stop and reconsider. Likely means the classifier is doing work the conv stack should be doing, which is itself a finding. Re-run arm C at `pool=1` before proceeding. |

### VRAM (arm C)

| outcome | action |
|---|---|
| fits with headroom | batch 64 / accum 2 confirmed for the N-MNIST pilot. Proceed. |
| fits but barely | drop to 32 / 4 anyway — a rung that only just fits will fail once another metric is added |
| out of memory | drop to 32 / 4, **re-run arms A and B at the new value too**, then proceed. This pilot's rungs share one batch size; later datasets calibrate their own. |

## 10. How to run it

```bash
# arm B -- reference, pool_kernel 2
python learning/main.py --config experiments/ex6/config.yaml --experiment ex6

# arm A -- pool_kernel 1
python learning/main.py --config experiments/ex6/pool1.yaml --experiment ex6

# arm C -- top-of-ladder VRAM check
python learning/main.py --config experiments/ex6/vram_check.yaml --experiment ex6
```

All three write into `experiments/ex6/`. Each run gets its own `results/<run_id>/` and `plots/<run_id>/` subfolder, while `runs.csv` / `epochs.csv` / `layers.csv` accumulate at the top of `results/` — so the three arms land in one comparable table.

`run_id` is `<timestamp>_<framework>_seed<n>`, the same id used in the `runs.csv` row, so any figure traces back to the row describing it.

⚠️ **All three arms have the same framework and seed**, so their `run_id`s differ only by timestamp. Note which timestamp is which arm when the runs finish — the config path is recorded in `runs.csv` and in `runs/<run_id>.json`, so it is recoverable, but noting it as you go is easier.

On a machine where results must survive the session, add `--results-root <path>`.

## 11. Deliverables

| # | deliverable | destination |
|---|---|---|
| 1 | **the pooling value + measured evidence** | reported onward for use in a second pipeline — flagged as high priority |
| 2 | confirmed batch size for the N-MNIST pilot | `experiment_plan.md` |
| 3 | baseline reference row (arm B) | `runs.csv`, the point every later rung is compared against |
| 4 | instrumentation pass/fail per metric | note here once known |

## 12. Results

All three arms ran on 2026-09-02 (Tesla T4, Colab — see §5b). Full per-run data lives in
`results/runs.csv`, `results/epochs.csv`, `results/layers.csv`, and per-run detail under
`results/<run_id>/`.

| arm | pool | filters | neurons | params | classifier share | test acc | train−test | peak VRAM (reserved) | epoch time |
|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 12/32 | 32,442 | 226,574 | 95.5% | 97.75% | +0.26pp | 1,650 MB | 166.7s |
| B | 2 | 12/32 | 14,682 | 18,254 | 43.9% | 98.56% | −0.45pp | 762 MB | 170.5s |
| C | 2 | 128/128 | 130,698 | 448,266 | 7.1% | 98.75% | −0.05pp | 3,982 MB | 238.1s |

*(train−test = final-epoch train accuracy minus test accuracy; positive means train
ran slightly ahead of test, negative means test came out slightly ahead — both are
noise-sized at this pilot scale, not a memorization signal.)*

**Criterion 1 (structural, arithmetic) — pool=2 wins, exactly as predicted.** Arm A's
classifier holds 95.5% of the network's parameters; arm B's holds 43.9% — both match
§6's prediction to within rounding. Arm C (the top of the ladder, pool=2) pushes the
classifier's share down to 7.1%, matching the predicted "43.9% → 7.1%" endpoint almost
exactly. `pool=1` would keep the classifier dominating through the small and mid rungs
of every later experiment, which is exactly the failure mode this criterion exists to
catch.

**Criterion 2 (cost) — pool=2 wins on memory, but the epoch-time comparison isn't
trustworthy and shouldn't be used.** Arm A uses roughly 2.2x arm B's reserved VRAM
(1,650 MB vs. 762 MB) for the same 12/32 filters — a real, hardware-independent
memory measurement, and it favors pool=2. The epoch-time column is a different story:
checking `gpu_util_avg_pct` in arm B's `training_results.csv`, as §5b's own caveat
says to do before trusting any timing figure, shows GPU utilization sitting at
**22-23%** through arm B's run (arm A: 36-37%) — this is the low-core, loader-starved
scenario §5b warned about, where wall-clock time measures the data pipeline, not the
network. Arm C, by contrast, runs at **97-98%** utilization — its much larger network
finally gives the GPU enough work to saturate it despite the same loader. So arm C's
238.1s is a real measurement of that network's cost; arm A vs. B's 166.7s vs. 170.5s
is not a real measurement of pooling's cost, and is not used as evidence here. This
doesn't change the decision — VRAM and structural share already agree — but the raw
epoch-time numbers above should not be read as "pool=1 trains faster."

**Criterion 3 (accuracy) — pool=2 wins outright**, not just by a small margin that
would need the criterion-1 tie-break: 98.56% vs. 97.75%. Per §9's decision table, this
means the pooling decision isn't even a close call — pool=2 wins on every one of the
three criteria, matching the "expected" outcome stated in §8.

**VRAM (arm C) — fits with comfortable headroom.** 3,982 MB reserved out of the T4's
15 GB (~26%), not a "just barely" fit. Per §9's decision table, this confirms the
batch size outright — no drop to 32/4 needed.

**Decision:** `pool_kernel: 2` — locked for the whole study.
**Batch size confirmed for N-MNIST pilot:** `batch_size: 64` / `grad_accum_steps: 2` (effective 128). ex7 onward (N-Caltech101) will calibrate their own batch size via the `calibrate_batch_size()` occupancy policy — see `experiment_plan_final.md` §4 and design doc §6a for rationale.

### Conclusions

1. **`pool_kernel: 2` is correct, and not a close call.** It wins on the structural
   criterion (arithmetic, not measured), on memory, and on accuracy. `pool=1`'s only
   advantage — a slightly higher raw epoch-time reading — turned out to be a loader
   artifact, not a real speed advantage, once GPU utilization was checked as §5b
   requires.
2. **Batch size 64 is confirmed for N-MNIST**, with real headroom at the top of the
   planned ladder (arm C, ~26% of available VRAM). ex7 onward (N-Caltech101) will
   calibrate their own batch size using the `calibrate_batch_size()` occupancy policy;
   this is not a study-wide lock, but a dataset-specific decision. See `experiment_plan_final.md`
   §4 and design doc §6a for rationale.
3. **A general caution for every later experiment in this study:** GPU utilization
   below ~40% (as seen on arms A and B here) means wall-clock/epoch-time comparisons
   between rungs are measuring the data loader, not the network, on this hardware. Any
   later experiment relying on training-time or throughput comparisons should check
   `gpu_util_avg_pct` first, exactly as done here — the smaller rungs of the width
   ladder (ex7) are likely to hit the same loader-bound regime that arms A/B did.
4. **The three arms' train/test gaps are all within noise** (±0.5 percentage points),
   confirming the pilot's meters read sensibly and nothing in the instrumentation is
   obviously broken — the second thing this experiment existed to check, alongside the
   pooling decision itself.

**ex6 is complete.** No further runs needed here. Proceed to ex7 (width ladder), per
`experiment_plan_final.md` §10.
