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
| **2** | Does the **largest** planned network fit in VRAM at the shared batch size? | yes / no / barely |
| **3** | Do all the meters read sensibly before real GPU time is spent? | pass / fail per metric |

Question 1 is the deliverable. Questions 2 and 3 ride along for free on the same three runs.

## 2. Why it has to run first

Pooling controls the spatial geometry of every layer after it. Change it and the flatten width changes, which changes the classifier's size, which changes the parameter count, the VRAM bill and the neuron count.

> **No two runs with different pooling are comparable to each other.** Every rung of every later experiment must share one value, so it has to be chosen before the ladder starts — not discovered midway.

The batch size has the same property (see [`config.yaml`](config.yaml)), which is why arm C is here too: both settings that must be global get fixed in one experiment.

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
| fits with headroom | batch 64 / accum 2 confirmed for the study. Proceed. |
| fits but barely | drop to 32 / 4 anyway — a rung that only just fits will fail once another metric is added |
| out of memory | drop to 32 / 4, **re-run arms A and B at the new value too**, then proceed. All rungs share one batch size. |

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
| 2 | confirmed batch size for the whole study | `experiment_plan.md` |
| 3 | baseline reference row (arm B) | `runs.csv`, the point every later rung is compared against |
| 4 | instrumentation pass/fail per metric | note here once known |

## 12. Results

*(to be filled in once the three arms have run)*

| arm | pool | filters | neurons | params | classifier share | test acc | train−test | peak VRAM | epoch time |
|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 12/32 | | | | | | | |
| B | 2 | 12/32 | | | | | | | |
| C | 2 | 128/128 | | | | | | | |

**Decision:** _pending_
**Batch size confirmed:** _pending_
