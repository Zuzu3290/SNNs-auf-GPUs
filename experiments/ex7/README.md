# Experiment 7 — Width ladder (Factor A)

**Study:** Boundaries of Emergence — Identifying Capacity Thresholds in Small-Scale SNNs
**Position:** the main event of the scalability study. Runs after ex6 (pilot, complete)
and before ex8 (depth ladder).
**Dataset:** N-Caltech101 · **Backend:** SpikingJelly (`sj`) · **Rungs:** 4 + 1 extended
check · **Seeds:** 1 per rung now, 3 on the winner once known

---

## 1. What this experiment decides

Grow the network **wider** — more filters per conv layer — one step at a time, and
find the point where it stops being worth it. The number where that happens is the
**Filter Ceiling**, one of the study's headline deliverables.

| # | question | answer form |
|---|---|---|
| 1 | Where does accuracy stop improving enough to justify the extra VRAM? | one filter count, the Filter Ceiling |
| 2 | Is that ceiling structural, or just under-trained? | the extended-epoch check on the smallest rung |
| 3 | How do Participation Ratio, spike entropy, and mutual information change as the network widens? | one reading per metric per rung — this is what ex6 could not measure (no capacity metrics existed yet) |

## 2. Why this runs on N-Caltech101, not N-MNIST

Per `experiment_plan_final.md` §2: N-MNIST was pilot-only, used up in ex6. Every
substantive experiment from here on — ex7, ex8, ex9 — runs on N-Caltech101: 101 classes
on ~7k samples is small and hard enough that a too-small network genuinely fails, which
is the whole point of the study. **ex6's settings still apply**: `pool_kernel: 2`,
timesteps `T=16`, backend `sj`, constant LR — all locked for the whole study regardless
of dataset. **ex6's batch size does not apply** — see `config.yaml`'s comments and
§4 below.

## 3. A caveat, decided before this experiment was designed

Checking the real numbers before writing this doc surfaced something ex6 could not have
caught (it never tested N-Caltech101): **the classifier dominates the network's
parameter count at every filter size in this sweep**, not just at the small end the way
it did on N-MNIST before pooling fixed it.

| filters | flatten width | classifier params | conv params | classifier's share |
|---|---|---|---|---|
| 16 | 38,304 | 3,868,805 | 7,232 | 99.8% |
| 32 | 76,608 | 7,737,509 | 27,264 | 99.6% |
| 64 | 153,216 | 15,474,917 | 105,728 | 99.3% |
| 128 | 306,432 | 30,949,733 | 416,256 | 98.7% |

Pooling helped on N-MNIST's 34×34 sensor, but N-Caltech101's 180×240 sensor leaves a
flatten width of `filters × 2,394` even after two pooling stages — big enough that the
classifier stays essentially the whole network regardless of filter count. Widening
barely moves this ratio (99.8% → 98.7% across the entire sweep).

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

## 4. Batch size: recalibrated per rung, not pinned

This is a deliberate reversal of ex6's own approach — see `config.yaml`'s comments and
`docs/superpowers/specs/2026-09-04-capacity-metrics-design.md` §6a for the full
reasoning. In short: `calibrate_batch_size()` is an occupancy policy (fill 30-35% of
VRAM), not an experimental variable to hold fixed. A bigger network needs more memory
per sample, so it earns a smaller batch — that's correct behavior, not a confound.
Every rung's actual calibrated batch size gets recorded in `runs.csv`; check it before
comparing wall-clock/throughput figures across rungs, the same way ex6 had to check
GPU utilization before trusting its own epoch-time numbers.

## 5. The rungs

| file | filters (both stages) | epochs | purpose |
|---|---|---|---|
| `f16.yaml` | 16 | 30 | smallest rung |
| `f32.yaml` | 32 | 30 | today's default width |
| `f64.yaml` | 64 | 30 | — |
| `f128.yaml` | 128 | 30 | largest rung (matches ex6 arm C's filter count, different dataset) |
| `f16_extended.yaml` | 16 | 100 | structural-ceiling check, see §6 |

All five `extend` a shared `config.yaml` (dataset, timesteps, LR schedule,
`compute_capacity_metrics: true`) and change only what's named in the table.

**30 epochs is an assumption, not a validated number** — no prior run of this pipeline
on N-Caltech101 exists to calibrate against. Check `f16.yaml`'s own training curve
first: if train accuracy is still climbing meaningfully at epoch 30, the whole ladder's
epoch count needs raising before the Filter Ceiling comparison can be trusted, since
every rung must train for the same budget to be comparable.

## 6. The extended-epoch check

`f16_extended.yaml` trains the *smallest* rung for ~3.3x the standard budget (100 vs.
30 epochs). Purpose: prove the smallest rung's ceiling is **structural** — the network
genuinely cannot represent more, not "it just needed more training." This is the first
objection anyone reviewing a Filter Ceiling claim will raise, so it's answered up front
rather than after the fact. If train accuracy at epoch 100 has clearly plateaued well
before then, the ceiling is structural. If it's still climbing meaningfully, the
30-epoch budget was too short for every rung, not just this one.

## 7. The stopping rule — the Filter Ceiling

Per `experiment_plan_final.md` §6: stop early if a rung gives **less than 1% accuracy
gain for more than 10% VRAM increase** over the previous rung. The last rung before
that threshold trips is the Filter Ceiling. If accuracy is still rising meaningfully at
filters=128, the ladder was too short and needs an additional rung above it before the
ceiling can be declared.

## 8. Seeds

Per `experiment_plan_final.md` §9: single seed (0) for every rung in this first pass —
that's what `config.yaml` sets. Once results are in and the Filter Ceiling rung is
identified, re-run **that one rung** at seeds 1 and 2 (2 more runs, not built as
separate config files here since the winner isn't known yet) to back the "accuracy
variance <2%" part of the Stable/Factory-Correct naming in `experiment_plan_final.md`
§7. Do not add seeds to every rung — that's the seed-budget blowup the plan's §9
deliberately avoids.

## 9. What gets recorded

Same schema every other experiment uses (`runs.csv`, `epochs.csv`, `layers.csv` at
`schema_version: 4`). This is the first experiment where the capacity-metric columns
(`participation_ratio`, `participation_ratio_normalized`, `spike_entropy`,
`spike_entropy_normalized`, `mutual_info_zy`, `grad_norm_mean`) will actually be
populated with real numbers, computed over the full test set for every hooked layer
(`lif1`, `lif2`, `lif_out`).

**Sanity-check the meters on the first rung before trusting the rest of the ladder**,
the same way ex6 checked its own instrumentation:
- `cv_isi_mean` and `total_synops_energy_pj` in the run's JSON should be **nonzero**
  (a pre-existing bug that silently zeroed both was fixed alongside this experiment's
  own capacity metrics — see the capacity-metrics design doc §"Revision 2" fix log).
- `total_neurons` should scale roughly with filter count squared for the conv layers
  (lif1/lif2), matching how neuron count is computed from spatial size × filter count.
- `gpu_util_avg_pct` should be checked before trusting any timing/throughput comparison
  across rungs (§4).

## 10. Deliverables

| # | deliverable | destination |
|---|---|---|
| 1 | **Filter Ceiling** — a number | `experiment_plan_final.md` §8 |
| 2 | Trade-off curve — accuracy/VRAM/capacity metrics vs. filter count | this experiment's results, all 4 rungs |
| 3 | Confirmation the ceiling is structural | the extended-epoch check, §6 |
| 4 | Confusion matrix per rung | `runs.csv` + the confusion-matrix data each run already produces |
| 5 | First real capacity-metric readings for the study | `layers.csv`, all rungs |

## 11. How to run it (Colab)

Steps 1-3 are CPU-only sanity checks — run them once, before spending real GPU time:

```bash
python check_env.py
python check_network.py --config experiments/ex7/f16.yaml --all
python equivalence_check.py --config experiments/ex7/f16.yaml --experiment ex7
```

Then the four rungs plus the extended check, one at a time (each is its own process —
run them in separate cells so a crash in one doesn't lose the others):

```bash
# rung 1 -- filters=16
python learning/main.py --config experiments/ex7/f16.yaml --experiment ex7 --framework sj --seed 0 --inference stats

# rung 2 -- filters=32
python learning/main.py --config experiments/ex7/f32.yaml --experiment ex7 --framework sj --seed 0 --inference stats

# rung 3 -- filters=64
python learning/main.py --config experiments/ex7/f64.yaml --experiment ex7 --framework sj --seed 0 --inference stats

# rung 4 -- filters=128
python learning/main.py --config experiments/ex7/f128.yaml --experiment ex7 --framework sj --seed 0 --inference stats

# extended-epoch check -- filters=16, 100 epochs
python learning/main.py --config experiments/ex7/f16_extended.yaml --experiment ex7 --framework sj --seed 0 --inference stats
```

After a Colab run, pull results back and build plots (no GPU needed for either):

```bash
python collect_results.py --from <drive folder> --experiment ex7
python make_plots.py --experiment ex7
```

Running locally instead of Colab? Skip the `collect_results.py` step — `learning/main.py`
already writes where `make_plots.py` reads. On a machine where results must survive the
session, add `--results-root <path>` to every `learning/main.py` call.

**Check after rung 1 (`f16.yaml`) specifically, before running the rest:** does
`cv_isi_mean` read nonzero, does the epoch count look sufficient (§5's caveat), and does
the calibrated batch size look sane for this dataset? Catching a problem here costs one
run; catching it after all 5 have finished costs the whole experiment.

## 12. Results

*(to be filled in once all rungs have run)*

| rung | filters | calibrated batch size | neurons (lif1+lif2) | test acc | peak VRAM (reserved) | PR (lif1) | PR (lif2) | entropy (lif1) | entropy (lif2) | I(Z;Y) (lif2) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | | | | | | | | | |
| 2 | 32 | | | | | | | | | |
| 3 | 64 | | | | | | | | | |
| 4 | 128 | | | | | | | | | |
| extended | 16 (100ep) | | | | | | | | | |

**Filter Ceiling:** _pending_
**Structural or under-trained:** _pending_
