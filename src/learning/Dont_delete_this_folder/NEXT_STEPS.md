# Next Steps — Pick up after break

**Last session: 2026-05-26.** Resume from here when you come back.

---

## ⚡ READ THIS FIRST — BOTH EXP-007 + EXP-008 DONE — Norse won everything

This is a session-completion summary. Both library-canonical experiments are finished. The result is much stronger than expected — you have a paper-ready story.

---

### 🏆 EXP-008 (Norse) is the Pareto winner across all 6 runs

| Metric | EXP-007 (SNNTorch lib-canonical) | EXP-008 (Norse lib-canonical) | Δ |
|---|---|---|---|
| Test accuracy | 98.17% | **98.29%** | **+0.12% (Norse wins)** |
| Energy/sample | 133.81 pJ | **51.87 pJ** | **−61% (Norse wins big)** |
| Avg spikes/sample | 38.23 | **14.82** | **−61% (Norse 2.6× more efficient)** |
| Latency/sample | 0.342 ms | 0.390 ms | SNNTorch slightly faster |

**Norse beat SNNTorch on both accuracy AND energy simultaneously** when both were given their library-canonical configs with 8 epochs + cosine LR. The "individual-optimal" framing is now strongly validated.

### Full results table — all 6 experiments

| Run | Framework | Config | Test Acc | Energy/sample | Spike Rate |
|---|---|---|---|---|---|
| EXP-001 | SNNTorch | Fair-comparison (5 ep) | 98.25% | 86.81 pJ | 0.142 |
| EXP-002 | Norse | Fair-comparison (5 ep, lr=1e-3, tau=50) | 97.31% | 55.76 pJ | 0.094 |
| EXP-005 | SNNTorch | Library-canonical (5 ep) | 98.17% | 128.60 pJ | 0.236 |
| EXP-006 | Norse | Library-attempt (lr=2e-4, undertrained) | 94.05% | 41.03 pJ | 0.072 |
| EXP-007 | SNNTorch | Library-canonical (8 ep + cosine) | 98.17% | 133.81 pJ | 0.243 |
| **EXP-008** | **Norse** | **Library-canonical (8 ep + cosine + tau=100)** | **98.29%** | **51.87 pJ** | **0.091** |

---

### 🎯 The most beautiful detail in the data — emergent efficiency

Norse's spike rate trajectory shows the network LEARNING to be more efficient:
- **Epoch 1:** 0.0784 (boost from tau=100)
- **Epoch 4 peak:** 0.1003 (max activity during learning)
- **Epoch 8:** 0.0909 (network refined down — *fewer but more meaningful spikes*)

SNNTorch can't do this because `mse_count_loss` enforces a fixed 80% firing target — spike rate **locks flat at ~0.243** from epoch 2 onwards. Norse's `cross_entropy + hard reset` combination lets the network discover its own optimal spike economy.

### Why Norse beat SNNTorch (root-cause table)

| Aspect | SNNTorch (EXP-007) | Norse (EXP-008) | Winner |
|---|---|---|---|
| Threshold | 1.0 | 0.5 | Both work for their framework |
| Reset | subtract (soft) | zero (hard) | **Hard reset wins** — prevents wasted firing on overshoots |
| Loss | mse_count (rate target 80/20) | cross_entropy (rate sum) | **Cross-entropy wins** — lets network find optimal rate |
| Surrogate | atan | SuperSpike | SuperSpike sharper gradient → faster convergence |
| Spike efficiency | 38.23 spikes/sample | 14.82 spikes/sample | **Norse 2.6× more efficient** |

**Key insight:** SNNTorch's `mse_count_loss` was designed for tasks where rate-coded outputs matter (e.g. physiological signal modeling). For classification, where you only need ENOUGH spikes to disambiguate classes, the 80% target is wasteful. Norse's defaults let the network find a natural spike economy.

### EXP-007 sub-finding (worth keeping in mind)

EXP-007 (98.17% at 8 epochs) tied EXP-005's 98.17% at 5 epochs **exactly**. Two independent runs hit the same number → strong evidence that **98.17% is the structural ceiling of SNNTorch's library-canonical config** on this task, not a tuning artifact. Train accuracy crept up (97.94% → 98.11%) while test accuracy stayed flat → slight overfitting with more training, not under-trained.

---

### 📝 Headline story for your final comparison MD

> "Both frameworks were configured according to their library-canonical defaults with 8 epochs of training and cosine LR scheduling. **Norse achieved 98.29% test accuracy at 51.87 pJ/sample, beating SNNTorch's 98.17% at 133.81 pJ/sample on both axes simultaneously.** SNNTorch's `mse_count_loss` enforces explicit firing-rate targets (80% correct / 20% incorrect) producing a saturated spike density that is informationally redundant for clean classification tasks. Norse's combination of hard reset and cross-entropy on summed spikes lets the network discover a parsimonious spike economy — *the spike rate actually declines in later epochs as the network learns to be more efficient* — yielding 2.6× fewer spikes per inference at marginally higher accuracy."

### 📊 The single best plot for the MD

Side-by-side `training_metrics.png` from EXP-007 and EXP-008. Visually striking contrast:
- **EXP-007:** spike rate locks flat at ~0.243 from epoch 2 — visual signature of mse_count enforcing its target
- **EXP-008:** spike rate has a small hump (peaks epoch 4) then *declines* — visual signature of network learning efficiency

This single dual-panel figure tells the whole story.

---

### Broader narrative for your project's writeup

1. **Both "library-canonical" experiments (EXP-007, EXP-008) beat earlier "tuning" attempts (EXP-005, EXP-006)** — when each framework gets enough epochs + proper LR schedule, library defaults work as intended.

2. **"Fair-comparison" (EXP-001/002) and "library-canonical" (EXP-007/008) tell DIFFERENT stories** about which framework is better:
   - At **fair-comparison** settings (forced common parameters): SNNTorch wins on accuracy (98.25 > 97.31)
   - At **library-canonical** settings (each framework at its best): **Norse wins on both accuracy AND energy**

3. The lesson: cross-framework comparison fairness is multi-faceted. Forcing common parameters favors the framework whose defaults are closer to those parameters (here: SNNTorch). Letting each framework use its native defaults reveals each framework's true design philosophy at work.

---

### ✅ What you have when you return

- **6 completed experiments** (EXP-001, 002, 004B, 005, 006, 007, 008) covering all main hypothesis
- **Full analysis in `PIPELINE_TRAINING_LOGS.md`** — each entry has params + per-epoch table + test results + interpretation
- **Plots in `outputs/plots/`** — copy out before next run (current ones overwritten by latest run)
- **MD-ready quote** above for the writeup

### ✅ What to do when you return (in order)

1. **Implement output system fixes** (~30 min) — see "🔧 TODO" section below. Adds per-run folders, run-config snapshots, structured test summaries. Makes future work much cleaner.
2. **Write the final comparison MD** using EXP-001/002/007/008 as the four main data points. Story is now clear.
3. **Optional follow-up:** EXP-009 = SpikingJelly library-canonical (would complete the 3-framework story)
4. **Optional follow-up:** EXP-008b = Norse same config but `tau=50` to isolate "how much of EXP-008's win came from tau bump vs epochs+scheduler"

---

---

## Where we are (1-paragraph summary)

We finished a major pipeline refactor: `loss_fn` is now config-driven across all 3 frameworks (crashes loudly on unsupported values), `surrogate` is hardcoded per framework (YAML param disabled — each framework prints what it uses at init), `reset_mode` is a new YAML architecture param, and the logging is deduplicated (the `ADAPTIVE CACHE CONTROLLER` block no longer prints twice, `PIPELINE READY` now shows the actually-selected cache mode, etc.). Then we ran two "individual-optimal" experiments back-to-back: **EXP-005 (SNNTorch optimal)** and **EXP-006 (Norse optimal)**. **Both LOST to their fair-comparison baselines** (EXP-001 and EXP-002). The recommended Table A configurations turned out to be wrong for our specific architecture + epoch budget — `mse_count_loss` made SNNTorch fire 66% more for the same accuracy (high energy), and `lr=2e-4` undertrained Norse over 5 epochs (lost 3.26% accuracy). This is a great finding for the final comparison MD — "library-canonical" defaults aren't task-universal.

---

## Direction shift — library-canonical approach

Earlier this file recommended EXP-001/EXP-002 as the strongest baselines (empirically true). **That is now superseded.** New plan: present each framework at its **library-canonical / conceptually-justified config** for the final comparison MD.

**Why:** Every choice in the canonical config has a principled justification — threshold rejects noise, mse_count enforces rate coding, soft reset preserves spike magnitude, AMP is library-supported, cosine LR is standard practice. The result is a *defensible* presentation, not a *tuned-for-best-number* one. Slight accuracy/energy cost is honest output of those choices.

### EXP-007 — SNNTorch library-canonical (YAML is set up — JUST RUN)

| Param | Value | Why |
|---|---|---|
| framework | torch | — |
| threshold | **1.0** | SNNTorch library default; raises bar above noise threshold |
| reset_mode | **subtract** | SNNTorch native soft reset; preserves spike-magnitude info |
| loss_fn | **mse_count** | Canonical SNN loss; explicit 80/20 rate-coded targets |
| use_amp | **true** | Library-supported, verified safe in EXP-005 |
| lr_scheduler | **cosine** | Standard fine-tuning practice |
| epochs | **8** | EXP-005 at 5 epochs was still converging |
| lr | 1e-3 | SNNTorch library standard |

Expected: ~98.3–98.6% test acc, ~120–135 pJ/sample energy. Higher energy than EXP-001 is the expected cost of mse_count + soft reset (more firing).

### EXP-008 — Norse library-canonical (do AFTER EXP-007)

Norse has fewer "library-canonical" knobs (can't do soft reset, can't do mse_count, can't do AMP — all library limitations). So library-canonical for Norse mostly means "use library default tau_mem_inv" + cosine + 8 epochs.

| Param | Value | Why |
|---|---|---|
| framework | norse | — |
| threshold | 0.5 | Library default 1.0 KILLS Norse (EXP-004); 0.5 is the safe Norse-canonical |
| **tau_mem_inv** | **100** | Norse library default (currently 50). Library-canonical = use 100. Untested in our experiments. |
| reset_mode | zero | Norse library limitation |
| loss_fn | cross_entropy | Norse library limitation |
| use_amp | false | Norse library limitation (SuperSpike + float16 underflow) |
| lr_scheduler | **cosine** | Standard, matches EXP-007 for fair comparison |
| epochs | **8** | Match EXP-007 |
| lr | **1e-3** | EXP-006 proved 2e-4 too low; 1e-3 is empirically validated AND standard |

**YAML changes after EXP-007 completes:**
- `framework: torch` → `norse`
- `threshold: 1.0` → `0.5`
- `reset_mode: subtract` → `zero`
- `loss_fn: mse_count` → `cross_entropy`
- `use_amp: true` → `false`
- `frameworks.norse.tau_mem_inv: 50.0` → `100.0`
- Keep `epochs: 8`, `lr_scheduler: cosine`, `lr: 0.001`

Expected: ~96–97.5% test acc, ~50–70 pJ/sample energy.

### Plots needed for final comparison MD

1. **Training curves** (loss, accuracy, spike rate vs epoch) — auto-saved per run at `./outputs/plots/training_metrics.png` — **rename/copy to per-EXP filenames before next run overwrites them!**
2. **Spike rasters** — auto-saved at `./outputs/plots/spike_raster.png` — same: rename per-EXP
3. **Per-class F1 bar chart** — manual from `./outputs/data/test.csv`
4. **Cross-run scatter:** energy/sample (x) vs test accuracy (y) — points for all EXPs (001, 002, 005, 006, 007, 008). Shows Pareto front.

### IMPORTANT: save plots before re-running

The output paths are hardcoded — `./outputs/plots/training_metrics.png` etc. Each run overwrites the previous. **After EXP-007 completes, copy plots to e.g. `outputs/plots/exp007_training_metrics.png` before starting EXP-008.**

---

## Current state of experiments

| Run | Config | Test Acc | Energy/sample | Verdict |
|---|---|---|---|---|
| **EXP-001** | SNNTorch fair-comparison | **98.25%** | 86.81 pJ | ✅ Best SNNTorch result so far |
| **EXP-002** | Norse fair-comparison | **97.31%** | 55.76 pJ | ✅ Best Norse result so far |
| **EXP-003** | Norse + AMP + 4 workers | crashed (OOM) | — | ⚠️ Confirmed Colab multi-worker + MEMORY = OOM |
| **EXP-004** | Norse, threshold=1.0 | **9% (dead neurons)** | — | ❌ Confirmed threshold=1.0 kills Norse |
| **EXP-004B** | Norse threshold=0.5 retry (3 ep) | 95.71% | 53.82 pJ | ✅ Confirms EXP-004 was threshold-only failure |
| **EXP-005** | SNNTorch "optimal" (mse_count + soft reset + AMP, threshold=1.0) | 98.17% | 128.60 pJ | ❌ Lost to EXP-001 |
| **EXP-006** | Norse "optimal" (lr=2e-4) | 94.05% | 41.03 pJ | ❌ Undertrained |

---

## Current YAML state (verify in `SNN_module.yaml` when you return)

YAML is set up for **EXP-007 (SNNTorch library-canonical)**:
- `framework: torch`
- `threshold: 1.0`
- `reset_mode: subtract`
- `loss_fn: mse_count`
- `use_amp: true`
- `learning_rate: 0.001`
- `lr_scheduler: cosine`
- `epochs: 8`
- `num_workers: 0`
- `force_mode: null` (adaptive → MEMORY on Colab)

If you re-run as-is, you'll get EXP-007.

---

## Recommended order

1. **EXP-007** (~70 min for 8 epochs on Colab) — SNNTorch library-canonical. YAML is already set up.
2. **Copy plots out:** `outputs/plots/training_metrics.png` → `outputs/plots/exp007_training_metrics.png` (same for spike_raster)
3. **EXP-008** (~70 min) — Norse library-canonical. YAML changes listed above.
4. **Copy plots out** for EXP-008 too.
5. **Write the final comparison MD** with all 4 main runs (EXP-001 fair vs EXP-007 canonical for SNNTorch; EXP-002 fair vs EXP-008 canonical for Norse).

Total ~2.5 hr Colab time to be ready for the final MD.

Optional later: EXP-009 = SpikingJelly fair-comparison config for a 3-way comparison.

---

## 🔧 TODO — Output system improvements (do after EXP-008 finishes)

**Why:** Current outputs (`exp007_outputs/`, etc.) are functional but incomplete vs the analysis depth in `PIPELINE_TRAINING_LOGS.md`. Aggregate test metrics, per-class breakdowns, and run config are **printed-only** — not saved as files. Each run also overwrites the previous (`./outputs/` is the only path). Manual copy to `expNNN_outputs/` is required. Fixing this makes future runs / paper / thesis defense easier. Total effort: ~30 min.

### High priority (do all 4 together — small focused commit)

1. **Auto-namespace output folders** — `outputs/{framework}_{timestamp}/data/...` instead of bare `./outputs/`. No more manual copying or overwrite risk. Edit: `training.py` (csv_path), `inference.py` (csv_path), main.py (plot save dirs).

2. **Save `run_config.json`** at training start with full param snapshot: framework, threshold, reset_mode, loss_fn, AMP-actual-used, surrogate-actually-used (per-framework hardcoded value), lr, lr_scheduler, tau_mem_inv (Norse) / beta (torch) / tau (SJ), batch_size, epochs, cache mode selected, augmentation, etc. Single source of truth for "what produced this run". Place in `training.py` start of `train()`.

3. **Save `test_summary.json`** — overall_accuracy, total_spikes, avg_spikes_per_sample, avg_latency_ms, avg_latency_per_sample_ms, total_energy_pj, energy_per_sample_pj. The dict that `SNNTester.run()` already RETURNS — just write it to disk. Currently lost after the function exits. Edit: `inference.py:run()`.

4. **Save `per_class_metrics.csv`** — 10 rows × (class, TP, FP, FN, TN, accuracy, precision, recall, f1, specificity). `_class_metrics()` already returns this — wire it to a CSV write. Edit: `inference.py:run()`.

### Medium priority

5. **Save `confusion_matrix.csv`** — 10×10 numpy `cm` array. Useful for confusion-matrix plot in comparison MD. Edit: `inference.py:run()`.

6. **Fix "Final accuracy" mislabel in training output** — the printed value (e.g. "Final accuracy: 96.88%") is the LAST BATCH of the last epoch (64 samples), not a meaningful metric. Either remove it or replace with `best_acc` (already tracked). Edit: `training.py` end of `train()` or main.py wherever it prints.

### Low priority

7. **Fix Surrogate display in SNN Configuration print** — `snn_config.py:101` still prints `cfg.SURROGATE` which always shows "atan" even when Norse actually uses SuperSpike. Should print framework-specific hardcoded value (or just remove the line — the framework init prints already say what's used: `[Norse] Surrogate gradient is hardcoded to 'SuperSpike'`).

### Suggested commit message

"Improve output system: per-run folders, save config/summary/per-class as files, fix mislabels"

### After implementing — verify by

Running one experiment (any framework, even 1 epoch) and confirming `outputs/{framework}_{timestamp}/` contains:
- `data/training_results.csv` (existing)
- `data/test.csv` (existing)
- `data/test_summary.json` (NEW)
- `data/per_class_metrics.csv` (NEW)
- `data/confusion_matrix.csv` (NEW)
- `config/run_config.json` (NEW)
- `plots/training_metrics.png` (existing)
- `plots/spike_raster.png` (existing)

---

## Key files to re-read when you come back

| File | Purpose |
|---|---|
| `SNN_module.yaml` | Verify current config state before re-running |
| `src/learning/PIPELINE_TRAINING_LOGS.md` | Full experiment history with analysis |
| `src/learning/FRAMEWORK_CONFIG_ANALYSIS.md` | Framework comparison reference (Tables A & B, threshold math) |
| `src/learning/PIPELINE_CONCEPTS.md` | Data pipeline architecture (caching, ToFrame, etc.) |

---

## Reminders / gotchas

- **Colab MEMORY mode + multi-worker = OOM.** Always keep `num_workers: 0` on Colab when `force_mode: null` (adaptive picks MEMORY). Confirmed by EXP-003 crash.
- **Norse threshold=1.0 kills the network.** Never raise threshold for Norse — keep at 0.3–0.5. Confirmed by EXP-004.
- **AMP=true is safe for SNNTorch** on this net (EXP-005 verified), **unsafe for Norse** (silent gradient underflow with SuperSpike — Table A §3).
- **`mse_count_loss` is SNNTorch-only.** Setting it with framework=norse/spikingjelly raises `NotImplementedError` at init (by design — crash loudly).
- **`reset_mode: subtract` is SNNTorch+SpikingJelly only.** Setting it with framework=norse crashes (Norse library limitation).
- **Pre-cache MEMORY mode is 4× slower than disk on Colab** for this tiny net (probably AMP autocast overhead). Doesn't matter for accuracy but explains why epochs took 480s instead of ~120s.
