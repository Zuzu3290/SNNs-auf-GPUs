# Next Steps — Pick up after break

**Last session: 2026-05-26.** Resume from here when you come back.

---

## ⚡ READ THIS FIRST — direction changed at end of last session

We pivoted from "empirical-best" baselines to **"library-canonical / conceptually-grounded"** configs. New target: present each framework at the config that has principled justification for every choice, even if marginally lower accuracy.

**YAML is currently set up for EXP-007 (SNNTorch library-canonical). Just run it.** Then EXP-008 (Norse equivalent). See "Direction shift" section below for full config + reasoning.

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
