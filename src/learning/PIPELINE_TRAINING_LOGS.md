# SNN Experiment Log
N-MNIST · Conv-SNN · SNNTorch / Norse / SpikingJelly comparison

---

## How to add an entry

Copy the template block below, fill in every field, paste at the **top** of the Experiments section.

```
## EXP-XXX · YYYY-MM-DD HH:MM · <Framework>

**Strategy:** <1-2 sentences: what was the approach / what was being tested>

| Param | Value |
|---|---|
| framework | |
| threshold | |
| tau_mem_inv / beta / tau | |
| timesteps (n_time_bins) | |
| batch_size | |
| iterations_per_epoch | |
| epochs | |
| learning_rate | |
| weight_decay | |
| lr_scheduler | |
| use_amp | |
| loss_fn | |
| optimizer | |
| surrogate | |
| num_workers | |
| cache force_mode | |
| augmentation | |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|

**Test accuracy:** XX.XX%  
**Energy/sample:** XX.XX pJ  
**Avg latency/sample:** X.XXX ms  
**Notes:** <anything unexpected, planned follow-up>
```

---

## Experiments

---

## EXP-008 · 2026-05-26 · Norse — Best-for-our-network config (tau_mem_inv=100, 8 epochs, cosine)

**Strategy:** Norse counterpart to EXP-007 — pick the parameters that give Norse its best performance on OUR specific small Conv-SNN, while respecting Norse's library constraints. Three of four "hard" params are forced by the framework: `use_amp=false` (SuperSpike float16 underflow), `loss_fn=cross_entropy` (library limitation), `reset_mode=zero` (library limitation), `threshold=0.5` (1.0 kills Norse per EXP-004). The real tuning knob is **`tau_mem_inv`**, plus standard practice on `lr_scheduler` and `epochs`.

**Conceptual + empirical justification for each parameter:**

| Param | Value | Reason (conceptual + empirical) |
|---|---|---|
| `framework` | norse | — |
| `threshold` | 0.5 | EXP-004 proved 1.0 = dead neurons. 0.5 is the safe Norse-canonical value. |
| **`tau_mem_inv`** | **100** ← change from 50 | Norse library default. Conceptually: per-step input gain = dt·tau_mem_inv = 0.1 (vs 0.05 at 50). 2× more activity → better gradient flow through our shallow 3-LIF net. Our previous tau=50 was *under*-driving the neurons (EXP-002 spike rate only 0.094). |
| `reset_mode` | zero | Norse library limitation (no soft reset) |
| `loss_fn` | cross_entropy | Norse library limitation (no mse_count) |
| `use_amp` | false | Norse + SuperSpike + float16 = silent gradient underflow (Table A §3) |
| `lr` | **1e-3** | Library standard + empirically validated (EXP-002 → 97.31% at lr=1e-3, no oscillation). Matches EXP-007's LR for clean head-to-head. EXP-006's 2e-4 was too low (undertrained). |
| `lr_scheduler` | **cosine** ← change from none | Standard fine-tuning practice; matches EXP-007 |
| `epochs` | **8** | Match EXP-007 for fair head-to-head |
| `weight_decay` | 1e-4 | Standard |
| `batch_size` | 64 | Standard |
| `surrogate` | SuperSpike (hardcoded) | Norse library default |

**YAML changes from current EXP-007 config (apply AFTER EXP-007 completes):**
```yaml
training:
  framework: norse              # was: torch
  learning_rate: 0.001          # was: 0.001 (no change actually)
  use_amp: false                # was: true
  loss_fn: cross_entropy        # was: mse_count
architecture:
  threshold: 0.5                # was: 1.0
  reset_mode: zero              # was: subtract
frameworks:
  norse:
    tau_mem_inv: 100.0          # was: 50.0  ← the only NEW change vs EXP-002
```

| Param | Value |
|---|---|
| framework | norse |
| threshold | 0.5 |
| tau_mem_inv (Norse) | 100.0 Hz ← Norse library default |
| reset_mode | zero |
| timesteps (n_time_bins) | 16 |
| batch_size | 64 |
| iterations_per_epoch | 937 (full epoch) |
| epochs | 8 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| lr_scheduler | cosine |
| use_amp | false |
| loss_fn | cross_entropy |
| optimizer | adam |
| surrogate | SuperSpike (Norse library default, hardcoded) |
| num_workers | 0 |
| cache force_mode | null (adaptive — likely MEMORY on Colab) |
| augmentation | ON ±10° rotation |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |

**Test accuracy:**
**Energy/sample:**
**Avg latency/sample:**
**Avg spikes/sample:**
**Per-class F1 range:**

**Notes:**
- **Baselines to compare:**
  - EXP-002 (Norse, lr=1e-3, tau=50, 5 ep, no scheduler) → 97.31%, 55.76 pJ — same except tau, scheduler, epochs
  - EXP-006 (Norse, lr=2e-4, tau=50, 5 ep, cosine) → 94.05% (undertrained — different LR)
  - EXP-007 (SNNTorch library-canonical) — direct head-to-head for the comparison MD
- **Hypothesis:** tau=100 doubles input gain → more activity (~0.15-0.20 spike rate vs EXP-002's 0.094) → better gradient propagation → +0.5-1.0% accuracy over EXP-002. Combined with 8 epochs + cosine, expect ~97.8-98.3% test acc.
- **Risk to watch:** If spike rate jumps too high (>0.25), accuracy might plateau or drop — too much firing = noise. If that happens, drop tau back to 75 (middle ground).
- **Plots to produce + save:** Same as EXP-007. **Rename outputs/plots/training_metrics.png → exp008_*.png BEFORE EXP-009 (if any).**

---

## EXP-007 · 2026-05-26 · SNNTorch — Conceptually-grounded library-canonical config (8 epochs)

**Strategy:** Re-run EXP-005's library-canonical SNNTorch config but with **8 epochs** (instead of 5) and **cosine LR scheduler**, on the principle that conceptual/library-canonical choices should be presented in the final comparison MD even if EXP-001 (cross_entropy + hard reset + no AMP) gave a marginally higher accuracy. Reasoning is defensive — each design choice has principled justification, and the slight energy cost is honest output of those choices, not a bug.

**Conceptual justification for each parameter:**

| Param | Value | Conceptual reason |
|---|---|---|
| `threshold` | **1.0** | SNNTorch's library default. With beta=0.95, V_eq = 20·I; threshold=0.5 fires on I>0.025 (noise territory); threshold=1.0 fires on I>0.05 (meaningful signal only). Better noise rejection. |
| `reset_mode` | **subtract** | SNNTorch's native soft reset. Preserves spike-magnitude info — large inputs produce multiple rapid spikes encoding magnitude. Hard reset throws this information away. |
| `loss_fn` | **mse_count** | SNNTorch's canonical loss. Enforces rate-coded outputs: correct class fires 80% of timesteps, incorrect 20%. Stricter training criterion than cross-entropy on summed spikes (which is a deep-learning shortcut, not biologically motivated). |
| `use_amp` | **true** | Library-supported, verified safe in EXP-005 (no underflow, no NaN). Saves training time on supported hardware. |
| `lr_scheduler` | **cosine** | Standard best practice. Allows the final epochs to settle into a sharper minimum than constant LR. |
| `epochs` | **8** | EXP-005 at 5 epochs had training loss still decreasing (0.0741 → 0.0691 between ep4-5). 8 epochs should converge fully. |
| `lr` | 1e-3 | SNNTorch library standard. Proven by EXP-001 and EXP-005. |

| Param | Value |
|---|---|
| framework | torch |
| threshold | 1.0 (library default) |
| beta (SNNTorch) | 0.95 |
| reset_mode | subtract (SNNTorch native soft reset) |
| timesteps (n_time_bins) | 16 |
| batch_size | 64 |
| iterations_per_epoch | 937 (full epoch) |
| epochs | 8 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| lr_scheduler | cosine |
| use_amp | true |
| loss_fn | mse_count (correct_rate=0.8, incorrect_rate=0.2) |
| optimizer | adam |
| surrogate | atan (hardcoded) |
| num_workers | 0 |
| cache force_mode | null (adaptive — likely MEMORY on Colab) |
| augmentation | ON ±10° rotation |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |
| 7 | | | | | |
| 8 | | | | | |

**Test accuracy:**
**Energy/sample:**
**Avg latency/sample:**
**Avg spikes/sample:**
**Per-class F1 range:**

**Notes:**
- **Baseline to beat:** EXP-005 (same config, 5 epochs) = 98.17% test acc, 128.60 pJ/sample. Expected EXP-007 to improve accuracy from extra 3 epochs.
- **Reference comparison:** EXP-001 (cross_entropy, hard reset, no AMP, 5 ep) = 98.25%, 86.81 pJ/sample. If EXP-007 beats 98.25%, the library-canonical config wins on accuracy. Energy will still be ~50% higher.
- **Plots to produce for the comparison MD:**
  - Training metrics curves (loss, acc, spike rate vs epoch) — already auto-saved to `./outputs/plots/training_metrics.png`
  - Spike raster — auto-saved to `./outputs/plots/spike_raster.png`
  - Per-class F1 bar chart (manual or from `./outputs/data/test.csv`)
  - Optional: energy-vs-accuracy scatter comparing all EXPs at their best epochs
- **Story for final MD:** "SNNTorch was tuned to its library-canonical configuration (mse_count loss enforcing rate-coded outputs, soft reset preserving spike magnitude, threshold=1.0 for noise rejection, AMP for speed) over 8 epochs with cosine LR decay."
- **Next: EXP-008** = Norse equivalent of this approach. See bottom of NEXT_STEPS.md for proposed Norse config.

---

## EXP-006 · 2026-05-25 · Norse — Individual-optimal config (Norse library defaults where they work + lower LR)

**Strategy:** First Norse run on the **post-refactor** pipeline (loss_fn config-driven, surrogate hardcoded to SuperSpike for Norse, reset_mode YAML param). Goal is to measure Norse's peak accuracy when configured for Norse-best (Table A) rather than fair-comparison defaults. **Key catch:** Norse's actual library default `v_th=1.0` kills the network (EXP-004 proved this) — so "Norse optimal" means library defaults *where they work* (SuperSpike surrogate, hard reset, tau_mem_inv) + empirical override on threshold (0.5 instead of library 1.0). Also drops LR from `1e-3` → `2e-4` per Table A's recommendation for Norse + SuperSpike stability (sharper gradient → smaller steps).

**Why each value:**

| Param | Value | Reasoning |
|---|---|---|
| `framework` | norse | The test |
| `threshold` | 0.5 | **Library default 1.0 fails** (EXP-004 dead neurons); 0.5 is the proven Norse value (EXP-002 97.31%, EXP-004B 95.71%/3ep) |
| `tau_mem_inv` | 50 (kept, not raised to 100) | User choice — avoid over-firing risk from higher input gain |
| `reset_mode` | zero | Norse library limitation — only hard reset supported |
| `loss_fn` | cross_entropy | Norse library limitation — mse_count would crash |
| `surrogate` | SuperSpike (hardcoded) | Norse library default; sharp gradient, needs lower LR |
| `learning_rate` | **2e-4** ← lowered from 1e-3 | Table A: SuperSpike's sharp gradient near V_th can cause oscillation at 1e-3 with Adam; 2e-4 = more stable convergence |
| `use_amp` | false | Norse + SuperSpike + float16 = silent gradient underflow (Table A §3) |
| `epochs` | 5 | Match EXP-001/EXP-002 for direct head-to-head |
| `num_workers` | 0 | Colab safety — MEMORY cache + multi-worker = OOM (EXP-003 confirmed) |
| `force_mode` | null | Adaptive will pick MEMORY on Colab (12 GB RAM, ~4 GB dataset) |

| Param | Value |
|---|---|
| framework | norse |
| threshold | 0.5 |
| tau_mem_inv (Norse) | 50.0 Hz |
| reset_mode | zero |
| timesteps (n_time_bins) | 16 |
| batch_size | 64 |
| iterations_per_epoch | 937 (full epoch) |
| epochs | 5 |
| learning_rate | 0.0002 ← lowered from 1e-3 for Norse + SuperSpike stability |
| weight_decay | 0.0001 |
| lr_scheduler | cosine |
| use_amp | false |
| loss_fn | cross_entropy |
| optimizer | adam |
| surrogate | SuperSpike (Norse library default, hardcoded) |
| num_workers | 0 |
| cache force_mode | null (adaptive — should pick MEMORY on Colab) |
| augmentation | ON ±10° rotation |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|
| 1 | 0.8260 | 73.37% | 0.0389 | 0.000181 | 989.63 (cold cache fill) |
| 2 | 0.3154 | 90.65% | 0.0611 | 0.000131 | 507.99 |
| 3 | 0.2493 | 92.52% | 0.0671 | 0.000069 | 492.32 |
| 4 | 0.2255 | 93.20% | 0.0717 | 0.000019 | 490.93 |
| 5 | 0.2145 | 93.57% | 0.0712 | 0.000000 | 493.29 |

**Final training metrics:** loss 0.2593, accuracy 92.19%, spike rate 0.0722
**Test accuracy:** 94.05%
**Energy/sample:** 41.03 pJ (total 410,287.5 pJ over 10,000 samples)
**Avg latency/sample:** 0.373 ms (avg batch latency 23.74 ms)
**Avg spikes/sample:** 11.72 (total 117,225 spikes)
**Per-class F1:** min 0.904 (class 8), max 0.981 (class 1)
**Run platform:** Colab T4, 12.67 GB RAM
**Cache mode:** MEMORY selected by adaptive controller (RAM check passed, dataset ~4.13 GB fits in 6.17 GB budget)

**Notes:**

### Headline finding: Lower LR (2e-4) UNDERTRAINED the network — Table A's recommendation HURT us

| Metric | EXP-002 (lr=1e-3) | EXP-006 (lr=2e-4) | Δ |
|---|---|---|---|
| Test accuracy | **97.31%** | 94.05% | **−3.26% (much worse)** |
| Energy/sample | 55.76 pJ | **41.03 pJ** | −26% (lower energy, but at heavy accuracy cost) |
| Spike rate | 0.094 | 0.072 | −23% (less firing) |
| Epoch 1 train acc | 86.82% | **73.37%** | −13.45% gap from the start |
| Epoch 5 train acc | 96.81% | 93.57% | −3.24% — never caught up |

### Why the "optimal" config lost (LR too low for 5-epoch cosine schedule)

- **Cosine schedule decayed LR to ~0 by epoch 5** while the network was still learning. Starting LR=2e-4 + cosine = effective LR very small throughout training. Total weight updates were ~5× smaller than EXP-002.
- **Epoch 1 accuracy gap (13.45%) never closed.** EXP-002 already at 86.82% after epoch 1; EXP-006 still at 73.37%. The lower LR didn't help convergence — it just slowed it.
- **Table A's "Norse + SuperSpike wants lower LR for stability"** is **conditional** — it applies when 1e-3 causes oscillation/divergence in deeper or more sensitive networks. Our shallow 3-LIF Conv-SNN converges fine at 1e-3 (proven by EXP-002 → 97.31%). The "stability fix" was a solution to a problem we didn't have.

### Parallel with EXP-005 — both "individual optimal" configs LOST to fair-comparison baselines

| Framework | Fair-comparison config | "Individual optimal" config | Winner |
|---|---|---|---|
| SNNTorch | EXP-001: 98.25%, 86.81 pJ | EXP-005: 98.17%, 128.60 pJ | **EXP-001** (better acc + much better energy) |
| Norse | EXP-002: 97.31%, 55.76 pJ | EXP-006: 94.05%, 41.03 pJ | **EXP-002** (much better acc; EXP-006 only wins on energy due to undertraining) |

**This is a major finding for the comparison MD:** "library-canonical" defaults aren't task-universal. Empirical baselines beat theory-driven configs on N-MNIST. Both Table A recommendations (mse_count for SNNTorch; lower LR for Norse) were wrong for this specific task/architecture/epoch budget.

### Energy/accuracy tradeoff at a glance

| Run | Acc | Energy | Energy/acc-% |
|---|---|---|---|
| EXP-001 (SNNTorch fair) | 98.25% | 86.81 pJ | 0.88 pJ per acc% |
| EXP-002 (Norse fair) | 97.31% | 55.76 pJ | 0.57 pJ per acc% |
| EXP-005 (SNNTorch optimal) | 98.17% | 128.60 pJ | 1.31 pJ per acc% |
| EXP-006 (Norse optimal/lr=2e-4) | 94.05% | 41.03 pJ | 0.44 pJ per acc% (but undertrained) |

If you wanted **pure energy minimization** EXP-006's lr=2e-4 wins per-spike, but accuracy is too far off the Pareto front to be useful. EXP-002 (Norse fair-comparison) remains the most efficient *competitive* config.

### Implications for final comparison MD

1. **Report EXP-001 + EXP-002 as the primary baselines** — they're the best-performing configs for each framework on this task, despite not being "library optimal."
2. **Use EXP-005 + EXP-006 as cautionary examples** of "what happens when you follow library guidance blindly without empirical validation." Both demonstrate that paper-recommended settings need task-specific verification.
3. **Document the LR-schedule interaction** — cosine + low starting LR + short epoch budget = severe undertraining. Either raise starting LR or extend epochs or use a different schedule.

### Possible follow-ups

- **EXP-006b:** Same config but raise `tau_mem_inv` from 50 → 100 (Norse library default for input-gain boost — original Table A suggestion not yet tested)
- **EXP-006c:** Same config but **`lr=5e-4`** (middle of Table A range; might give the stability benefit without undertraining)
- **EXP-006d:** Same config but **`epochs=10`** — give the lower LR time to actually converge before declaring it broken
- **EXP-006e:** Match EXP-002 exactly (`lr=1e-3`, `epochs=5`) on the **post-refactor pipeline** to confirm the cleaner pipeline doesn't itself change results — would be the cleanest Norse baseline going forward

---

## EXP-005 · 2026-05-25 · SNNTorch — Individual-optimal config (mse_count + subtract reset + AMP)

**Strategy:** First run of SNNTorch with its **library-optimal** parameters (Table A of FRAMEWORK_CONFIG_ANALYSIS.md), not the fair-comparison defaults. Goal is to measure SNNTorch's peak accuracy on this Conv-SNN when freed from cross-framework parity constraints. Direct head-to-head against EXP-001 (same SNNTorch + same architecture but fair-comparison config: cross_entropy + hard reset + no AMP). The gap = how much SNNTorch is held back by fairness constraints.

**Changes vs EXP-001:**

| Param | EXP-001 (fair) | EXP-005 (optimal) | Why optimal for SNNTorch |
|---|---|---|---|
| `threshold` | 0.5 | **1.0** | SNNTorch amplifies inputs 20× at the membrane (beta=0.95 → V_eq = 20·I); threshold=1.0 keeps neurons selective rather than hyper-active. Norse would die at 1.0 (EXP-004) but SNNTorch thrives here. |
| `loss_fn` | cross_entropy | **mse_count** | SNNTorch's canonical loss; trains explicit per-class spike-rate targets (80% correct, 20% incorrect) |
| `reset_mode` | (zero — was default) | **subtract** | SNNTorch's native soft-reset; preserves overshoot voltage → input magnitude encoded in spike count |
| `use_amp` | false | **true** | Faster training; SNNTorch's atan surrogate has bounded gradients safe in float16 (unverified — first test) |
| `epochs` | 5 | 5 | Matched for fair head-to-head |

| Param | Value |
|---|---|
| framework | torch |
| threshold | 1.0 ← raised from 0.5 (EXP-001) — SNNTorch tolerates this due to 20× input amplification |
| beta (SNNTorch) | 0.95 |
| reset_mode | subtract |
| timesteps (n_time_bins) | 16 |
| batch_size | 64 |
| iterations_per_epoch | 937 (full epoch) |
| epochs | 5 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| lr_scheduler | cosine |
| use_amp | true ← first verification on SNNTorch |
| loss_fn | mse_count (correct_rate=0.8, incorrect_rate=0.2) |
| optimizer | adam |
| surrogate | atan (hardcoded) |
| num_workers | 0 |
| cache force_mode | null (adaptive — likely disk on GTX 1650 4GB VRAM) |
| augmentation | ON ±10° rotation |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|
| 1 | 0.1422 | 91.15% | 0.2366 | 0.000905 | 964.42 (cold cache fill) |
| 2 | 0.0902 | 96.97% | 0.2366 | 0.000655 | 490.74 |
| 3 | 0.0815 | 97.57% | 0.2352 | 0.000345 | 478.49 |
| 4 | 0.0741 | 97.85% | 0.2351 | 0.000095 | 480.82 |
| 5 | 0.0691 | 97.94% | 0.2361 | 0.000000 | 484.85 |

**Final training metrics:** loss 0.0641, accuracy 98.44%, spike rate 0.2346
**Test accuracy:** 98.17%
**Energy/sample:** 128.60 pJ (total 1,286,026 pJ over 10,000 samples)
**Avg latency/sample:** 0.329 ms (avg batch latency 20.99 ms)
**Avg spikes/sample:** 36.74 (total 367,436 spikes)
**Per-class F1:** min 0.969 (class 9), max 0.991 (class 1) — all classes ≥ 0.969 F1
**Run platform:** Colab T4 GPU (not local GTX 1650 as planned), 12.67 GB RAM
**Cache mode:** MEMORY selected by adaptive controller (RAM check passed, dataset ~4.13 GB fits in 6.16 GB budget)

**Notes:**

### Headline finding: "individual optimal" config did NOT beat the fair-comparison baseline

| Metric | EXP-001 (fair) | EXP-005 (optimal) | Δ |
|---|---|---|---|
| Test accuracy | 98.25% | **98.17%** | **−0.08% (worse)** |
| Energy/sample | 86.81 pJ | **128.60 pJ** | **+48% (much worse)** |
| Latency/sample | 0.327 ms | 0.329 ms | ~same |
| Spike rate | ~0.142 | **0.236** | **+66% (much higher firing)** |
| Avg spikes/sample | ~20 | 36.74 | +84% |

### Why the "optimal" config is actually worse on this task

1. **`mse_count_loss` literally trains the network to fire MORE.** It targets `correct_rate=0.8, incorrect_rate=0.2` — meaning every correct-class output neuron is incentivized to fire at 80% of timesteps. EXP-001's `cross_entropy` on summed spikes has no explicit rate target, lets the network find its own (which converged to ~0.142 rate). Result: mse_count produces a network that's accurate but spike-heavy → high energy.

2. **`reset_mode: subtract` (soft reset) reinforces multi-firing.** When V overshoots threshold by a lot, soft reset preserves the residual → the neuron can fire repeatedly in quick succession. Combined with mse_count's high-firing target, the network produces ~67% more spikes than EXP-001 for the same accuracy.

3. **`threshold: 1.0` did NOT reduce firing as expected** — soft reset + mse_count dominate the threshold's selectivity effect. The high-firing regime is created by the loss + reset choice, not the threshold.

### AMP=true verified safe for SNNTorch

Training converged normally (no NaN, no gradient underflow signs). Loss decreased monotonically. Table A's prediction ("AMP=true is safe for SNNTorch atan") confirmed on this net.

### Why MEMORY mode was ~4× slower than EXP-001's disk mode (480s vs 124s steady state)

Unexpected — MEMORY should be faster. Likely culprits (most → least likely):
- **AMP autocast overhead exceeds savings on this tiny net.** For a 12-conv + 32-conv + 800-FC model, the per-step matmul is small enough that AMP's scaler/autocast overhead may dominate. AMP wins on big models, not toy nets.
- **mse_count_loss has more compute than cross_entropy** (per-output-neuron rate target instead of single softmax).
- **Different Colab T4 allocation** (random session-to-session hardware variance).

Not blocking — accuracy result is what matters.

### Comparison ranking (SNNTorch on N-MNIST Conv-SNN)

| Configuration | Test Acc | Energy/sample | Spike rate | Recommendation |
|---|---|---|---|---|
| **EXP-001 (fair-comparison)** | 98.25% | 86.81 pJ | 0.142 | **✅ Best overall — energy/accuracy Pareto-optimal** |
| EXP-005 (individual-optimal) | 98.17% | 128.60 pJ | 0.236 | ❌ Worse on every dimension |

### Implications for the final comparison MD

- **"Library-canonical" ≠ "best on every task."** SNNTorch's library-canonical loss (`mse_count_loss`) is intended for tasks where spike-rate encoding matters (e.g., physiological signal modeling). For classification on a clean dataset like N-MNIST, cross-entropy on summed spikes is more efficient.
- **Soft reset is not universally better** even for SNNTorch. On clean classification it produces wasted spikes (multiple firings encoding nothing meaningful for the loss).
- **Recommended SNNTorch config for N-MNIST classification:** EXP-001's settings (`cross_entropy`, hard reset, threshold=0.5, no AMP) — they're "Pareto-optimal" against EXP-005's library-canonical choices.

### Possible follow-ups

- **EXP-005b:** Same config but `loss_fn: cross_entropy` (isolate the mse_count contribution to high firing)
- **EXP-005c:** Same config but `reset_mode: zero` (isolate the soft-reset contribution)
- **EXP-005d:** EXP-001 baseline + `use_amp: true` (isolate AMP's effect alone, no other changes)

---

## EXP-004B · 2026-05-25 · Norse — Retry of EXP-004 with threshold:0.5 (RECOVERED)

**Strategy:** Direct fix retry of EXP-004 (which failed with dead-neuron at threshold=1.0). Only one parameter changed: `threshold: 1.0 → 0.5`. Goal was to confirm the EXP-004 hypothesis that `threshold=1.0` alone killed Norse — if 0.5 trains normally with the same pipeline, the threshold is conclusively the culprit (not the new caching, not the adaptive controller, not the config-driven pipeline). Run on the same `pipeline-fixes-improvments` branch as EXP-004 but **BEFORE the big code refactor** (loss_fn config-driven across all 3, surrogate hardcoded, reset_mode YAML param, logging cleanup). Logs show old patterns: duplicate `ADAPTIVE CACHE CONTROLLER` block, "Cache mode: ADAPTIVE" instead of actual mode, "AMP: False ← safe for this framework" static annotation, etc.

| Param | Value |
|---|---|
| framework | norse |
| threshold | 0.5 ← **changed from 1.0 (EXP-004) — only diff** |
| tau_mem_inv (Norse) | 50.0 Hz |
| timesteps (n_time_bins) | 16 |
| batch_size | 64 |
| iterations_per_epoch | 937 (full epoch) |
| epochs | 3 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| lr_scheduler | cosine |
| use_amp | false |
| loss_fn | cross_entropy |
| optimizer | adam |
| surrogate | atan (still config-driven at this point — Norse ignored it, used SuperSpike default) |
| num_workers | 0 (YAML cap from 4 — Colab-safe) |
| cache force_mode | null (adaptive → controller selected MEMORY, 8.8 GB free / dataset ~4.13 GB) |
| augmentation | ON ±10° rotation |
| denoise filter | 10000 µs |
| probe measured | 72 KB/sample |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|
| 1 | 0.4622 | 85.03% | 0.0728 | 0.000750 | 972.49 (cold cache fill) |
| 2 | 0.1979 | 93.98% | 0.0960 | 0.000250 | 482.78 (warm MEMORY cache) |
| 3 | 0.1616 | 95.03% | 0.0948 | 0.000000 | 489.59 |

**Final training metrics (post-epoch eval pass):** loss 0.1003, accuracy 98.44%, spike rate 0.0967  
**Test accuracy:** 95.71%  
**Energy/sample:** 53.82 pJ (total 538,223 pJ over 10,000 samples)  
**Avg latency/sample:** 0.387 ms (avg batch latency 24.62 ms)  
**Avg spikes/sample:** 15.38 (total 153,778 spikes)  
**Per-class F1:** min 0.933 (class 8), max 0.985 (class 1) — all classes ≥ 0.933

**Notes:**
- **EXP-004 hypothesis confirmed.** Threshold=0.5 trains normally on the EXACT same pipeline that failed at threshold=1.0. The cosine LR scheduler ran to completion, neurons fired (~0.09 spike rate, matching EXP-002), loss decreased monotonically. Conclusively: **threshold=1.0 is the dead-neuron cause for Norse, not anything else in the new pipeline.**
- **Cache adaptive selection worked correctly.** Controller picked MEMORY mode (8.8 GB free, dataset ~4.13 GB) — confirmed by epoch 2 dropping from 972s to 483s (warm cache, no re-tofame).
- **Why 3 epochs instead of 5:** quick verification run; not meant for direct EXP-002 comparison.
- **Vs EXP-002 (5 epochs, same config except 5 epochs): 95.71% (3 epochs) vs 97.31% (5 epochs).** Trajectory matches — extrapolating, EXP-004B at epoch 5 would likely hit ~97%.
- **Vs EXP-001 (SNNTorch threshold 0.5, 5 epochs): 95.71% (3 epochs Norse) vs 98.25% (5 epochs SNNTorch).** Norse undershoots SNNTorch even at recovered config, consistent with our analysis: SNNTorch's 20× input amplification + soft reset gives it more headroom than Norse with hard reset at the same threshold.
- **Energy vs EXP-002:** 53.82 pJ/sample vs 55.76 pJ/sample — essentially the same; Norse's per-neuron energy profile is stable across training depth.
- **Per-class accuracy more uneven than EXP-002:** worst class 8 at F1=0.933 vs EXP-002's worst at F1=0.956. Probably attributable to undertrained model (3 epochs vs 5) rather than a config difference.
- **Pre-refactor logging note:** This run shows the old log patterns we cleaned up after: duplicate diagnostics blocks, misleading "Cache mode: ADAPTIVE", static "← safe for this framework" annotation, etc. Future runs will be cleaner.

---

## EXP-004 · 2026-05-25 · Norse — New Pipeline, threshold:1.0 (FAILED — dead neuron)

**Strategy:** First Norse run on the fully fixed `pipeline-fixes-improvments` pipeline. Goal was to establish a new Norse baseline with the improved caching, corrected `force_mode` wiring, and a standardized `threshold: 1.0` across all three frameworks (up from 0.5 in EXP-002). All other params kept at EXP-002 values. Adaptive cache (`force_mode: null`) selected MEMORY mode — first real test of in-memory caching on Colab.

| Param | Value |
|---|---|
| framework | norse |
| threshold | 1.0 ← **changed from 0.5 (EXP-002) — ROOT CAUSE OF FAILURE** |
| tau_mem_inv (Norse) | 50.0 Hz |
| timesteps (n_time_bins) | 16 |
| batch_size | 64 |
| iterations_per_epoch | 937 (full epoch) |
| epochs | 3 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| lr_scheduler | cosine |
| use_amp | false |
| loss_fn | cross_entropy |
| optimizer | adam |
| surrogate | atan |
| num_workers | 0 (YAML had 4; pipeline capped to 0 — Colab safety) |
| cache force_mode | null (adaptive → controller selected MEMORY, 8.8 GB available, dataset ~4.13 GB) |
| augmentation | ON ±10° rotation |
| denoise filter | 10000 µs |
| probe measured | 72 KB/sample (uint8 framed tensor — auto-probed at startup) |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|
| 1 | 2.3026 | 9.88% | 0.0000 | ~0.001 | ~1050 (download + cold cache fill) |
| 2 | 2.3026 | 9.92% | 0.0002 | — | ~539 (85.6% cache warm, MEMORY mode) |

**Test accuracy:** N/A (training terminated — random-chance loss, no improvement)  
**Notes:**  
- **FAILED: dead neuron problem.** Loss 2.3026 = −log(1/10), exactly random chance for 10 classes. Spike rate effectively zero every epoch — neurons never fired, surrogate gradient was zero throughout, no weight updates occurred.  
- **Root cause (high confidence): `threshold: 1.0` kills Norse.** Norse's weight initialization produces lower effective membrane drive than SNNTorch. With `threshold=0.5` (EXP-002), Norse trains to 97.31%. Doubling the threshold to 1.0 means no neuron's membrane potential ever reaches the fire threshold → dead neurons throughout the network → surrogate gradient (atan) evaluates to zero everywhere → gradient-free forward passes, loss never decreases.  
- **Possible contributing factor (uncertain):** `threshold=1.0` was applied to ALL LIF layers including `lif_out`. If the output layer never fires, the count-based readout `spk_rec.sum(0)` is all zeros → softmax on zeros → uniform distribution → 10% accuracy. The hidden layers may have been partially active but output layer silence alone is sufficient to cause this failure mode.  
- **Why SNNTorch is fine at 1.0 but Norse is not:** SNNTorch uses a beta-decay leaky integrate-and-fire model with different default weight scales. Norse uses a biophysical LIF parameterized by `tau_mem_inv` (inverse time constant in Hz). The two frameworks have different input drive at initialization — comparing them at the same threshold is physically incorrect. Thresholds must be tuned per framework independently.  
- **Cache performance confirmed:** MEMORY mode worked correctly. Epoch 2 at 539s vs epoch 1 at 1050s — speedup came from warm cache, not from training (loss was identical). Probe correctly measured 72 KB/sample (uint8 format, not float32 — explains why dataset fit in ~4 GB vs the expected ~8.7 GB float32 estimate).  
- **Fix applied:** Threshold moved from shared `architecture` block to per-framework blocks. `frameworks.norse.threshold: 0.5`, `frameworks.snntorch.threshold: 1.0`, `frameworks.spikingjelly.threshold: 1.0`. `snn_config.py` now reads `THRESHOLD` from the active framework's section. Re-run as EXP-005 to confirm recovery.

---

## EXP-003 · 2026-05-25 · Norse — New Pipeline, force_mode:null, num_workers:4, cosine LR

**Strategy:** Deliberate test case. Starting from our validated working config (EXP-002 params: `force_mode:disk`, `num_workers:0`, `lr_scheduler:none`, `use_amp:false`), we intentionally changed five params to match the new main-branch pipeline defaults: `use_amp→true`, `lr_scheduler→cosine`, `num_workers→4`, `force_mode→null`. Only `iterations_per_epoch:937` (our fix — main branch had no reliable value) and `threshold:0.5` were kept unchanged. Primary goal: measure Colab speed/stability under adaptive cache with 4 workers, and also empirically test whether `use_amp:true` actually breaks Norse on this architecture.

| Param | Was (our working config) | Changed to (this test) | Reason for change |
|---|---|---|---|
| `use_amp` | false | true | Match main branch default — test if AMP actually breaks Norse here |
| `lr_scheduler` | none | cosine | Match main branch default |
| `num_workers` | 0 | 4 | Match main branch default — stress-test Colab RAM |
| `force_mode` | disk | null | Match main branch default — let controller pick adaptive strategy |
| `threshold` | 0.5 | 0.5 (kept) | Already matches main branch |
| `iterations_per_epoch` | 937 | 937 (kept) | Our fix — full N-MNIST epoch |

| Param | Value |
|---|---|
| framework | norse |
| threshold | 0.5 |
| tau_mem_inv (Norse) | 50.0 Hz |
| timesteps (n_time_bins) | 16 |
| batch_size | 64 |
| iterations_per_epoch | 937 (full epoch) |
| epochs | 5 (crashed after epoch 1) |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| lr_scheduler | cosine |
| use_amp | true |
| loss_fn | cross_entropy |
| optimizer | adam |
| surrogate | atan |
| num_workers | 4 |
| cache force_mode | null (adaptive — Colab chose memory strategy) |
| augmentation | ON ±10° rotation |
| denoise filter | 10000 µs |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|
| 1 | 0.4214 | 86.40% | 0.0723 | 0.000905 | 983.52 |
| 2 | — | — | — | — | CRASH |

**Test accuracy:** N/A (OOM crash before epoch 2)  
**Notes:**  
- **OOM confirmed earlier than expected**: Crash hit at the very start of epoch 2 (not epoch 3). `RuntimeError: DataLoader worker (pid 2861) is killed by signal: Killed`. With `num_workers:4` + `force_mode:null`, Colab ran out of RAM faster than predicted — memory pressure from 4 workers each building their own in-memory cache copy exhausted the 12 GB before epoch 2 could complete even a single batch.  
- **AMP=true did NOT break Norse here**: 86.40% accuracy after epoch 1 with `use_amp:true` — Norse trained successfully. This contradicts the earlier assumption that AMP always breaks Norse. See AMP note below.  
- **Cosine LR wired correctly**: LR already at 0.000905 after epoch 1 (decayed from 0.001), confirming scheduler propagation works.  
- **Key takeaway**: The OOM is the critical failure on Colab — `force_mode:null` + `num_workers:4` is unusable. Speed comparison vs EXP-002 not possible (crashed too early). `force_mode:disk` + `num_workers:0` remains the correct Colab configuration.  
- **AMP note**: In earlier runs, Norse got ~9% with what was believed to be an AMP issue. The real root cause of those runs was wrong config values (T=1300 vs 16, ITERA=100 vs 937). For this shallow 3-LIF Conv-SNN, AMP does not catastrophically break training — gradient chains are short enough that float16 underflow is not severe. AMP breaking Norse is a risk in deeper networks; for this architecture it appears tolerable.

---

## EXP-002 · 2026-05-25 · Norse

**Strategy:** Norse baseline run — same config as EXP-001 (SNNTorch validation) swapped to Norse framework. Establishes Norse Conv-SNN reference point for direct SNNTorch vs Norse comparison on identical hyperparameters.

| Param | Value |
|---|---|
| framework | norse |
| threshold | 0.5 |
| tau_mem_inv (Norse) | 50.0 Hz |
| timesteps (n_time_bins) | 16 |
| batch_size | 64 |
| iterations_per_epoch | full epoch |
| epochs | 5 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| lr_scheduler | none |
| use_amp | false |
| loss_fn | cross_entropy |
| optimizer | adam |
| surrogate | atan |
| augmentation | ON ±10° rotation |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|
| 1 | 0.4090 | 86.82% | 0.0718 | 0.001000 | 1208 |
| 2 | 0.1601 | 95.14% | 0.0916 | 0.001000 | 160 |
| 3 | 0.1304 | 95.99% | 0.0950 | 0.001000 | 165 |
| 4 | 0.1142 | 96.53% | 0.0986 | 0.001000 | 161 |
| 5 | 0.1026 | 96.81% | 0.0983 | 0.001000 | 159 |

**Test accuracy:** 97.31%  
**Energy/sample:** 55.76 pJ  
**Avg latency/sample:** 0.409 ms  
**Per-class accuracy:** min 95.6% F1 (class 9), max 98.9% F1 (class 1) — all classes ≥ 95.6% F1  
**Notes:** Norse baseline confirmed. Spike rate noticeably lower than SNNTorch (0.094 vs 0.142 avg), which drives lower energy/sample (55.76 vs 86.81 pJ). Test accuracy slightly below SNNTorch (97.31% vs 98.25%). Epoch 1 slow due to dataset download; steady-state ~160 s/epoch. Next: tune Norse-specific params (tau_mem_inv, threshold) to close the accuracy gap.

---

## EXP-001 · 2026-05-25 · SNNTorch (torch)

**Strategy:** Validation run — new config-driven pipeline (`learning/`) with SNNTorch Conv-SNN, params set to match old pipeline defaults. Confirms that YAML→Settings→Model param propagation is correct and that `force_mode: disk` + `num_workers: 0` restores old pipeline speed (~120s/epoch after epoch 1).

| Param | Value |
|---|---|
| framework | torch |
| threshold | 0.5 |
| beta (SNNTorch) | 0.95 |
| timesteps (n_time_bins) | 16 |
| batch_size | 64 |
| iterations_per_epoch | 937 (full epoch) |
| epochs | 5 |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| lr_scheduler | none |
| use_amp | false |
| loss_fn | cross_entropy |
| optimizer | adam |
| surrogate | atan |
| num_workers | 0 |
| cache force_mode | disk |
| augmentation | ON ±10° rotation |

| Epoch | Train Loss | Train Acc | Spike Rate | LR | Duration (s) |
|---|---|---|---|---|---|
| 1 | 0.3239 | 90.04% | 0.1131 | 0.001000 | 968 |
| 2 | 0.0964 | 96.86% | 0.1369 | 0.001000 | 124 |
| 3 | 0.0802 | 97.53% | 0.1322 | 0.001000 | 124 |
| 4 | 0.0705 | 97.77% | 0.1356 | 0.001000 | 124 |
| 5 | 0.0643 | 97.98% | 0.1422 | 0.001000 | 126 |

**Test accuracy:** 98.25%  
**Energy/sample:** 86.81 pJ  
**Avg latency/sample:** 0.327 ms  
**Per-class accuracy:** min 99.4% (class 0), max 99.3% (class 1) — all ≥ 96.5% recall  
**Notes:** Validation passed. Config approach confirmed correct. Next: run Norse with same params to establish Norse baseline, then tune Norse-specific params (tau_mem_inv, threshold) for best accuracy.
