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
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Test accuracy:**  
**Energy/sample:**  
**Avg latency/sample:**  
**Notes:**  
- **Baseline to beat:** EXP-001 = 98.25% test acc, 86.81 pJ/sample, 0.327 ms/sample (fair-comparison SNNTorch)
- **Hypothesis:** mse_count + soft reset should give SNNTorch a small edge over cross_entropy + hard reset (a few tenths of a percent) — large gap would indicate the fair-comparison constraints were significantly limiting.
- **AMP verification:** If accuracy regresses noticeably vs EXP-001, suspect AMP first — disable and re-run as EXP-005b.
- **Hardware note:** Running on GTX 1650 (4 GB VRAM) + 15.7 GB RAM; expect adaptive cache to select disk mode (GPU pressure threshold or RAM constraint).

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
