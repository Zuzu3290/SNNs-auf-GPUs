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
