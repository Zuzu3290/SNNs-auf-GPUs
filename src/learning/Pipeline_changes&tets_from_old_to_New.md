# SNN Pipeline Design Notes
N-MNIST · Conv-SNN · SNNTorch / Norse / SpikingJelly
Branch: `pipeline-fixes-improvments`

---

## 1. Old Pipeline (reference — `src/old_pipeline` on `18-norse` branch)

### What it was
A working Conv-SNN pipeline for N-MNIST, hardcoded constants throughout, no YAML config wiring for model/training params. Norse only.

### Key design choices
| Aspect | Old Pipeline |
|---|---|
| Config | Hardcoded constants inside `_NorseNet` (IN_CHANNELS=2, CONV1_OUT=12, FC_IN=800, etc.) |
| Framework param | `tau_mem_inv` hardcoded to 50.0 Hz in `snn_norse.py`, no YAML |
| tau conversion | `tau_mem_inv = -math.log(beta)/dt` from beta=0.95 — wrong physics but landed near 50.0 Hz |
| Framing | `ToFrame` baked into N-MNIST dataset constructor before `DiskCachedDataset` → disk stored **framed tensors** → epoch 2+ ToFrame ran zero times |
| Timesteps | `n_time_bins=16` (fixed T, no padding needed) |
| Iterations | 937/epoch = full 60k N-MNIST epoch at batch_size=64 |
| AMP | Off (not implemented) |
| LR scheduler | None |
| Workers | Single-process (no multi-worker) |
| Cache | `DiskCachedDataset` — framed tensors on disk → epoch 1 slow (~1200s), epoch 2+ fast (~160s) |
| Loss | `F.cross_entropy(spk_rec.sum(0), targets)` — no `.float()` cast (worked because dtype matched) |
| Framework selection | Hardcoded to Norse |

### Old pipeline results (EXP-002)
- Epoch 1: 86.82%, 1208s (cold disk)
- Epochs 2–5: 95–96.8%, ~160s/epoch
- Test accuracy: **97.31%** | Energy: 55.76 pJ | Spike rate avg: ~0.095

---

## 2. New Pipeline — Main Branch (unaltered, before our fixes)

### What changed vs old pipeline
| Aspect | Main Branch (new, unaltered) |
|---|---|
| Framing | `time_window=15000µs` → variable T per recording → needs `PadTensors` collate |
| data_workflow.yaml | Exists but **never read** by `data_pipeline.py` — completely decorative |
| Cache design | `DiskCachedDataset(raw_events, transform=full_train_tf)` → disk stores **raw events** → ToFrame runs every access, every epoch → epoch 2+ NOT faster |
| AMP | `use_amp: true` default |
| LR scheduler | `cosine` default |
| Workers | `num_workers=4` default, `persistent_workers=True` |
| Cache strategy | `force_mode: null` → adaptive (probes Colab RAM → picks memory → OOM at epoch 2-3) |
| Framework selection | `SNN_NORSE(cfg)` hardcoded in `main.py` regardless of YAML |
| Norse loss | `F.cross_entropy(spk_rec.sum(0), targets)` — no `.float()` cast → dtype crash risk |
| Config file | Old MLP-style keys in YAML: `INPUT_SIZE`, `HIDDEN_SIZE`, etc. — no Conv-SNN architecture keys |
| YAML encoding | `open(yaml_path, "r")` — no `encoding="utf-8"` → Windows cp1252 crash on `×` symbol |
| `NUM_WORKERS` bug | Set twice in `snn_config.py` — line 31 from `architecture.num_workers`, overwritten at line 48 from `training.num_workers` |

### Key consequence
Because `data_workflow.yaml` was never read, all its settings (framing mode, cache config, augmentation) had zero effect on main branch. The pipeline was effectively hardwired at the Python level.

---

## 3. Our Config-Wired Pipeline (`pipeline-fixes-improvments` branch)

### What we fixed / added
| Fix | Detail |
|---|---|
| Conv-SNN YAML schema | New keys: `sensor_h/w`, `in_channels`, `conv1_out`, `conv1_kernel`, `conv2_out`, `conv2_kernel`, `pool_kernel`, `output_size`, `threshold` |
| `FC_IN` auto-computed | `_h = (SENSOR_H - CONV1_KERNEL + 1) // POOL_KERNEL; _h = (_h - CONV2_KERNEL + 1) // POOL_KERNEL; FC_IN = CONV2_OUT * _h * _h` — change conv/pool params, FC_IN updates automatically |
| Per-framework neuron params | `frameworks.snntorch.beta`, `frameworks.norse.tau_mem_inv`, `frameworks.spikingjelly.tau` — each framework reads its own block, no cross-conversion |
| YAML encoding | `open(yaml_path, "r", encoding="utf-8")` — Windows cp1252 fix |
| Framework selection | `_FRAMEWORKS[cfg.FRAMEWORK]` in `main.py` — one line in YAML switches everything |
| Norse `.float()` cast | `F.cross_entropy(spk_rec.float().sum(0), targets)` — prevents dtype crash |
| `_FramedDataset` wrapper | New top-level picklable class. Wraps raw dataset, applies `frame_tf` in `__getitem__`. DiskCachedDataset wraps `_FramedDataset` → **disk stores framed tensors** → epoch 2+ ToFrame runs zero times (matches old pipeline behavior) |
| `data_workflow.yaml` actually read | Framing mode, cache config, augmentation all driven from YAML |
| `cfg.NUM_WORKERS` wired into `create_loaders` | `PipelineMemoryCoordinator` result capped by `cfg.NUM_WORKERS`; if 0, removes `prefetch_factor` and sets `persistent_workers=False` |
| Framing mode | `n_time_bins=16` (fixed T, no padding) — matches old pipeline, dropped `time_window` variable-T approach |

---

## 4. Validation Test 1 — SNNTorch (EXP-001, 2026-05-25)

**Goal:** Confirm config wiring is correct — new pipeline + old pipeline params = old pipeline results.

### What we kept the same as old pipeline
- `threshold: 0.5`
- `lr_scheduler: none`
- `use_amp: false`
- `num_workers: 0`
- `force_mode: disk`
- `timesteps: 16` (n_time_bins)
- `batch_size: 64`
- `iterations_per_epoch: 937`

### What was different (pipeline improvement, not param change)
- Framework: SNNTorch (not Norse) — first validation used SNNTorch as it was simpler to validate
- Config now YAML-driven (not hardcoded)
- `_FramedDataset` wrapper → disk caches framed tensors (same as old pipeline end-result, different mechanism)
- `data_workflow.yaml` actually read

### Result
| Epoch | Train Acc | Duration |
|---|---|---|
| 1 | 90.04% | 968s |
| 2–5 | 96.86–97.98% | ~124s/epoch |

**Test accuracy: 98.25%** — matched old pipeline SNNTorch performance (~98.71%).  
**Conclusion: Config wiring validated.** YAML params propagate correctly end-to-end.

---

## 5. Validation Test 2 — Norse + Main-Branch Params (EXP-003, 2026-05-25)

**Goal:** Measure speed/stability impact of main-branch default settings on our new pipeline. Stress-test `force_mode:null` + `num_workers:4` on Colab.

### What we kept from our working config
- `threshold: 0.5`
- `use_amp: true` (kept true intentionally — to test if AMP actually breaks Norse, see §6)
- `timesteps: 16` (n_time_bins — our fix, NOT the `time_window` approach from main)
- `iterations_per_epoch: 937` (our fix — main branch had no reliable value here)
- `batch_size: 64`

### What we changed to match main-branch defaults
- `lr_scheduler: none → cosine`
- `num_workers: 0 → 4`
- `force_mode: disk → null` (adaptive — Colab chose memory strategy)

### Result
| Epoch | Train Acc | Duration |
|---|---|---|
| 1 | 86.40% | 983.52s |
| 2 | CRASH (OOM) | — |

**Crash:** `RuntimeError: DataLoader worker (pid 2861) killed by signal: Killed` — hit at the very start of epoch 2. Colab 12 GB RAM exhausted by 4 persistent workers each holding their own in-memory cache copy.

### What this proved
1. **`force_mode:null` + `num_workers:4` is unusable on Colab** — OOM kills before epoch 2 even starts. `force_mode:disk` + `num_workers:0` is the correct Colab config.
2. **Cosine LR scheduler wires correctly** — LR already at 0.000905 after epoch 1 (decayed from 0.001).
3. **AMP=true did NOT break Norse here** — 86.40% epoch 1 shows healthy learning (see §6).

---

## 6. AMP Discovery

### What we believed initially
`use_amp: true` always breaks Norse because:
- Norse LIF surrogate gradients produce small values (~1e-4 to 1e-2)
- `torch.autocast(float16)` casts to float16 where values below ~6e-5 underflow to zero
- `GradScaler` silently skips `optimizer.step()` when gradients contain inf/nan
- Result: model stuck at ~9% (random chance) with no error

### What EXP-003 showed
With `use_amp:true`, Norse reached **86.40% after epoch 1** — healthy learning, not 9%.

### Reconciliation
The earlier ~9% Norse runs were caused by **wrong config values** — `timesteps=1300` (should be 16) and `iterations_per_epoch=100` (should be 937). The model saw ~1/60th of the data it should have per epoch, and ran T=1300 timesteps of zero-padded noise. AMP was blamed but was not the actual cause.

### Why AMP is still off by default for Norse
For this shallow 3-LIF Conv-SNN, float16 underflow is not catastrophic — gradient chains are short. But the risk scales with network depth. In deeper SNNs (more LIF layers), chain-rule multiplication of small surrogate gradients will hit the float16 floor. **`use_amp:false` is the safe default for Norse** — it costs nothing on accuracy and eliminates a silent failure mode that produces no error and no obvious warning.

For SNNTorch and SpikingJelly: `use_amp:true` is safe (they have internal float32 casts; SNNTorch is AMP-tested by the library).

### Rule
- Norse: `use_amp: false` (safe default; can experiment in playground mode for shallow nets)
- SNNTorch / SpikingJelly: `use_amp: true` (fine)

---

## 7. Experiment Summary Table

| EXP | Framework | Approach | Test Acc | Epoch 2 Speed | Outcome |
|---|---|---|---|---|---|
| EXP-001 | SNNTorch | New pipeline + old params | 98.25% | ~124s | Config wiring validated |
| EXP-002 | Norse | Old pipeline (reference) | 97.31% | ~160s | Norse baseline established |
| EXP-003 | Norse | New pipeline + main defaults (null cache, 4 workers, cosine) | N/A (OOM) | CRASH | null+4workers unusable on Colab |
