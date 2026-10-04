# Experiment 8 — Depth ladder (Factor B)

**Study:** Boundaries of Emergence — Identifying Capacity Thresholds in Small-Scale SNNs
**Position:** after ex7 (width ladder), before ex9 (combine + stress).
**Dataset:** N-Caltech101 · **Backends:** all four (`sj`, `torch`, `norse`, `sinabs`)
· **Rungs:** 4 (d0, d1, d2, d4) · **Width:** fixed at f12 · **Epochs:** 15 · **Seeds:** 1 per
rung, 3 on the winning rung (SpikingJelly)

Design decided 2026-10-04. Master plan: `scalability_tests/experiment_plan_final.md` §6
("ex8 background" and "ex8 design") and §9.

---

## 1. What this experiment decides

**The Depth Ceiling:** the number of hidden FC layers past which adding another one
stops paying off. Measured at fixed width (f12), on all four frameworks.

- **Width** = how much the network sees at one step (ex7, filters).
- **Depth** = how many steps of combining it gets to do (ex8, hidden layers).

## 2. What is fixed, what varies

| | |
|---|---|
| **Fixed** | width f12 (`conv1_out = conv2_out = 12`), N-Caltech101, T=16, optimizer + LR (constant, no scheduler), BPTT surrogate gradient, 15 epochs, seed 0, 1 worker, AMP off |
| **Varied** | `fc_hidden.layers` only — 0, 1, 2, 4. Hidden size fixed at **128** |
| **Frameworks** | all four, every rung |

Everything fixed is **inherited** from `../ex7/config.yaml` (`extends:`), not copied —
ex7 and ex8 cannot drift apart.

## 3. The network

At f12 on the 180×240 sensor:

| layer | output shape | parameters |
|---|---|---|
| **input** (one timestep of events) | 2 × 180 × 240 = 86,400 | – |
| Conv1 (2→12, k5) + LIF `lif1` | 12 × 176 × 236 | 612 |
| Pool 2 | 12 × 88 × 118 | 0 |
| Conv2 (12→12, k5) + LIF `lif2` | 12 × 84 × 114 | 3,612 |
| Pool 2 | 12 × 42 × 57 | 0 |
| **Flatten** | **28,728** | 0 |
| *hidden:* [Linear(→128) + LIF] × N | 128 each | 3.68M for the first, 16.5k each after |
| **Linear (→101) + LIF `lif_out`** | **101** | 2.90M at d0, 13k at d1+ |

- Hidden LIFs are named `lif3`, `lif4`, … (output is always `lif_out`).
- Their neuron type comes from `neuron_types.<fw>.lif_hidden` in
  `configuration/network_architecture.yaml`.

### How the network code was extended for depth (2026-10-04)

- **Config:** a new `fc_hidden: {layers, size}` section in
  `configuration/network_architecture.yaml`.
- **Builder:** `frameworks/spiking_net.py` inserts N × `Linear → LIF` blocks between
  `Flatten` and the classifier:

  ```
  Flatten → [Linear(128) → LIF] × N → Linear(101) → lif_out
  ```

- **`layers: 0` is the default** and builds exactly the old network, so ex6 and ex7 are
  unaffected.
- **Naming:** hidden LIF layers are `lif3`, `lif4`, …; the output stays `lif_out`.
- **Neuron type:** a new `lif_hidden` entry per framework under `neuron_types`.
- **Nothing else needed changing:** capacity metrics, gradient norms, SynOps and the
  `fc_hidden_*` results columns already handle any depth.
- **Tests:** `tests/unit_layer_naming.py` — depth 0 builds the old network; depths 1/2/4
  have the right layers, names and metric wiring; a depth-2 net runs forward+backward on
  all four frameworks; a missing `lif_hidden` key fails loudly.

### Why the hidden size is 128

"Half or a quarter of the input" doesn't work here: half of 28,728 is 14,364 neurons ≈
412M parameters for one layer. Instead:

1. **≥ the number of classes (101)** — narrower would itself be a bottleneck.
2. **A power of 2** — convention, GPU-efficient.
3. **As small as rule 1 allows** — ~7k samples, so a bigger layer only adds overfitting
   risk.

## 4. The rungs

| rung | file | hidden layers | total parameters |
|---|---|---|---|
| d0 | `d0.yaml` | 0 (baseline) | 2.91M |
| d1 | `d1.yaml` | 1 × 128 | 3.69M |
| d2 | `d2.yaml` | 2 × 128 | 3.71M |
| d4 | `d4.yaml` | 4 × 128 | 3.74M |

4 rungs × 4 frameworks = **16 runs**. d0 is not reused from ex7 — f12 was not an ex7 rung.

## 5. Stopping mechanism — the Depth Ceiling

- **Rule:** <1% accuracy gain for >10% more time or parameters → Depth Ceiling.
- **Applied afterwards**, not as an abort. Every rung runs.
- **Gradient norms are a diagnostic, not a trigger** (plan §9).
- **Caveat:** the "params" half fires only once. d0→d1 adds +27% parameters; every
  layer after adds ~0.5%. From d1 on, the real cost of depth is **training difficulty**
  (fading gradients, slower convergence, overfitting), not hardware.

## 6. What to expect, in theory

Working hypotheses — **not acceptance criteria**. A different outcome is still a result.

- **Accuracy:** small gain d0→d1, flat at d2, likely *worse* at d4 (surrogate-gradient
  error compounds; deeper nets converge slower within a fixed 15 epochs — state this
  confound in the report).
- **Gradient norms:** conv1/conv2 norms shrink as depth grows — the vanishing-gradient
  signature.
- **Firing:** deep hidden layers may go quiet (near-dead) or saturate.
- **Capacity metrics:** a hidden layer's PR is capped at 128; I(Z;Y) should rise layer by
  layer toward the output.
- **Train − test gap:** expect a jump at d1 (+0.8M parameters).
- **Frameworks:** may tolerate depth differently — itself a result.

## 7. What gets recorded, and where

| metric | where |
|---|---|
| test accuracy, train time, VRAM, batch size | `results/runs.csv` |
| `fc_hidden_layers`, `fc_hidden_size`, `fc_hidden_params`, `classifier_params` | `results/runs.csv` (measured off the built model) |
| train accuracy / loss per epoch | `results/epochs.csv` |
| per-layer PR, entropy, I(Z;Y), firing rate, **gradient norms** — incl. every hidden layer | `results/layers.csv` |
| SynOps energy | each run's own `training_results.csv` (not in the cross-run schema — plan §10 item 6) |

**Train − test gap** = last `train_accuracy_pct` in `epochs.csv` minus
`test_accuracy_pct` in `runs.csv`.

## 8. Seeds

Seed 0 for every rung. Once the winning rung is known, re-run it at seeds 1 and 2 on
**SpikingJelly only** (plan §9).

## 9. How to run it

CPU sanity checks first (seconds each):

```bash
python check_env.py
python check_network.py --config experiments/ex8/d4.yaml --all
python tests/run_all.py layer_naming
```

Then the 16 runs, each its own process:

```bash
for FW in sj torch norse sinabs; do   # "torch" = snnTorch
  for RUNG in d0 d1 d2 d4; do
    python learning/main.py --config experiments/ex8/$RUNG.yaml --experiment ex8 \
        --framework $FW --seed 0 --inference stats \
        --results-root <results path> --cache-root <cache path>
  done
done
```

Run them one per session, not as one loop. On Kaggle:

- **Save & Run All (Commit)**, not an interactive session.
- Results are written **only after training and inference both finish** — an interrupted
  run loses everything.
- Point `--cache-root` away from `/kaggle/working` (cache ~18GB).
- Same GPU model for every run (ex7 used Tesla T4).

**Time budget:** ex7's f8/f16 ran at ~180–230 s/epoch on a T4; f12 sits between and the
hidden layers are cheap, so expect **~1 h per run, ~16–18 GPU-hours total**.

Collect and plot (no GPU):

```bash
python collect_results.py --from <folder> --experiment ex8
python make_plots.py --experiment ex8
```

## 10. Caveats to state in the report

- **Width is fixed at f12, not an ex7 rung.** ex7's best accuracy was f8; f12 was chosen
  between f8 and f16 (the pipeline's original 12-filter size).
- **15 epochs is not converged** (same as ex7) — depth and convergence speed can mix.
- **Single seed per rung** — small differences between rungs may be seed noise.
- **LR fixed across rungs** — not tuned per depth.
- **Classifier dominance (ex7 §3) moves**: at d0 the classifier holds ~99.9% of
  parameters; at d1+ the *first hidden layer* holds ~99.5% instead.

## 11. Results

| rung | sj | torch | norse | sinabs |
|---|---|---|---|---|
| d0 | _pending_ | _pending_ | _pending_ | _pending_ |
| d1 | _pending_ | _pending_ | _pending_ | _pending_ |
| d2 | _pending_ | _pending_ | _pending_ | _pending_ |
| d4 | _pending_ | _pending_ | _pending_ | _pending_ |

**Depth Ceiling:** _pending_
