# Learning Layer — Change Report
**Baseline:** `89f29525` (Spikingjelley working)  
**Head:** `6850566d` (Backpropagation implemented with STDP)  
**Commits spanned:** 6 (`abe753cc` → `6850566d`)

---

## Summary

Six commits added three major capabilities to the learning layer: **biologically richer neuron models**, a **framework-agnostic training infrastructure**, and **two new training objectives** (adversarial robustness via TRADES and biologically-inspired spike regularization via activity regularization and STDP). Every framework backend now speaks the same internal interface, and the evaluator covers both clean and adversarial accuracy.

---

## 1. New Files

### `src/learning/adversarial_robustness.py`
**What it adds:** Attack generation (FGSM, PGD) and an `AdversarialEvaluator` class that benchmarks the trained network against input perturbations.

| Function / Class | Role |
|---|---|
| `generate_fgsm_input()` | Single-step Fast Gradient Sign Method. Moves input one step in the direction of the gradient. |
| `generate_pgd_input()` | Iterative FGSM with epsilon-ball projection (Projected Gradient Descent). Stronger than FGSM. |
| `AdversarialEvaluator.evaluate()` | Runs clean, FGSM, and PGD sweeps across a configurable list of epsilon values, prints a table of accuracy drops, and saves a CSV. |

**Impact:** Post-training robustness measurement is now built-in. Each framework exposes it via `model.get_adversarial_evaluator(test_loader)`.

**Improves:** You can now quantify how much adversarial noise the network can tolerate without a separate script or external library.

---

### `src/learning/frameworks/activity_reg.py`
**What it adds:** Two biologically-inspired regularization losses and the hook infrastructure to collect hidden-layer spike recordings during the forward pass.

#### Hook infrastructure

| Function | Role |
|---|---|
| `register_activity_hooks(model, layer_map)` | Attaches `forward_hook` callbacks to named LIF layers. Spike tensors from every timestep accumulate in `model.hidden_spk_buf`. |
| `clear_hidden_spikes(model)` | Clears the buffers at the start of each forward pass. |
| `get_hidden_spike_recordings(model)` | Returns `{name: (T, B, ...)}` tensors, stacked over timesteps. |

The hooks handle framework differences automatically: Norse `LIFCell` returns `(spk, state)` tuples; the hook extracts `output[0]`.

#### Activity Regularization

Two-sided per-neuron penalty that detects dead neurons (firing rate < `min_rate`) and saturated neurons (firing rate > `max_rate`) and penalizes each independently.

```
penalty = lambda_low  * mean(relu(min_rate - rate)²)
        + lambda_high * mean(relu(rate - max_rate)²)
```

Rates are computed per-neuron (not collapsed to a global mean), so a handful of overactive neurons cannot mask a majority of silent ones.

**Improves:** Prevents the common SNN failure modes of dead (gradient vanishing) and saturated (always-on) neurons, which both prevent effective learning.

#### STDP Regularization

Spike-Timing Dependent Plasticity as a soft loss term, applied alongside BPTT. For each consecutive layer pair:

- Exponential pre/post-synaptic eligibility traces decay with time constant `tau`.
- LTP accumulates when the pre-synaptic trace is high and the post-synaptic neuron fires (causal order rewarded).
- LTD accumulates when the post-synaptic trace is high and the pre-synaptic neuron fires (anti-causal order penalized).
- `L_STDP = -A_plus * LTP + A_minus * LTD` is added to the task loss.

SpikingJelly's pre-summed `[B, C]` output is automatically excluded from the output layer pair so no framework-specific branching is needed in the loss.

**Improves:** Nudges the network toward temporally causal spike patterns without replacing gradient-based learning. This is a step toward biologically plausible temporal structure in learned representations.

---

## 2. Modified Files

### `src/learning/training.py`

#### `aggregate_spike_output()` — new utility function
Normalises any spike recording to `[B, C]` class logits:
- `[B, C]` (SpikingJelly pre-summed) → returned as-is.
- `[T, B, C]` (Norse, SNNTorch) → summed over the time axis.

**Impact:** The trainer no longer needs to know which framework produced the spikes. Every accuracy and loss computation routes through this function, making the training loop truly framework-agnostic.

#### TRADES adversarial training path
A second training branch is activated when `cfg.TRADES_ENABLED = True`:

1. A clean forward pass produces `clean_prob` (detached softmax).
2. `generate_trades_adversarial()` finds the worst-case perturbation within the epsilon-ball by maximising KL divergence from the clean prediction, using `torch.autograd.grad` so model parameter gradients are never accumulated in the inner loop.
3. Both clean and adversarial passes run under AMP.
4. Loss = `CrossEntropy(clean) + lambda * KL(clean || adversarial) + activity_penalty + stdp_penalty`.

Activity regularization and STDP run on both the standard and TRADES paths.

**Improves:** Adversarial training makes the network resistant to input perturbations at inference time, trading a small amount of clean accuracy for robustness that is preserved when data is corrupted or perturbed.

#### Firing rate in Hz
Spike rate (fraction of possible spikes) is converted to Hz using the temporal window duration from config:

```python
firing_rate_hz = train_spike * timesteps / window_s
```

This is logged per epoch to CSV and printed in the training console. The same conversion is applied in inference.

**Improves:** Raw spike rate fractions are unitless and hard to relate to biological neuron behavior. Hz gives a physically interpretable quantity that maps directly to energy and biological plausibility assessments.

#### Bug fixes
- `targets` cast to `.long()` before loss computation — required by `CrossEntropyLoss`.
- `plot_raster()` now handles both `[T, B, C]` and `[B, C]` spike shapes from the last batch.

---

### `src/learning/inference.py`

- Dropped `snntorch.functional.accuracy_rate` — replaced with `aggregate_spike_output` from `training.py`. The tester is no longer SNNTorch-specific.
- Added `firing_rate_hz` per batch (same Hz formula as training) to the batch log CSV and the summary printout.
- Added `avg_firing_rate_hz` to the returned dict.
- Fixed a division-by-zero guard in `_class_metrics` when `total == 0`.

**Improves:** Inference output is now consistent with training output (same Hz metric, same framework-agnostic spike aggregation).

---

### `src/learning/frameworks/snn_spikingjelly.py`

**Neuron model change:** `LIFNode` → `IzhikevichNode` in all three layers.

The Izhikevich model uses two coupled differential equations (membrane potential `v` and recovery variable `u`) compared to LIF's single equation. This gives it richer firing dynamics: it can reproduce regular spiking, bursting, and chattering patterns depending on its parameter regime.

- Activity hooks wired to the two hidden layers (`net[1]`, `net[4]`).
- `clear_hidden_spikes(self)` called at the start of every forward pass.
- `get_adversarial_evaluator()` added.

---

### `src/learning/frameworks/snn_torch.py`

**Neuron model change:** `snn.Leaky` → `snn.Alpha` in all three layers.

`snn.Alpha` is a second-order neuron model with two coupled exponential decays: a synaptic current trace (`alpha`) and a membrane potential (`beta`), where `alpha > beta` is enforced. This makes the neuron's response to input more temporally distributed compared to the instantaneous synaptic integration in `Leaky`.

The forward unpacking changed from `spk_out, _` to `spk_out, *_` because `Alpha` returns three values: `(spk, synaptic_current, membrane_potential)`.

- Activity hooks wired to `net[1]` and `net[4]`.
- `clear_hidden_spikes(self)` called at the start of every forward pass.
- `get_adversarial_evaluator()` added.

---

### `src/learning/frameworks/snn_norse.py`

No neuron model change — Norse `LIFCell` was already the active baseline.

- Activity hooks wired to `net.lif1` and `net.lif2`.
- `clear_hidden_spikes(self)` called at the start of `forward()`.
- `get_adversarial_evaluator()` added.
- Minor cleanup: removed stale `TODO` comment about the `tau_mem_inv` conversion.

---

### `src/learning/main.py`

- Active model switched from `SNN_SJ` to `SNN_NORSE`.
- `adversarial_evaluator` added as a fourth return value.
- Post-inference output now prints `avg_firing_rate_hz`.
- `adversarial_evaluator.evaluate()` called at the end of the run.

---

### `SNN_module.yaml` + `src/skeleton/snn_config.py`

Thirteen new configuration keys, all with safe defaults:

| Key | Default | Purpose |
|---|---|---|
| `trades_enabled` | `false` | Enable TRADES adversarial training |
| `trades_epsilon` | `0.05` | Max perturbation budget per input element |
| `trades_lambda` | `6.0` | Weight of KL robustness term vs clean CE |
| `trades_steps` | `10` | PGD iterations per batch in TRADES |
| `activity_reg_enabled` | `false` | Enable activity regularization |
| `activity_reg_min_rate` | `0.01` | Dead-neuron threshold (1%) |
| `activity_reg_max_rate` | `0.50` | Saturation threshold (50%) |
| `activity_reg_lambda_low` | `0.1` | Dead-neuron penalty weight |
| `activity_reg_lambda_high` | `0.1` | Saturation penalty weight |
| `stdp_enabled` | `false` | Enable STDP regularization |
| `stdp_tau` | `20.0` | Trace decay time constant (timesteps) |
| `stdp_a_plus` | `0.01` | LTP strength |
| `stdp_a_minus` | `0.01` | LTD strength |

Training schedule adjusted: `epochs 10 → 5`, `iterations_per_epoch 100 → 300` (more batches per epoch, fewer epochs total).

In `SNN_module.yaml` the current active config has `trades_enabled: true`, `activity_reg_enabled: true`, and `stdp_enabled: true` — all three objectives are live.

---

## 3. Interface Changes

Every framework model now exposes a third method alongside `get_trainer` and `get_inference`:

```python
model.get_adversarial_evaluator(test_loader) -> AdversarialEvaluator
```

This is a **breaking change** for callers of `main()` — it now returns four values: `(model, trainer, inference, adversarial_evaluator)`.

---

## 4. What Remains Unchanged (at the time of this report)

- Data pipeline (`data_pipeline.py`, `cache_engine.py`) — no functional changes.
- Network architecture shape — still Conv→Neuron→Pool→Conv→Neuron→Pool→FC→Neuron, 2-channel event input, 10 classes.
- Optimizer — Adam with the same hyperparameters.
- AMP, gradient accumulation, CosineAnnealingLR — all unchanged.
- Checkpoint format — unchanged.

> Note: the data pipeline changed substantially in a later session — see
> [`event_data_workflow/caching_pipeline_refactor.md`](event_data_workflow/caching_pipeline_refactor.md).
> `temporal_slicer.py` no longer exists (folded into `data_pipeline.py`).
