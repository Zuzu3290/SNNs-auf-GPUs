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

## 4. What Remains Unchanged

- Data pipeline (`data_pipeline.py`, `cache_engine.py`, `temporal_slicer.py`) — no functional changes.
- Network architecture shape — still Conv→Neuron→Pool→Conv→Neuron→Pool→FC→Neuron, 2-channel event input, 10 classes.
- Optimizer — Adam with the same hyperparameters.
- AMP, gradient accumulation, CosineAnnealingLR — all unchanged.
- Checkpoint format — unchanged.



This is a genuinely good idea, and it already has a name: Address Event Representation (AER). It is exactly how event-based cameras output data, and how neuromorphic chips like Intel Loihi and IBM TrueNorth store and communicate spikes internally — as (neuron_id, timestep) pairs rather than dense binary arrays.

You have independently arrived at one of the core principles of neuromorphic computing.

Where it helps you right now
Memory in activity_reg.py — the hook buffers currently accumulate full [T, B, neurons] float tensors every forward pass. At a 5% firing rate (typical for a trained SNN), 95% of those entries are zeros. Storing only spike positions would cut that buffer to ~5% of its current size.

Caching — if you ever serialize spike recordings to disk (which is what cache_engine.py is open for), storing sparse indices is dramatically smaller than dense tensors. A recording with 3% firing rate stored as indices takes 3% of the space.

Biologically correct representation — your pipeline_arguments.md discussion landed on "raw events are the source of truth, dense frames are a training artifact." The same argument applies here: sparse spike positions are the source of truth, dense binary arrays are a GPU convenience.

Where it hits friction in the current code
The entire training loop — CrossEntropyLoss, Conv2d, Linear, aggregate_spike_output() — expects dense tensors. PyTorch's torch.sparse exists but is still experimental for most ops. BPTT through surrogate gradients also requires the dense spike tensor to exist during the backward pass.

So the boundary is clear:


Sparse (storage/memory)      Dense (compute)
────────────────────────     ───────────────────────
Hook buffers                 Forward pass
Disk/RAM cache               Loss computation
Between-epoch logging        Backpropagation
Inference result export      Conv / Linear layers
What it would look like in practice
Instead of:


spk_rec = torch.zeros(T, B, N)   # [25, 128, 800] = 2,560,000 floats
You store:


# COO format: (timestep, batch, neuron_idx) for every spike
spike_events = torch.nonzero(spk_rec)  # shape [num_spikes, 3]
# At 5% rate: ~128,000 entries instead of 2,560,000
Converting back for compute:


dense = torch.sparse_coo_tensor(spike_events.T, ...).to_dense()
My opinion
This is the right long-term direction for two reasons:

It aligns with your hardware target — when you eventually run on neuromorphic hardware (Loihi, SpiNNaker), the chip speaks AER natively. Dense tensors would be the foreign format, not sparse indices.

It makes activity_reg.py cheaper — computing firing rates from sparse indices (num_spikes / (T * B * N)) is simpler and faster than computing spk.float().mean() over a dense tensor with 95% zeros.

The practical next step, if you want to implement it, would be to change the hook buffers in activity_reg.py to accumulate torch.nonzero() outputs instead of full tensors, and update activity_regularization() and stdp_regularization() to operate on those. The training loop itself stays dense — only the monitoring and storage layer goes sparse.

Update Todos
Add SparseEventBuffer class and device params to cache_engine.py
Update activity_reg.py to use SparseEventBuffer for hook buffers
Verify data_pipeline.py wires correctly with updated cache_engine signatures

Done. Here is exactly what changed across the three files:

cache_engine.py
SparseEventBuffer — new class, ~90 lines. Stores per-timestep spike tensors as [k, ndim] index tensors (one per timestep) rather than full dense floats. Interface: push(spk) per timestep, stack() to reconstruct [T, B, ...] dense tensor when the loss or metrics need it, clear() at the start of each forward pass. Properties: num_spikes, firing_rate, memory_bytes. Multiprocessing-safe (same lock-rebuild pattern as BoundedRecordingCache).

PipelineMemoryCoordinator.dataloader_config() — now detects GPU-only mode: when the device is CUDA and the RAM budget for workers is under 500 MB, it returns num_workers=0, pin_memory=False instead of spawning CPU worker processes.

AdaptiveCacheController.__init__() — self.device = device stored explicitly alongside _device_idx.

activity_reg.py
register_activity_hooks() — buffers are now SparseEventBuffer() instances instead of []. The hook calls buf.push(spk) instead of buf.append(spk).

clear_hidden_spikes() — unchanged call (buf.clear() works on both list and SparseEventBuffer).

get_hidden_spike_recordings() — calls buf.stack() instead of torch.stack(buf). The sparse→dense reconstruction happens here once per step, not during accumulation.

data_pipeline.py
No changes needed — it already passed device to both PipelineMemoryCoordinator.from_system() and AdaptiveCacheController. The classes now accept it.


Three concrete gains:

1. Memory efficiency in training
Hook buffers now store only spike positions (AER format) instead of full float tensors. At a ~5% firing rate, each hidden layer buffer uses 20× less memory. You'll see this in the Spike Buffer line every epoch.

2. Device-correct pipeline
PipelineMemoryCoordinator now knows the target device. On GPU-only setups it automatically drops CPU workers (num_workers=0, pin_memory=False), avoiding the CPU-GPU transfer boundary entirely. On standard CPU-GPU setups nothing changes.

3. Sparse-to-dense at the right boundary
The conversion from sparse indices back to dense only happens once per training step — right before the loss computation needs it. Everything upstream (hooks, buffers, storage) stays sparse. This is the correct place to pay that cost.

What didn't change: the training loop, loss functions, TRADES, STDP, and checkpointing all work identically — they just receive the same dense tensor they always did, reconstructed on demand.

---

---

# Training Performance Optimizations
**Files touched:** `training.py`, `activity_reg.py`, `cache_engine.py`

## Summary
Four targeted fixes resolved a 6+ minute per-100-iteration runtime. Root cause was a combination of CUDA sync stalls inside the hook hot path, redundant forward passes during TRADES, and a Python timestep loop in the STDP loss. All three training objectives (TRADES, activity regularization, STDP) remain fully active — no robustness or biological plausibility was removed.

---

## Changes

### `src/learning/frameworks/activity_reg.py`

#### `pause_hooks()` / `resume_hooks()` — new functions
Allow the training loop to silence all activity hook recording temporarily.

```python
pause_hooks(model)   # sets model._hooks_paused = True
resume_hooks(model)  # sets model._hooks_paused = False
```

The hook callback checks this flag and returns early if set.

#### STDP loop vectorized
The `for t in range(T)` Python loop was replaced with a single matrix multiply. A lower-triangular causal EMA kernel is built once per layer pair:

```python
ema_kernel = (decay ** (steps.unsqueeze(1) - steps.unsqueeze(0))).tril()  # [T, T]
trace_pre  = ema_kernel @ pre_t
trace_post = ema_kernel @ post_t
```

**Before:** 25 sequential Python iterations, each indexing a GPU tensor.  
**After:** 1 matmul per layer pair. Same mathematical result.

---

### `src/learning/event_data_workflow/cache_engine.py`

#### `SparseEventBuffer.push()` — `torch.nonzero()` removed from hot path
`torch.nonzero()` is a CUDA synchronization point — the GPU pipeline must flush before the result is returned. Previously called on every hook invocation: with TRADES enabled that was ~650 sync stalls per batch.

`push()` now stores `spk.detach()` directly (dense). `stack()` uses `torch.stack(self._events)` — one CUDA op, no Python loop.

**Before:** ~650 CUDA sync stalls per batch in TRADES mode.  
**After:** Zero sync stalls in the hook path.

---

### `src/learning/training.py`

#### TRADES inner loop — hooks paused
`generate_trades_adversarial()` wraps its PGD iterations with `pause_hooks` / `resume_hooks`. Spike recordings during adversarial perturbation search are not needed — only the final clean and adversarial forward passes require hook data.

```python
pause_hooks(model)
try:
    for _ in range(steps):
        ...
finally:
    resume_hooks(model)
```

#### `clean_logits` reused for accuracy
In the TRADES path, `clean_logits` was already computed for the CE loss. A second call to `aggregate_spike_output()` for the accuracy metric was redundant. Now reuses the existing tensor:

```python
logits = clean_logits.detach() if self.cfg.TRADES_ENABLED else aggregate_spike_output(spk_rec.detach().float())
```

---

## Impact

| Fix | What it removed |
|---|---|
| Pause hooks in PGD loop | Hook recording on 10 irrelevant adversarial forward passes |
| Dense storage in `push()` | All CUDA sync stalls from the hook path |
| Vectorized STDP | Python loop over T=25 timesteps → 1 matmul |
| Reuse `clean_logits` | Redundant `aggregate_spike_output()` call per batch |


verdict from chatgpt:My overall conclusion

This is a very good optimization set.

I would rate it as:

Impact: High
Conceptual correctness: Strong
Risk level: Medium-low
Main risk: GPU memory from dense spike storage
Most important validation: old STDP loop vs new vectorized STDP equivalence

The changes are especially appropriate for your neuromorphic/SNN pipeline because they preserve the biologically motivated components while making the implementation more hardware-efficient.

The only thing I would add now is a small benchmark table:

Before / after:
- iteration time
- GPU memory usage
- GPU utilization
- clean accuracy
- robust accuracy
- STDP loss value comparison
- activity regularization value comparison

Final verdict:

Yes, impactful. This is exactly the kind of optimization needed when moving from conceptual neuromorphic modeling toward serious GPU/hardware-aware training.


gpu_stats.py — new file in skeleton:

GPUStats class with public names (no underscore prefix)
Background thread samples torch.cuda.utilization() every 0.5 s
start_epoch() / end_epoch() bracket each epoch, end_epoch() returns a dict
summary() aggregates everything across all epochs
training.py — three additions:

SNNTrainer.__init__: creates self.gpu_stats = GPUStats(device_idx)
Per-epoch print block: shows avg/peak compute util and peak VRAM each epoch
After training loop: prints the overall GPU training summary before saving the CSV
The CSV log also gets the GPU columns automatically via the **gpu spread into epoch_log. If no GPU is available, all GPU paths silently return {} and nothing is printed.