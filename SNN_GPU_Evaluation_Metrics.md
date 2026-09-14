# SNN-on-GPU: Evaluation Metrics Documentation

**Project:** Spiking Neural Networks on GPUs
**Purpose:** Define the evaluation metrics used to validate the network, explain the rationale for each, describe how each is measured, and situate the metric set against what is used elsewhere in the SNN literature.

---

## 1. How this document is organized

For every metric below you'll find:
- **Definition** — what it is, in plain terms.
- **Why we track it** — what it tells us about whether the network is "working" (learning, generalizing, being efficient, being fast, or being deployable).
- **How it's computed** — formula / measurement method.
- **Caveats** — where the metric can mislead if read alone.

The metric set is organized into four families: **task performance**, **spiking-activity / sparsity**, **timing**, and **resource / energy**. SNN evaluation in the literature consistently reports metrics from all four families together, because no single family is sufficient — a network can be accurate but slow, sparse but power-hungry on real hardware, or fast but only because it undertrains.

**Note on the deferred-sync logging model:** `learning/training.py` and
`learning/inference.py` compute every per-batch metric above as a
GPU-resident tensor (or, for latency, an un-synced `torch.cuda.Event` pair)
rather than calling `.item()`/`.cpu()` on it immediately. `.item()`/`.cpu()`
force a CUDA-stream synchronization — waiting for the GPU to finish
everything queued so far — so doing it every batch (as the code used to)
stalls the GPU on every iteration. Instead, results accumulate GPU-side for
the whole training run, and are read back to host memory in one bulk
transfer only after the last epoch finishes, right before inference starts —
at which point the full per-epoch report prints in order. The one exception
is a single small sync per epoch used purely to decide whether to save the
best-so-far checkpoint (control flow, not a metric read). Practically: don't
expect per-epoch terminal output to stream live during a training run — it
now appears as one block immediately after training completes.

---

## 2. Metric Definitions

### 2.1 Task performance

| Metric | Definition | Formula |
|---|---|---|
| **Accuracy** | Fraction of correctly classified test samples (top-1, or top-k if relevant) | `correct / total` |
| **Loss (avg. per epoch)** | Mean training/validation loss across all batches in an epoch | `mean(batch_loss)` over the epoch |

**Why we track them:** Accuracy is the ground-truth check that the network is actually solving the task; it's the metric every other metric gets traded off against. Per-epoch average loss is the earliest, cheapest signal of whether training is converging, plateauing, diverging, or overfitting (compare train vs. validation loss curves) — it's what you look at *before* a full epoch's accuracy is even meaningful, and it's the standard diagnostic for tuning learning rate, surrogate-gradient shape, and number of timesteps.

**Caveat:** Accuracy alone says nothing about *how* the network got there — two networks at 91% accuracy can differ by 10x in energy or 5x in latency. This is precisely why the other three families exist.

---

### 2.2 Spiking-activity / sparsity metrics

| Metric | Definition | Formula |
|---|---|---|
| **Spike rate** | Average number of spikes emitted per neuron per timestep, over the inference window | `total_spikes / (num_neurons × T)`, unit-free (spikes/neuron/timestep) |
| **Spike ratio (network-wide sparsity)** | Fraction of neurons that are active (fire at least once, or fraction of possible spike-events that actually occur) across the *whole* network over an inference window | `active_neuron_events / total_possible_neuron_events` |

**Why we track them:** These are the metrics that are unique to SNNs and have no ANN equivalent — they're the direct evidence that the network is exploiting temporal/event-driven sparsity rather than behaving like a disguised dense ANN. A network that fires on nearly every timestep isn't gaining anything from the spiking paradigm; it's paying SNN training cost for ANN-level density. Firing rate is also diagnostic of training pathologies (dead neurons if it's near 0, saturating/ANN-like behavior if it's near 1).

**Caveat:** Spike rate and spike ratio are *proxies*, not the same thing as energy — see §2.4. A network can have low average firing rate but concentrate spikes in a few very fan-out-heavy layers, which dominates real energy cost.

**Report `spikes_per_neuron_per_inference` as the headline, not Hz.** Event-driven data (a DVS camera, or any spike train derived from one) has no characteristic real-world frequency the way a clocked signal does, so converting a spike count into Hz manufactures a unit the underlying data never had — and a plain average across neurons also hides that some fire heavily while others stay silent. `spikes_per_neuron_per_inference = spike_rate × T` needs no time unit, cannot be wrong, and is the unit the SNN literature reports — so it is directly comparable with published figures. This project does not compute or report a Hz figure anywhere.

#### Inter-Spike Interval variability — CV(ISI)

Implemented in `learning/utilities.py`'s `compute_cv_isi()`, called from
`SNNTrainer.finalize_one_epoch_report()` immediately after each epoch finishes
(one value per epoch, last batch / sample 0) and `SNNTester.run()` (one value for
the whole test pass, last batch / sample 0).

| Metric | Definition | Formula |
|---|---|---|
| **Mean Inter-Spike Interval (ISI)** | Average time between consecutive spikes for a neuron | `mean(t[i+1] - t[i])` over a neuron's spike train — mathematically ≈ `1 / firing_rate`, so mostly a unit-conversion of spike rate above |
| **Coefficient of Variation of ISI (CV_ISI)** | Regularity of a neuron's firing, independent of its rate | `std(ISI) / mean(ISI)`, averaged per layer and network-wide |

**Why we track it:** mean ISI alone doesn't add information beyond spike rate — but CV_ISI does, and it's the metric that actually answers "is this neuron spiking too fast/erratically or too slow/rarely," independent of whether its *average* rate looks fine. A CV_ISI near 0 means clock-like, regular firing; a CV_ISI at or above 1 means irregular/bursty firing. A network can have a perfectly healthy-looking mean firing rate while individual neurons are alternating between bursts and silence — CV_ISI is what surfaces that. It's a standard neuroscience regularity measure, and recent SNN work uses it directly: neurons with low CV_ISI (regular, "clock-like" firing) are treated as encoding stable, task-relevant features, while high-CV_ISI neurons are treated as noisy or uncommitted.

**Caveat — don't over-read this as "robustness":** CV_ISI is a *stability/health diagnostic* for individual neurons (flags dead, saturating, or erratically-firing units), not a direct measure of network robustness to input noise. If robustness is the actual target property, measure it explicitly — e.g., accuracy degradation under injected spike noise or dropout (the literature in §3 reports numbers like a specific accuracy drop under a given noisy-spike rate) — and treat CV_ISI as a complementary diagnostic that helps *explain* a robustness result, not a substitute for measuring it.

---

### 2.3 Timing metrics

| Metric | Definition | Formula / method |
|---|---|---|
| **Latency (network)** | Wall-clock time for one forward pass (one inference), typically per sample or per batch, including all T timesteps | Measured with `torch.cuda.synchronize()` bracketing the forward call |
| **Backward-pass latency (network)** | Wall-clock time for one backward pass — from the loss to the gradient at the first layer's parameters/inputs | `torch.cuda.Event`-timed (see §4.5 — not a `synchronize()`-bracketed timer; see §1's note on the deferred-sync logging model) |
| **Training time per epoch** | Wall-clock time to complete one full epoch (forward + backward + optimizer step, all batches) | Stopwatch around the epoch loop |
| **Throughput** | Samples processed per second | `num_samples / wall_clock_time` |

**Why we track them:** Latency is what determines whether the network is deployable for a given real-time constraint (e.g., robotics, edge sensing) — this is a hardware-facing metric, distinct from accuracy or loss. Backward-pass latency is the direct mirror of forward latency, measured on the gradient computation instead — it's the diagnostic for "how long does it take the error to propagate from the output back to the input layer," which matters because in BPTT-style training the backward sweep moves through *both* network depth and simulated time jointly, so it can dominate total step time in ways forward latency alone won't reveal. Training time per epoch is what determines whether your experiment matrix (hyperparameter sweeps, architecture search) is actually feasible, and it's the metric that best signals GPU under-utilization: if compute-per-epoch is far below what the hardware should deliver, the bottleneck is elsewhere (data loading, Python-level spike-loop overhead, small batch size).

**Caveat:** Latency (forward *and* backward) depends heavily on the number of simulated timesteps T — always report both alongside T, and separate "per-timestep" latency from "per-inference" (T-timestep) latency so different encoding schemes are comparable. Backward-pass latency is also meaningless as a cross-framework comparison unless you also record *which* credit-assignment algorithm produced it — see §2.5.

#### Two per-sample timings that must not share a name

Both are recorded, because they answer different questions and neither substitutes for the other. What matters is that they are labelled distinctly.

| reported as | how it is obtained | what it actually answers |
|---|---|---|
| **`latency_single_stream_ms`** | batch size **1**, `synchronize()` around each individual sample, reported as **median** and **p90** over N samples | *"If one event arrives, how long until the answer is ready?"* The deployment/real-time question. MLPerf Single-Stream convention. |
| **`latency_per_sample_amortised_ms`** | one batch timed as a whole, then divided by the batch size | *"At this batch size, how much wall-clock does each sample cost?"* A throughput figure expressed per sample. |

Why the distinction is not pedantic:

- The amortised figure is **throughput under batching**, not latency. It benefits from parallelism a single arriving event cannot use, so it is systematically optimistic as a deployment number — often by a large factor.
- **Percentiles built from the amortised figure describe the wrong thing.** Every sample in a batch is assigned the same divided value, so the spread measures batch-to-batch variation, not sample-to-sample. A p90 computed that way is not a tail latency.
- Only the single-stream figure is comparable with published latency numbers, which almost universally use batch size 1.

Both are cheap and non-intrusive: the amortised figure falls out of timings already taken during the normal test pass, and the single-stream measurement is a separate, small, dedicated pass (default 100 samples) run **outside** any timed region, so recording one never disturbs the other.

---

### 2.3b Credit-assignment (backpropagation) method — record as metadata, not a scalar

This isn't a number to plot, but it needs to be logged alongside every timing/memory result, because it's the single biggest determinant of what those numbers mean. SNN frameworks differ fundamentally in how they compute gradients, and the differences aren't cosmetic:

| Method | How it works | Time/memory profile |
|---|---|---|
| **BPTT + surrogate gradient** (most common — SpikingJelly, snnTorch, Norse) | Unrolls the network across all T timesteps, replaces the non-differentiable spike function's derivative with a smooth surrogate, backpropagates through the full unrolled graph (spatial *and* temporal) | Memory and compute scale with network size × T; this is why memory cost "grows with sequence length," a documented limitation |
| **e-prop / RTRL (eligibility propagation)** | Forward-propagates local eligibility traces per synapse instead of unrolling backward through time; more biologically plausible, online-capable | Constant memory w.r.t. T, but forward-propagated gradient methods carry substantial multiply-op overhead relative to backprop, and can be dramatically slower on GPU — one benchmark reported a 108× GPU speed gap between a segment-parallel method and mathematically-equivalent e-prop |
| **FPTT / truncated BPTT** | Computes updates from an instantaneous loss with a regularization term against a running parameter average, avoiding full-sequence unrolling | Reduces memory/compute relative to full BPTT for long sequences, trades off some gradient fidelity |
| **SLTT / spatial-only gradients** | Decomposes the BPTT gradient into spatial and temporal components and drops (or approximates) the temporal component | Significantly lower training time and memory than full BPTT with comparable accuracy in the reporting papers |
| **Equilibrium propagation / tandem learning** | Uses a single (or paired) circuit relaxing to fixed points, or trains an auxiliary ANN-level proxy, instead of a separate backward pass | Different computational graph entirely — direct latency comparison to BPTT needs care |

**Why this matters for your metric set:** a backward-pass latency number from a BPTT-trained network and one from an e-prop-trained network are not directly comparable — they're measuring different algorithms with different big-O behavior in T. Record the credit-assignment method as a required field on every result row, the same way you'd record model architecture or dataset.

**Implementation:** `ModelInterface.credit_assignment()` (default `"BPTT+SG"`,
covering every framework currently wired in) is logged as a field on every
epoch/batch record in `training.py`/`inference.py`, right alongside the
backward-latency number it qualifies.

---

### 2.4 Resource & energy metrics

| Metric | Definition | Formula / method |
|---|---|---|
| **Power** | Instantaneous GPU power draw during training or inference | Read from `nvidia-smi` / NVML power sensor (Watts) |
| **Energy (total)** | Energy consumed over a phase, **idle draw included** | `∫ Power(t) dt` by trapezoidal integration over the sampled series (Joules) |
| **Energy (dynamic)** | Energy **above** the idle baseline — what the computation itself cost | `total − idle_power_w × elapsed_s`, clamped at 0 |
| **Energy (inference)** | Either of the above, scoped to the inference-only time window | Same method, different window |
| **Memory / VRAM usage** | Peak and average GPU memory allocated during training/inference | `torch.cuda.max_memory_allocated()`, `torch.cuda.memory_allocated()` |
| **SynOps (Synaptic Operations)** | Hardware-agnostic proxy for the actual event-driven compute cost, accounting for spike sparsity | `SynOps = DenseOps × SpikingRate` (per layer, then summed), i.e. spikes emitted by a layer weighted by its fan-out |

**Why we track them:** Energy is arguably *the* headline justification for using SNNs at all — the entire research motivation is "comparable accuracy at lower energy than an ANN," so without an energy number the project can't actually support its own premise. Power and time are reported separately (rather than only energy) because they diagnose different problems: high power at short time = compute-bound and GPU well utilized; low power at long time = GPU sitting idle waiting on something else (data loading, host-device sync, small batch size), which is exactly the failure mode you're asking about below. Memory usage tells you whether you have headroom to grow batch size / model size, or whether you're close to OOM.

**Caveat — important for SNN work specifically:** raw GPU-measured energy (via `nvidia-smi`/NVML) reflects the cost of *simulating* the SNN densely on a GPU, which is not event-driven hardware — a GPU still computes the zero-valued (non-spiking) terms, it just doesn't skip them the way neuromorphic silicon (Loihi, etc.) would. This is why the SNN literature almost universally reports a *second*, hardware-agnostic energy number alongside the measured GPU number: **SynOps-based estimated energy**, using standard per-operation energy costs from Horowitz's 45nm process numbers (~0.9 pJ per accumulate (AC), ~4.6 pJ per multiply-accumulate (MAC)) to convert SynOps into an estimated Joules figure that reflects what a spiking accelerator *would* consume. Report both: "measured GPU energy" (what your experiment actually cost) and "estimated event-driven energy" (what the network's sparsity implies it would cost on target hardware).

#### How the measured GPU energy is obtained, and why each choice

Every point below exists because the naive version of it produces a number that looks fine and is wrong.

**Total and dynamic are both reported, and they are not interchangeable.** `energy_j_total` includes the 20–40 W a GPU draws simply being powered on; `energy_j_dynamic` subtracts it. Two consequences decide which to quote:

- *Idle draw dilutes differences.* At a 30 W idle, one framework drawing 50 W against another at 70 W differs by **2.0×** dynamically but only **1.4×** in total. The total understates exactly what a comparison is looking for.
- *Total scales with duration.* A framework that takes twice as long reports twice the energy at identical power draw — which measures runtime a second time rather than efficiency.

Both are written to the results row alongside `idle_power_w`, so either can be reconstructed and the baseline used is never implicit. `avg_power_w × elapsed` equals the total; `dynamic_power_w × elapsed` equals the dynamic. Nothing sits on a third basis.

**Trapezoidal integration, not mean × elapsed.** Power under a real training loop is not flat — it dips between batches and during data stalls. `mean × elapsed` is only equal to the integral when sampling is perfectly uniform and the window matches the samples exactly; integrating the actual `(timestamp, watts)` series makes no such assumption.

**The idle baseline is measured hot, not just cold.** Idle draw on a warm card is materially higher than on a cold one. The honest baseline for work that has just finished is the one taken immediately after it, at working temperature — so both are recorded and the **hot** one is subtracted. Where no baseline was measured at all, dynamic falls back to the total and `idle_power_w` reads `None`, rather than a baseline being invented.

**Poll rate is checked against the sensor's own refresh rate.** NVML updates its power reading only every N milliseconds; sampling faster returns the same value repeatedly and manufactures precision that is not there. The detected interval is recorded, and polling faster than it raises a warning.

**Negative dynamic energy is clamped at 0 and flagged.** A load quieter than the recorded idle means the baseline was wrong, not that the work produced energy.

**Sanity warnings are emitted, not suppressed.** Each of these has been seen in practice and each invalidates the number rather than merely degrading it:

| warning | what it means |
|---|---|
| dynamic energy is negative | the idle baseline is above the measured load — baseline is wrong |
| mean power under load ≤ idle | the measured region did not actually load the GPU |
| idle baseline range > 50% of its mean | the baseline is unstable; subtraction is not meaningful |
| poll interval < NVML update interval | duplicate readings; the precision is fictitious |

**Implementation:** the dense-MAC side of SynOps is measured, not assumed —
`learning/utilities.py`'s `measure_dense_macs()` runs one real forward pass
with temporary hooks capturing the exact input each downstream dense module
(the conv/linear layer immediately after a spiking layer) receives, then
measures that module's dense FLOPs with `torch.utils.flop_counter.FlopCounterMode`
(ships with torch ≥2.1, no fvcore/ptflops dependency) and halves it to MACs.
`ModelInterface.synops_layer_map()` supplies which spiking layer feeds which
downstream module, derived from the shared network's own layer list so it
cannot fall out of sync with the architecture.

It returns empty only for a framework that genuinely cannot be hooked once
per timestep. Sinabs *used* to be that case — its LIF consumed a whole
(B, T, ...) sequence per call — so it silently produced no SynOps, no CV_ISI
and, with activity regularisation on, no penalty at all, while the other
three produced all three. The shared network feeds it one timestep at a
time, so all four are now hooked and all four report identical dense-MAC
counts. SynOps
itself (`firing_rate × dense_MACs × T`, summed over layers, ×4.6 pJ/MAC) is
computed by `SNNTrainer.train()`/`SNNTester.run()` and reported alongside —
not instead of — the flat spike-count energy estimate.

---

### 2.4b Runtime GPU diagnostics

| Metric | Definition | Formula / method |
|---|---|---|
| **GPU temperature** | Instantaneous GPU die temperature | `pynvml.nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)`, °C |
| **SM clock speed** | Streaming-multiprocessor (compute) clock frequency | `pynvml.nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM)`, MHz |
| **Memory clock speed** | GPU memory-bus clock frequency | `pynvml.nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM)`, MHz |
| **Max memory reserved (cached)** | High-water mark of memory PyTorch's *caching allocator* has claimed from the driver — distinct from max memory *allocated* (§2.4's Memory/VRAM usage), which is memory actually in use at the peak moment | `torch.cuda.max_memory_reserved()` |
| **CUDNN autotune status** | Whether cuDNN's algorithm autotuner (picks the fastest conv algorithm for the actual input shapes, at the cost of a short warm-up) is active | `torch.backends.cudnn.benchmark` (boolean) |

**Why we track them:** GPU-Util% and power (§2.4) tell you *whether* the GPU
is busy; clock speed and temperature tell you *how hard* it's running while
busy — a GPU throttling under thermal/power limits shows the same "busy"
utilization number as one running at full clock, but produces very different
throughput, which matters for interpreting any latency/throughput regression
that isn't explained by a code change. Max memory *reserved* vs. *allocated*
distinguishes "the allocator is holding VRAM it isn't currently using" (often
recoverable via `torch.cuda.empty_cache()`) from "the model genuinely needs
this much" — a config/batch-size tuning signal §2.4's allocated-memory number
alone can't give. CUDNN autotune status is metadata, not a measurement, but
it materially changes forward-latency numbers between runs (first-batch
warm-up cost, and different chosen kernels) — logged for the same reason
credit-assignment algorithm is (§2.3b): so a latency comparison across runs
isn't silently comparing apples to oranges.

**Caveat:** these are point-in-time NVML/driver queries taken once per
epoch (training) or once per test run (inference), not integrated/averaged
over the run the way §2.4's power sampling is — a momentary temperature or
clock reading, not a peak or mean. Reading them doesn't force a CUDA-stream
sync (they query the driver, not kernel completion), so they're safe to call
without reintroducing the per-batch stalls described in §1's note on the
deferred-sync logging model.

**Implementation:** `learning/utilities.py`'s `read_gpu_runtime_diagnostics()`,
called once per epoch from `SNNTrainer.train()` and once per run from
`SNNTester.run()`. Reads NVML directly off `GPUStats`'s already-public
`nvml_handle`/`device_idx` rather than extending `GPUStats` itself.

---

## 3. What the field uses: literature cross-check

To validate that this metric set is aligned with how the SNN community evaluates networks, here's what recent representative work reports:

| Work / focus | Metrics used |
|---|---|
| Multi-framework SNN benchmark (SpikingJelly, BrainCog, Sinabs, SNNGrow, Lava) | Accuracy, latency, energy consumption, noise immunity, plus qualitative model-complexity and framework-adaptability scores |
| Pruned spiking SqueezeNet vs. CNN baselines | Top-1 accuracy, F1-score, parameter count, MACs (via `ptflops`), estimated energy |
| SNN hardware-accelerator evaluation (non-volatile-memory devices) | Inference accuracy, latency, energy consumption (down to pJ scale) |
| Sensor-encoding scheme comparison (deployed on Intel Loihi 2) | Average firing rate, signal-to-noise ratio, classification accuracy, robustness to spike noise, inference latency, inference energy |
| SNN Architecture Search survey | Accuracy, latency, energy consumption, memory footprint, silicon area; explicitly **warns that raw FLOPs is a weak/misleading metric** for SNNs since identical FLOPs can produce very different latency/energy on different hardware; recommends spike-count-based proxies (SynOps, "Bit-SynOps", number-of-spikes) instead |
| Practical SNN tutorial/benchmark (Lava, SLAYER, SpikingJelly, Norse) | Accuracy, number of timesteps, spike activity, power-oriented proxy metrics — explicitly framed as an **accuracy-vs-energy trade-off** study |
| Accuracy-vs-energy space exploration (space-applications SNNs) | Accuracy plotted against a dimensionless per-MAC energy metric, reported as a Pareto front rather than single numbers |
| Human-activity-recognition SNN on Loihi vs. RISC-V ANN chip | Accuracy, and the combined **energy-delay product** as a single efficiency figure |
| SNN NoC/architecture benchmarking | Latency, energy, and the combined **energy-latency product** |
| EFLOP metric proposal | A sparsity/zero-skip-aware FLOP variant designed specifically to fix the "FLOPs is misleading for sparse/spiking models" problem noted above |
| SynOp-loss / synaptic-operation regularization work | SynOps computed directly as (spikes emitted per layer × layer fan-out), used both as a *metric* and as a training-time regularization target |
| BPTT vs. e-prop / RTRL comparisons (HYPR, SpiNNaker2 e-prop implementation) | Peak/total memory (KB), GPU wall-clock training time, and explicit reporting of which credit-assignment algorithm produced each number — treated as a required experimental variable, not incidental detail |
| SLTT (spatial-only gradient) work | Training time and memory cost directly compared against full BPTT-with-surrogate-gradient on the same architectures/datasets, at equivalent accuracy |
| ISI-CV for SNN continual learning | Coefficient of variation of inter-spike intervals used as a gradient-free, per-neuron stability/importance signal — low CV flags stable feature-encoding neurons, high CV flags noisy/uncommitted ones |
| Encoding-scheme robustness study (Loihi 2, cited above) | Explicit robustness metric: classification accuracy measured *under injected spike noise* at varying noise rates, reported as accuracy drop (%) — the direct way the field measures robustness, distinct from firing-regularity diagnostics |

**Takeaway for your documentation:** your current metric set (accuracy, loss, spike rate, spike ratio, latency, energy, power, time, training-time/epoch, memory) already matches the core of what's published. Two gaps are worth closing: an explicit **SynOps-based energy estimate** (near-universal in the literature because it's the number that lets you claim SNN efficiency in a hardware-target-relevant way, not just a GPU-simulation way), and — per your latest round of additions — **the credit-assignment algorithm as a required metadata field** plus **backward-pass latency and CV_ISI** as first-class metrics, both of which are directly precedented above. Consider also reporting a combined **energy-delay product (EDP)** or **energy × 1/accuracy** trade-off point if you want a single headline efficiency number for comparisons.

---

## 4. Measuring these metrics: tooling

### 4.1 Quick, real-time GPU checks (no code changes)

```bash
# Live utilization / memory / power, refreshed every second
nvidia-smi --loop=1

# Per-process compute + memory-bandwidth breakdown
nvidia-smi pmon -s um

# Lightweight persistent dashboard
gpustat -i 1        # pip install gpustat
nvtop                # interactive terminal UI
```
`nvidia-smi`'s "GPU-Util %" is the fraction of time *some* kernel was running on the GPU — it tells you whether the GPU is idle, but **it does not tell you whether that kernel is using the GPU's full arithmetic throughput.** A training loop can show 95%+ util while running tiny, memory-bound kernels far below peak FLOPs/s. That distinction matters for your "is it using all available resources" question — see §4.3.

### 4.2 Sustained/long-run monitoring (over a whole training job)

- **NVIDIA DCGM** (Data Center GPU Manager) + Prometheus/Grafana — logs SM utilization, SM occupancy, memory-bandwidth utilization, PCIe/NVLink traffic, and power over the full run, so you can plot energy and utilization against epoch number instead of single-point snapshots.
- A simple background logger also works for a single-GPU project: poll `nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1` into a CSV during training, then integrate the power column over time to get Joules for §2.4.

### 4.3 FLOPs / compute-utilization measurement (PyTorch)

Three complementary layers:

**a) Static op/MAC counting** (how much compute the model *should* need per forward pass):
```python
from fvcore.nn import FlopCountAnalysis
flops = FlopCountAnalysis(model, sample_input)
print(flops.total())          # total MACs for one forward pass
print(flops.by_module())      # per-layer breakdown
```
`ptflops` and `thop` are drop-in alternatives; DeepSpeed's `flops-profiler` and the standalone `flops-profiler` package do the same thing but also report backward-pass FLOPs (estimated as ~2× forward) and per-module latency in the same pass, which is convenient for spotting bottleneck layers.

**b) Dynamic profiling** (what the GPU actually executed, with timing and memory):
```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    output = model(sample_input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
```
This gives you, per operator: CUDA time, achieved FLOPs, and memory allocated — which is the direct way to see whether GPU time is going to actual compute or to overhead (data transfer, kernel launch, Python-level spike loops).

**c) Achieved vs. peak throughput ("is it using all available resources")**:
GPU-Util% near 100 is necessary but not sufficient for "fully utilized." The number you actually want is **Model FLOPs Utilization (MFU)**: achieved FLOPs/s (from the profiler above) divided by the GPU's theoretical peak FLOPs/s (from its spec sheet, at the precision you're training in — fp32/fp16/bf16 peaks differ a lot). Getting the true kernel-level roofline picture (compute-bound vs. memory-bound, occupancy, warp efficiency) requires **NVIDIA Nsight Systems** (`nsys profile`, timeline view of GPU-active vs. GPU-idle vs. data-movement) and **NVIDIA Nsight Compute** (`ncu`, per-kernel roofline analysis) — these are the tools to reach for once `torch.profiler` tells you *which* op is the bottleneck and you want to know *why* it isn't hitting peak throughput.

For VRAM specifically:
```python
torch.cuda.memory_allocated()       # current allocation
torch.cuda.max_memory_allocated()   # peak since last reset — report this
torch.cuda.reset_peak_memory_stats()
```

### 4.5 Backward-pass latency (and per-layer breakdown)

A `torch.cuda.synchronize()`-bracketed timer works but stalls the GPU at
every measurement — fine for a one-off profiling script, but not for
something measured every training batch (see §1's note on the deferred-sync
logging model). The implementation here uses non-blocking `torch.cuda.Event`
pairs instead — `record()` queues on the stream without waiting, so
measuring every batch costs nothing until the numbers are actually read:
```python
import torch, time

if device.type == "cuda":
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    loss.backward()
    end.record()
    # elapsed_time() is the only blocking call — do this once, later, not per batch:
    # backward_latency_ms = start.elapsed_time(end)
else:
    t0 = time.perf_counter()
    loss.backward()
    backward_latency_ms = (time.perf_counter() - t0) * 1000
```
See `SNNTrainer._timed()` / `SNNTester._timed()` in `learning/training.py` /
`learning/inference.py` for the context-manager version actually used, and
`SNNTrainer.train()` for where `elapsed_time()` is read back once per epoch
boundary, not per iteration.

For a per-layer breakdown (which layer/timestep dominates the backward sweep), register a full backward hook on each module and timestamp when its grad arrives:
```python
layer_times = {}

def make_hook(name):
    def hook(module, grad_input, grad_output):
        layer_times[name] = time.perf_counter()
    return hook

for name, module in model.named_modules():
    module.register_full_backward_hook(make_hook(name))
```
Comparing these timestamps against the forward-pass op timings from `torch.profiler` (§4.3b) tells you directly whether the backward sweep is dominated by a specific layer, a specific timestep, or is roughly uniform — and whether it's the surrogate-gradient computation itself or memory movement that's the bottleneck. Remember to log the credit-assignment algorithm (§2.3b) alongside every number here — the profile looks structurally different for BPTT vs. e-prop/RTRL-style methods.

### 4.6 Inter-spike interval and CV_ISI

```python
import numpy as np

def cv_isi(spike_times):
    # spike_times: sorted 1D array of spike timestamps for one neuron
    if len(spike_times) < 2:
        return None  # not enough spikes to define an interval
    isi = np.diff(spike_times)
    return isi.std() / isi.mean()

# Per layer: collect spike_times per neuron (e.g. via SpikingJelly's
# activation_based.monitor, or a hook accumulating spike indices per timestep),
# compute cv_isi() per neuron, then average across the layer / network.
```
Implemented as `compute_cv_isi()` in `learning/utilities.py`, following this
same per-neuron logic on sample 0 of each layer's recorded spike train.

### 4.4 SNN-specific: measuring SynOps rather than raw FLOPs

Because SNN compute cost is event-driven, a static FLOP count (§4.3a) describes only the *dense* upper bound. To get the sparsity-adjusted number used throughout §3's literature:

1. Get per-layer dense MACs from `ptflops`/`fvcore` as in §4.3a — or, as
   actually implemented here (`learning/utilities.py`'s `measure_dense_macs()`),
   `torch.utils.flop_counter.FlopCounterMode` (built into torch ≥2.1), which
   avoids adding either as a project dependency.
2. Register forward hooks on each spiking layer to accumulate output spike counts over the inference window (SpikingJelly's `activation_based.monitor` module does this out of the box if you're using that framework; otherwise a simple hook summing `output.sum()` per timestep works) — this project's `ActivityMonitor`/`DenseTimestepBuffer` (§2.2) already does this for all four frameworks, including Sinabs.
3. Combine: `SynOps_layer = mean_firing_rate_layer × dense_MACs_layer`, sum across layers.
4. Convert to an estimated-Joules figure using per-operation energy constants (commonly the Horowitz 45nm values: ≈0.9 pJ/AC, ≈4.6 pJ/MAC) to get the "if this ran on event-driven hardware" energy number described in §2.4.

Report this SynOps-based estimate *alongside* (not instead of) the directly-measured GPU energy from §4.2 — the gap between the two numbers is itself informative (it's roughly how much headroom your sparsity would buy you on real neuromorphic hardware vs. what you're actually paying on a GPU today).

---

## 5. Suggested reporting template (per experiment / checkpoint)

| Field | Value |
|---|---|
| Accuracy (top-1) | |
| Loss (final epoch, train / val) | |
| **Spikes / neuron / inference** (rate × T) — the headline sparsity figure | |
| Mean spike rate (per neuron per timestep) | |
| Network-wide spike ratio | |
| Timesteps (T) | |
| Credit-assignment algorithm (BPTT+SG / e-prop / FPTT / SLTT / other) | |
| **Latency, single-stream** (batch 1) — median / p90 | |
| **Latency, per-sample amortised** (batch time ÷ B) — *throughput, not latency* | |
| Latency / inference, backward | |
| Mean CV_ISI (network-wide) | |
| Training time / epoch | |
| Throughput (samples/s) | |
| Peak VRAM (allocated) | |
| Max VRAM reserved (cached) | |
| **Measured GPU energy — total** (idle included) / inference / epoch | |
| **Measured GPU energy — dynamic** (above idle) / inference / epoch | |
| **Idle power baseline (W)** — hot; `None` if never measured | |
| Estimated SynOps-based energy / inference | |
| GPU temperature | |
| SM / memory clock speed | |
| CUDNN autotune enabled | |
| GPU (model, precision used) | |
| **Warm-up iterations** (untimed, before the timed region) | |
| **Batch size, and whether it was VRAM-calibrated** | |
| **Seed** | |

**Two figures per row, not one, in three places** — spikes (unit-free vs Hz), latency (single-stream vs amortised), energy (total vs dynamic). In each case the pair answers two different questions, and collapsing them to one number silently picks an answer for the reader. Where the second figure is not knowable, report it as unavailable rather than substituting a default; see §2.2 and §2.4 for why that rule exists.

**Batch size and seed are recorded because they legitimately vary.** `calibrate_batch_size` sizes the batch from live VRAM, which is the mechanism that keeps it inside the card and that characterises the scalability constraints — so it differs by machine by design. Batch size is also the largest single lever on wall-clock time, so two rows can only be compared on speed once this column is known to match.

---

## References (for further reading)

- Comprehensive multimodal benchmark of neuromorphic training frameworks — ScienceDirect, 2025. https://www.sciencedirect.com/science/article/abs/pii/S0952197625015453
- From Lightweight CNNs to SpikeNets: pruned spiking SqueezeNet — arXiv. https://arxiv.org/pdf/2602.09717
- A Unified Evaluation Framework for SNN Hardware Accelerators (NVM devices) — arXiv, 2024. https://arxiv.org/html/2402.19139
- Evaluation of Encoding Schemes on Ubiquitous Sensor Signal for SNN (Loihi 2) — arXiv. https://arxiv.org/pdf/2407.09260
- Spiking Neural Network Architecture Search: A Survey — arXiv, 2025. https://arxiv.org/pdf/2510.14235
- A Practical Tutorial on Spiking Neural Networks — MDPI, 2025. https://www.mdpi.com/2673-4117/6/11/304
- Energy efficiency analysis of SNNs for space applications — arXiv, 2025. https://arxiv.org/pdf/2505.11418
- Evaluating SNN on Neuromorphic Platform for Human Activity Recognition — arXiv. https://arxiv.org/pdf/2308.00787
- Benchmarking ANN Architectures for High-Performance SNNs (NoC) — MDPI Sensors, 2024. https://www.mdpi.com/1424-8220/24/4/1329
- EFLOP: a sparsity-aware metric for evaluating computational cost — IOPscience, 2025. https://iopscience.iop.org/article/10.1088/2634-4386/addee8/meta
- Optimizing the Energy Consumption of SNNs for Neuromorphic Applications (SynOp-loss) — Frontiers in Neuroscience, 2020. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00662/full
- SpikeReg: Energy-Efficient 3D Deformable Medical Image Registration with SNNs — arXiv, 2026 (Horowitz per-op energy constants). https://arxiv.org/pdf/2605.25144
- PyTorch Profiler documentation / flops-profiler / DeepSpeed Flops Profiler — https://github.com/cli99/flops-profiler , https://www.deepspeed.ai/tutorials/flops-profiler/
- A Scalable Hybrid Training Approach for Recurrent Spiking Neural Networks (HYPR vs. e-prop) — arXiv, 2025. https://arxiv.org/pdf/2506.14464
- Benchmarking Spiking Neural Network Learning Methods with Varying Locality (BPTT vs. e-prop) — arXiv, 2024. https://arxiv.org/html/2402.01782v1
- SpiNNaker2: A Large-Scale Neuromorphic System (e-prop memory comparison) — arXiv, 2024. https://arxiv.org/pdf/2401.04491
- Towards Memory- and Time-Efficient Backpropagation for Training SNNs (SLTT) — ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/papers/Meng_Towards_Memory-_and_Time-Efficient_Backpropagation_for_Training_Spiking_Neural_Networks_ICCV_2023_paper.pdf
- Regulation of Irregular Neuronal Firing by Autaptic Transmission (CV_ISI definition) — arXiv. https://arxiv.org/pdf/1606.01684
- Gradient-Free Continual Learning in SNNs via Inter-Spike Interval Regularization (ISI-CV) — arXiv, 2026. https://arxiv.org/pdf/2604.16496
