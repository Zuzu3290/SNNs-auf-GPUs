# Framework Diagnostic Benchmark — All 6 Backends

A real, end-to-end run of all 6 SNN framework backends (Norse, snnTorch,
SpikingJelly, Sinabs, BindsNET, Spyx) on the real cached N-MNIST dataset, via
the actual training/inference pipeline (`NeuromorphicEncoder` → `SNNTrainer` →
`SNNTester`) — not synthetic data, not isolated unit tests. Produced 3 real bug
fixes (see below) and the plots/data in this folder.

## Reproduce

```bash
python docs/results/run_benchmark.py   # trains + tests all 6, writes docs/results/data/*.json
python docs/results/make_plots.py      # reads that data, writes docs/results/plots/*.png
```

## This is a diagnostic run, not a quality benchmark

**EPOCHS=1, ITERA=20** (20 training batches), test capped at 10 batches. That's
enough to confirm every backend runs correctly end-to-end and to compare *shape*
of behavior (loss trend, energy/latency order of magnitude), but nowhere near
enough to train a real model — none of these accuracy numbers reflect how good
a framework "is." Sinabs in particular ends this run with **zero output spikes**
(a dead network — loss flatlines at `ln(10) ≈ 2.303`, the random-guess baseline)
and its 76.6% test accuracy is purely an artifact of the small test slice's class
imbalance (it predicts class 0 for everything, and class 0 happens to dominate
those 10 batches). BindsNET's loss also doesn't trend down — expected, since its
weights update via STDP, not the cross-entropy loss the trainer logs for display.
TRADES and the custom CUDA kernel are both disabled (see `run_benchmark.py`'s
docstring for why).

## Results

| Framework | Final loss | Train acc | Test acc | Energy/sample (pJ, neuromorphic est.) | Latency/sample (ms) | Firing rate (Hz) | Train time (s) | **Actual GPU energy (J)** | **Actual avg power (W)** |
|---|---|---|---|---|---|---|---|---|---|
| Norse | 1.219 | 59.4% | 73.7% | 12.80 | 0.516 | 3120.7 | 79.6 | 526.5 | 6.6 |
| snnTorch | 0.522 | 13.3% | 78.6% | 101.23 | 1.085 | 24682.0 | 15.8 | 255.1 | 16.2 |
| SpikingJelly | 1.987 | 28.9% | 58.9% | 6.22 | 1.001 | 1516.7 | 16.3 | 257.4 | 15.8 |
| Sinabs | 2.303 | 14.8% | 76.6%¹ | 0.00¹ | 0.286 | 0.0¹ | 40.7 | 327.6 | 8.1 |
| BindsNET | 2.303² | 8.6% | 76.6%¹ | 105.00 | 0.877 | 25600.0 | 16.3 | 288.1 | 17.7 |
| Spyx | 2.320 | 6.2% | 64.5% | 425.54 | 2.944 | 103751.3 | 81.0 | 387.8 | 4.8 |

¹ Dead network (see caveat above) — not a meaningful accuracy/energy number.
² Loss doesn't drive learning for this backend; see caveat above.

The "Energy/sample" columns are the project's per-spike neuromorphic-hardware
*estimate* (`ENERGY_PER_SPIKE_PJ × spike count` in `inference.py`) — a model of
what a real neuromorphic chip would use, not what this GPU actually drew. The
**actual GPU energy/power** columns are real measurements (`GPUStats`/NVML,
captured by `SNNTrainer` during training) — note Norse draws the *least* power
(6.6W) but the *most total energy* (526.5J), simply because it ran longest;
Spyx is the inverse (lowest power, third-most energy, due to its long
trace/compile-dominated wall time on CPU-only JAX).

Norse and Spyx take noticeably longer per run — Norse because of its
`LIFCell`'s explicit per-timestep state-tuple handling, Spyx because every
batch round-trips through DLPack to JAX and back, plus JAX's CPU-only
trace/compile overhead on this machine (no Windows CUDA wheels — see
`docs/frameworks/additional_frameworks.md`).

## Plots

- `plots/loss_curves.png`, `plots/accuracy_curves.png`, `plots/spike_rate_curves.png` — all 6 overlaid per training iteration
- `plots/test_accuracy.png`, `plots/test_energy.png`, `plots/test_latency.png`, `plots/test_firing_rate.png`, `plots/train_time.png` — per-framework bars
- `plots/train_energy.png`, `plots/train_power.png` — **actual NVML-measured** GPU energy (J) and average power (W) during training, pulled from `{name}_train.csv` (not the neuromorphic-hardware estimate used by the test-phase energy plot)
- `plots/confusion_matrices.png` — grid, one per framework

## Bugs this run found and fixed

Running the real pipeline (vs. isolated forward-pass checks) surfaced 4 real
bugs, all fixed in this session:

1. **`SNN_SPYX.optimizer` name collision** — `SNNTrainer` builds a
   `CosineAnnealingLR(model.optimizer, ...)` scheduler for any model with an
   `optimizer` attribute. Spyx's attribute held an optax `GradientTransformation`,
   not a torch optimizer — would have crashed on scheduler construction. Renamed
   to `optax_opt`.
2. **`SNN_BINDSNET` / `SNN_SPYX` not callable** — neither inherits `nn.Module`,
   but `training.py`/`inference.py` call models as `model(data)`, relying on
   `nn.Module.__call__`. Added explicit `__call__` methods to both.
3. **Sinabs broke `stdp_regularization`** — `activity_reg.py`'s hooks assume a
   layer is called once *per timestep*; Sinabs calls its LIF layers once per
   forward() with the whole `(B,T,...)` tensor, so the hook recorded one
   timestep containing all of T, producing a `1×1` correlation kernel that
   couldn't multiply against the real per-timestep output trace. Sinabs no
   longer registers activity hooks (documented in `snn_sinabs.py`).
4. **`SNNTester.forward_pass()` (and `AdversarialEvaluator`) missing the
   `tensor_format() == "BT"` transpose** — `SNNTrainer.forward_pass()` already
   had this check; inference and adversarial eval didn't. Harmless for the
   original 3 time-first backends, but silently fed Sinabs transposed
   `(T,B,...)` data labeled as `(B,T,...)` during testing — batch and time got
   swapped, and `aggregate_spike_output` summed over the wrong axis. Surfaced
   as `RuntimeError: size of tensor a (16) must match tensor b (128)` — 16
   being that batch's real (variable, event-count-dependent) timestep count,
   not a fixed dimension. Fixed in both `inference.py` and
   `adversarial_robustness.py`.

None of these were visible from isolated `model.forward(dummy_data)` checks —
all 4 only showed up by running the real `SNNTrainer`/`SNNTester` loop end to end.
