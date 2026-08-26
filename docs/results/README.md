# Framework Diagnostic Benchmark

A real, end-to-end run of the SNN framework backends (Norse, snnTorch,
SpikingJelly, Sinabs) on the real cached N-MNIST dataset, via
the actual training/inference pipeline (`NeuromorphicEncoder` → `SNNTrainer` →
`SNNTester`) — not synthetic data, not isolated unit tests. Produced 3 real bug
fixes (see below) and the plots/data in this folder.

## Reproduce

```bash
python docs/results/run_benchmark.py   # trains + tests all backends, writes docs/results/data/*.json
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
those 10 batches). TRADES and the custom CUDA kernel are both disabled (see
`run_benchmark.py`'s docstring for why).

## Results

| Framework | Final loss | Train acc | Test acc | Energy/sample (pJ, neuromorphic est.) | Latency/sample (ms) | Firing rate (Hz) | Train time (s) | **Actual GPU energy (J)** | **Actual avg power (W)** |
|---|---|---|---|---|---|---|---|---|---|
| Norse | 1.219 | 59.4% | 73.7% | 12.80 | 0.516 | 3120.7 | 79.6 | 526.5 | 6.6 |
| snnTorch | 0.522 | 13.3% | 78.6% | 101.23 | 1.085 | 24682.0 | 15.8 | 255.1 | 16.2 |
| SpikingJelly | 1.987 | 28.9% | 58.9% | 6.22 | 1.001 | 1516.7 | 16.3 | 257.4 | 15.8 |
| Sinabs | 2.303 | 14.8% | 76.6%¹ | 0.00¹ | 0.286 | 0.0¹ | 40.7 | 327.6 | 8.1 |

¹ Dead network (see caveat above) — not a meaningful accuracy/energy number.

The "Energy/sample" columns are the project's per-spike neuromorphic-hardware
*estimate* (`ENERGY_PER_SPIKE_PJ × spike count` in `inference.py`) — a model of
what a real neuromorphic chip would use, not what this GPU actually drew. The
**actual GPU energy/power** columns are real measurements (`GPUStats`/NVML,
captured by `SNNTrainer` during training) — note Norse draws the *least* power
(6.6W) but the *most total energy* (526.5J), simply because it ran longest.

Norse takes noticeably longer per run, because of its `LIFCell`'s explicit
per-timestep state-tuple handling.

## Plots

- `plots/loss_curves.png`, `plots/accuracy_curves.png`, `plots/spike_rate_curves.png` — all backends overlaid per training iteration
- `plots/test_accuracy.png`, `plots/test_energy.png`, `plots/test_latency.png`, `plots/test_firing_rate.png`, `plots/train_time.png` — per-framework bars
- `plots/train_energy.png`, `plots/train_power.png` — **actual NVML-measured** GPU energy (J) and average power (W) during training, pulled from `{name}_train.csv` (not the neuromorphic-hardware estimate used by the test-phase energy plot)
- `plots/confusion_matrices.png` — grid, one per framework
