# SNNs-auf-GPUs

A research platform for running Spiking Neural Networks on GPU hardware.
Trains and compares SNN framework backends on real event-camera datasets,
measuring runtime, scalability, and accuracy.

---

## What This Is

Neuromorphic computing on commodity GPUs. The project bridges event-based
sensor data (DVS / DAVIS cameras) with SNN training frameworks.

| Location | Role |
|----------|------|
| `src/learning/`, `event_data_workflow/`, `skeleton/` | Training loop, framework wrappers, data pipeline, configuration |

`SNNTrainer`/`SNNTester` run the plain per-framework PyTorch path.

---

## Project Layout

```
src/
  learning/         SNN framework wrappers — SNNTorch, Norse, SpikingJelly, Sinabs
skeleton/           Configuration and settings (SNN_module.yaml)
event_data_workflow/ Neuromorphic data pipeline — caching, slicing, DataLoader
docs/               Architecture references and hardware notes
```

---

## Entry Point

```
python src/learning/main.py
```

Reads `SNN_module.yaml`, loads the neuromorphic dataset via
`event_data_workflow`, builds the SNN model, runs training, then evaluation.

---

## Configuration

All runtime parameters live in `SNN_module.yaml` at the project root:
architecture, training schedule, dataset path, device, and data pipeline
settings. No hardcoded values in source files.

---

## Framework Backends

The trainer and inference pipeline are framework-agnostic at the model boundary.
Any model that satisfies the `ModelInterface` contract can be plugged in — the
pipeline does not care what runs inside.

| Backend | Status | Notes |
|---------|--------|-------|
| SNNTorch | Working | Default |
| Norse | Working | Current default in `main.py` |
| SpikingJelly | Working | |
| JAX + Flax/Haiku | Extension point | Trains via XLA; DLPack bridge to PyTorch at boundary |
| TensorFlow | Extension point | DLPack bridge at boundary |
| Custom / from scratch | Extension point | Return a PyTorch tensor — everything else is your choice |

For details on how each backend cooperates with the training loop, backward pass,
and adversarial evaluation, see [`docs/frameworks/`](docs/frameworks/).

---

## Current Capabilities

- Four SNN backends: SNNTorch, Norse, SpikingJelly, Sinabs — switchable via config
- Six event-camera datasets, selectable from the terminal at runtime — see
  `event_data_workflow/data_pipeline.py`'s `DATASET_REGISTRY`
- Adaptive data pipeline: selects memory, disk, hybrid, or GPU-VRAM cache
  strategy automatically based on available system resources
- Activity regularization and STDP as differentiable loss terms alongside BPTT
- Adversarial robustness evaluation via TRADES

---

## Growing Analytics

Diagnostic benchmark runs across framework backends are ongoing, with results
and plots accumulating per-dataset in [`docs/results/`](docs/results/)
(`docs/results/plots/<dataset>/`). Latest snapshot (N-MNIST):

| Training Accuracy | Actual GPU Energy (Training) |
|---|---|
| ![accuracy curves](docs/results/plots/n_mnist/accuracy_curves.png) | ![training energy](docs/results/plots/n_mnist/train_energy.png) |

| Spike Rate | Inference Latency per Sample |
|---|---|
| ![spike rate curves](docs/results/plots/n_mnist/spike_rate_curves.png) | ![test latency](docs/results/plots/n_mnist/test_latency.png) |

See [`docs/results/README.md`](docs/results/README.md) for the full results
table, all plots, caveats on what this run does and doesn't measure, and the
real bugs this benchmarking work has already found and fixed.

