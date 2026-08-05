# Learning

Training, inference, and evaluation pipeline for Spiking Neural Networks.

## Structure

```
learning/
├── main.py                        # Entry point — runs training then inference
├── training.py                    # SNNTrainer: training loop, checkpointing, GPU stats
├── inference.py                   # SNNTester: evaluation, per-class metrics, energy estimate
├── adversarial_robustness.py      # TRADES adversarial evaluation
├── utilities.py                    # build_optimizer/build_loss factories, ActivityMonitor + DenseTimestepBuffer
└── frameworks/
    ├── model_interface.py          # ModelInterface ABC — contract every backend must implement
    ├── snn_norse.py                # SNN model built with Norse (LIFCell)
    ├── snn_spikingjelly.py         # SNN model built with SpikingJelly (IzhikevichNode)
    ├── snn_torch.py                # SNN model built with SNNTorch (Alpha)
    ├── snn_sinabs.py               # SNN model built with Sinabs (DVS-first, batch-first tensors)
    └── personal/                   # gitignored/local-only DSEC regression variants
```

Per-layer spike recording + activity regularization (`ActivityMonitor`,
`DenseTimestepBuffer`) live in `utilities.py`, attached in each model's
`__init__` as `self.activity`. STDP regularization was removed — not needed
without an unsupervised-learning application, and the neuron-comparison work
it supported is out of scope. `activity_reg.py` no longer exists.

`event_data_workflow/` is a sibling package at the project root (not nested
under `learning/`), imported as `from event_data_workflow...`. See
[`event_data_workflow/README.md`](../event_data_workflow/README.md) —
`data_pipeline.py` builds DataLoaders and includes DataLoader worker sizing
and temporal slicing (folded in from since-removed `pipeline_coordinator.py`/
`temporal_slicer.py`); `cache_engine.py` is the adaptive RAM/disk/VRAM cache;
`prefetch.py` holds the async batch-prefetch thread.

## Quick Start

```bash
python src/learning/main.py
```

All hyperparameters are in `SNN_module.yaml` at the project root.

## Key Features

- **Three framework backends** — Norse, SpikingJelly, SNNTorch, switchable from `main.py`
- **TRADES robustness training** — adversarial perturbation during training
- **Activity regularisation** — prevents dead and saturated neurons
- **STDP loss** — biologically inspired spike-timing correlation term
- **Adaptive caching** — automatically picks RAM, disk, or hybrid caching based on available resources
- **GPU stats** — per-epoch VRAM usage tracked during training
