# Learning

Training, inference, and evaluation pipeline for Spiking Neural Networks.

## Structure

```
learning/
├── main.py                        # Entry point — runs training then inference
├── training.py                    # SNNTrainer: training loop, checkpointing, GPU stats
├── inference.py                   # SNNTester: evaluation, per-class metrics, energy estimate
├── adversarial_robustness.py      # TRADES adversarial evaluation
├── frameworks/
│   ├── snn_norse.py               # SNN model built with Norse (LIFCell)
│   ├── snn_spikingjelly.py        # SNN model built with SpikingJelly (IzhikevichNode)
│   ├── snn_torch.py               # SNN model built with SNNTorch (Leaky)
│   └── activity_reg.py            # Activity regularisation + STDP loss hooks
└── event_data_workflow/
    ├── data_pipeline.py           # NeuromorphicEncoder: loads dataset, builds DataLoaders
    ├── cache_engine.py            # Adaptive cache (RAM/disk) + PipelineMemoryCoordinator
    └── temporal_slicer.py         # Temporal slicing of event recordings
```

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
