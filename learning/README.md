# learning

Training, testing and metrics.

| File | Role |
|---|---|
| `main.py` | Entry point: loads settings and data, calibrates the batch size, trains, tests, writes the results |
| `training.py` | `SNNTrainer`: training loop with backpropagation through time, per-epoch metrics, GPU energy, checkpoint |
| `inference.py` | `SNNTester`: test pass, accuracy, spike rate, latency, throughput, SynOps energy estimate |
| `utilities.py` | Batch-size calibration, optimiser and loss factories, `ActivityMonitor`, SynOps and CV-ISI helpers |

```
python learning/main.py
python learning/main.py --framework sinabs --seed 1
```

All settings come from `configuration/`; see `HOW_TO_RUN.md` for every option.
