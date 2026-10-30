# learning

Training, testing and metrics.

| File | Role |
|---|---|
| `main.py` | Entry point: loads settings and data, calibrates the batch size, trains, tests, writes the results |
| `training.py` | `SNNTrainer`: training loop with backpropagation through time, per-epoch metrics, GPU energy, training plots |
| `inference.py` | `SNNTester`: test pass, accuracy, per-class metrics, spike rate, throughput, SynOps energy estimate |
| `utilities.py` | Batch-size calibration, optimiser and loss factories, `ActivityMonitor`, SynOps helpers, single-sample latency, activation density |

```
python learning/main.py
python learning/main.py --framework sinabs
```

All settings come from `configuration/`; see `HOW_TO_RUN.md` for every option.
