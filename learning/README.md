# learning

Training, testing and metrics.

| File | Role |
|---|---|
| `main.py` | Entry point: loads settings and data, calibrates the batch size, trains, saves the checkpoint, tests, measures single-sample latency and activation density, writes the results |
| `training.py` | `SNNTrainer`: training loop with backpropagation through time, per-iteration and per-epoch metrics, GPU energy, training curves |
| `inference.py` | `SNNTester`: test pass on the chosen inference device (cuda or cpu), accuracy, spike rate, throughput, SynOps energy estimate, spike raster of one test sample |
| `utilities.py` | The spike-count loss, timing and SynOps helpers, `SpikeRecorder` for the activity penalty, warm-up, single-sample latency, activation density, batch-size calibration |

```
python learning/main.py
python learning/main.py --framework sinabs
```

At startup it asks for the inference device (only when `inference.device` is null in the config), then the dataset.
Pipe the answers to run without typing, e.g. `echo 1 | python learning/main.py --framework sinabs` (dataset 1, device set in the config)
or `printf "2\n1\n" | python learning/main.py` (inference on cpu, then dataset 1).

All settings come from `configuration/`; see `HOW_TO_RUN.md` for every option.
