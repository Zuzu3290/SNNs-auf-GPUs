# How to run

Every setting lives in the three YAML files in `configuration/`. The only command-line option is
`--framework`, which picks the SNN library for one run.

---

## 1. Install

```bash
# torch first, matching your machine
pip install torch==2.13.0 torchvision==0.28.0                                                  # CPU
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8

# tonic without its dependency check, then everything else
pip install --no-deps tonic==1.6.0
pip install -r requirements.txt
```

- **torch** is installed on its own because the right build differs per machine (CPU or CUDA).
  `requirements.txt` pins only the version, so one file serves both.
- **tonic** is installed with `--no-deps` because tonic 1.6.0 declares `numpy<2.0`, which has no
  build for current Python versions. Installing it together with the rest makes pip give up
  (`ResolutionImpossible`). The packages tonic actually needs are pinned in `requirements.txt`.

---

## 2. Configure

| File | What it controls |
|---|---|
| `configuration/SNN_module.yaml` | Dataset, framework, device, seed, epochs, optimiser, loss, output folders, checkpoint saving |
| `configuration/network_architecture.yaml` | Convolution blocks, pooling, hidden layers, and each framework's neuron parameters |
| `configuration/data_workflow.yaml` | Time bins per sample, denoising, cache location, DataLoader workers, GPU memory target for the batch size |

The settings most people change, all in `SNN_module.yaml`:

```yaml
dataset:
  name: N-MNIST        # null = ask at startup
training:
  framework: sj        # torch | norse | sj | sinabs
  device: cuda         # cuda | cpu
  seed: 0
  epochs: 2
output:
  save_checkpoint: true
```

### Datasets

Set `dataset.name` to a name or its number. Case, spaces, hyphens and underscores are ignored, so
`DVS128 Gesture` and `dvs128-gesture` are the same. If you leave it at `null`, the run asks at
startup.

| Value | Dataset | Classes | Download |
|---|---|---|---|
| `N-MNIST` or `1` | N-MNIST | 10 | 1.2 GB |
| `N-Caltech101` or `2` | N-Caltech101 | 101 | 3.7 GB |
| `DVS128 Gesture` or `3` | DVS128 Gesture | 11 | 3.0 GB |

Datasets download into `./tmp/data` on first use. A misspelled name stops the run immediately and
suggests the closest match:

```
dataset.name = 'N-MNSIT' is not a known dataset. Did you mean: N-MNIST?
```

The registry also lists **DAVIS Camera Pose** (`4`) and **Eye Tracking** (`5`). These are
regression datasets: their loaders work, but the pipeline only trains classifiers, so selecting
them fails. They are kept as a starting point for anyone adding a regression output.

**Adding your own dataset:** add an entry to `DATASET_REGISTRY` in
`event_data_workflow/dataset_registry.py` with its tonic class, sensor size and number of classes.
The input size, output layer and batch size adapt automatically. Events must come as
`(x, y, t, p)`: pixel column, pixel row, timestamp, polarity.

---

## 3. Run

```bash
python learning/main.py                       # framework from SNN_module.yaml
python learning/main.py --framework sinabs    # override the framework for this run
```

Each run prints the configuration and the network layer by layer, sizes the batch to fit the GPU,
trains, tests, and writes the results.

### Comparing frameworks

Run the same configuration once per framework. Every run adds one row to the same `runs.csv`:

```bash
for fw in torch norse sj sinabs; do python learning/main.py --framework $fw; done
```

All four start from identical weights for a given seed, so the neuron implementation is the only
thing that differs. For more seeds, change `training.seed` and repeat.

---

## 4. Results

Everything goes to the `output:` folders in `SNN_module.yaml`:

| File | Contents |
|---|---|
| `outputs/data/runs.csv` | One row per run: accuracy, throughput, single-sample latency, training speed, peak memory, GPU energy, SynOps energy, spike rates, environment |
| `outputs/data/epochs.csv` | One row per epoch per run: loss, accuracy, time, spike rate, latency, SynOps |
| `outputs/data/layers.csv` | One row per spiking layer per run: neurons, spikes, spike rate |
| `outputs/data/runs/<run_id>.json` | The full record of one run, including the neuron parameters actually built |
| `outputs/data/test.csv` | Per-class precision, recall, F1 and specificity for the latest run |
| `outputs/data/batch_metrics.csv` | One row per training iteration for the latest run |
| `outputs/data/model_checkpoint.pt` | Trained weights, when `save_checkpoint: true` |
| `outputs/plots/` | Training curves, per-iteration metrics, spike raster |

`runs.csv`, `epochs.csv` and `layers.csv` are appended to, never overwritten. If a code update
changes their columns, move the old files away before the next run.

The `config_hash` column identifies the exact settings behind each row: two rows with the same
hash used the same configuration.

---

## 5. Using a trained model

Set `output.save_checkpoint: true` **before** training. Each run trains from scratch, so weights
that were not saved are gone when the run ends.

To load the weights elsewhere, build the model from the **same configuration** it was trained
with (same framework, same architecture, same dataset shape):

```python
import torch
from skeleton.config import Settings, load_config
from frameworks.snn_model import SNNModel

cfg = Settings(load_config())                       # the configuration used for training
cfg.apply_dataset_shape(sensor_h=34, sensor_w=34, in_channels=2, num_classes=10)  # e.g. N-MNIST

model = SNNModel(cfg)                               # framework comes from training.framework
model.load_state(torch.load("outputs/data/model_checkpoint.pt"))
model.eval_mode()
# model(data) with data shaped [time, batch, channels, height, width]
```

If the architecture does not match, loading fails instead of silently loading wrong weights.

---

## 6. On Google Colab

A Colab runtime starts empty every session, so start with step 1. Then:

- set `dataset.name` in `SNN_module.yaml`, since a notebook cell cannot answer the startup prompt;
- point the `output:` folders at your mounted Google Drive, because `/content` is deleted when the
  runtime stops;
- if space is short, point `cache.path` in `data_workflow.yaml` at Drive as well.

```bash
python learning/main.py --framework norse
python learning/main.py --framework sj
```
