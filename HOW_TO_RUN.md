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
| `configuration/SNN_module.yaml` | Framework, training and inference device, seed, epochs, optimiser, loss, output folders, checkpoint saving |
| `configuration/network_architecture.yaml` | Convolution blocks, pooling, hidden layers, and each framework's neuron parameters |
| `configuration/data_workflow.yaml` | Time bins per sample, denoising, augmentation, cache location, DataLoader workers, GPU memory target for the batch size |

The settings most people change, all in `SNN_module.yaml`:

```yaml
training:
  framework: sj        # torch | norse | sj | sinabs
  device: cuda         # training runs on a CUDA GPU
  seed: 0
  epochs: 2
  calibrate_batch_size: true   # true = batch size fitted to the GPU; batch_size is used only when false
inference:
  device: null         # cuda | cpu | null = ask at startup
output:
  save_checkpoint: true
```

Training needs a CUDA GPU (the batch size is fitted to its memory). Testing, single-sample latency
and activation density run on the inference device, so a model trained on the GPU can also be
evaluated on the CPU. GPU energy and memory are only meaningful when inference runs on `cuda`.

### Designing your own network

Everything about the network is in `network_architecture.yaml` and needs no code changes:

- `convolution.blocks`: one entry per Conv → LIF → MaxPool block (`out` channels, `kernel` size);
  add or remove entries to make the network deeper or shallower.
- `convolution.fc_hidden`: a list of hidden layer widths before the output, e.g. `[256]`; `null` for none.
- `convolution.batch_norm`: BatchNorm between each convolution and its neuron.
- `neuron.<framework>`: the LIF parameters for each library (decay, threshold, reset, surrogate gradient).
  Sinabs accepts five surrogates; the parameters each one needs are the names its constructor uses.

The input size and the number of outputs come from the dataset, so the same configuration works on
every dataset.

### Augmentation

`augmentation.enabled: true` in `data_workflow.yaml` rotates every training clip by a random angle
up to `rotation_degrees`, one angle for all frames of a clip, freshly drawn each epoch. It is
applied after the cache, so the cached frames are unchanged, and never to the test set.

### Datasets

Each run asks for the dataset at startup. Answer with its number:

| Answer | Dataset | Classes | Download |
|---|---|---|---|
| `1` | N-MNIST | 10 | 1.2 GB |
| `2` | N-Caltech101 | 101 | 3.7 GB |
| `3` | DVS128 Gesture | 11 | 3.0 GB |

Datasets download into `./tmp/data` on first use. A number that is not on the menu stops the run
immediately.

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

Each run asks for the inference device (when `inference.device` is `null`) and then the dataset,
prints the configuration and the network layer by layer, sizes the batch to fit the GPU, trains,
tests, and writes the results.

The answers can also be piped in, one per line in that order, so a run needs nobody at the
keyboard. With `inference.device` set in the config, only the dataset is asked:

```bash
echo 1 | python learning/main.py --framework sinabs           # N-MNIST
printf "1\n3\n" | python learning/main.py --framework sinabs  # inference on cuda, DVS128 Gesture
```

### Comparing frameworks

Run the same configuration once per framework. Every run adds one row to `training_runs.csv` and
one to `inference_runs.csv`:

```bash
for fw in torch norse sj sinabs; do echo 1 | python learning/main.py --framework $fw; done
```

All four start from identical weights for a given seed, so the neuron implementation is the only
thing that differs. For more seeds, change `training.seed` and repeat.

---

## 4. Results

Everything goes to the `output:` folders in `SNN_module.yaml`:

| File | Contents |
|---|---|
| `outputs/data/training_runs.csv` | One row per training run: settings, architecture, training loss and time, forward/backward latency, spike rate, peak memory, GPU energy, environment |
| `outputs/data/inference_runs.csv` | One row per test pass: inference device, accuracy, throughput, single-sample latency, activation density, SynOps per sample, peak memory, GPU energy, environment |
| `outputs/data/epochs.csv` | One row per epoch per run: loss, accuracy, time, spike rate, forward/backward latency |
| `outputs/data/layers.csv` | One row per spiking layer per run: neurons, spikes, spike rate |
| `outputs/data/runs/<run_id>.json` | The full record of one run, including the neuron parameters and the inference device |
| `outputs/data/batch_metrics.csv` | One row per training iteration for the latest run: loss, accuracy, spike rate |
| `outputs/data/model_checkpoint.pt` | Trained weights, when `save_checkpoint: true` |
| `outputs/plots/` | Training curves (loss, accuracy and spike rate per iteration) and a spike raster of one test sample |

The two run files share `run_id`: join them on it to put a run's training and test figures side by
side. When inference runs on the CPU, the GPU memory and energy columns of `inference_runs.csv` are
left empty, since the GPU was only idling.

`training_runs.csv`, `inference_runs.csv`, `epochs.csv` and `layers.csv` are appended to, never
overwritten. If a code update changes their columns, move the old files away before the next run.
Results written before the split are in `runs.csv`, which is no longer written to.

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
from frameworks.snntorch_lif import Neuron   # the framework the checkpoint was trained with

cfg = Settings(load_config())                       # the configuration used for training
cfg.apply_dataset_shape(sensor_h=34, sensor_w=34, in_channels=2, num_classes=10)  # e.g. N-MNIST

model = SNNModel(cfg, Neuron)
model.load_state_dict(torch.load("outputs/data/model_checkpoint.pt")["model_state_dict"])
model.eval()
# model(data) with data shaped [time, batch, channels, height, width]
```

If the architecture does not match, loading fails instead of silently loading wrong weights.

---

## 6. On Google Colab

A Colab runtime starts empty every session, so start with step 1. Then:

- set `inference.device` in `SNN_module.yaml` and pipe the dataset number in (below), since a
  notebook cell cannot answer the startup prompts;
- point the `output:` folders at your mounted Google Drive, because `/content` is deleted when the
  runtime stops;
- if space is short, point `cache.path` in `data_workflow.yaml` at Drive as well.

```bash
echo 1 | python learning/main.py --framework norse
echo 1 | python learning/main.py --framework sj
```
