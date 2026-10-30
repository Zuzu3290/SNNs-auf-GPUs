# How to run

Every setting is in the three files in `configuration/`. The command line only chooses the
framework, the seed, the cache folder and an optional overlay file; output always goes to the
`output:` paths in `SNN_module.yaml` (`./outputs/data`, `./outputs/plots`).

---

# Step 0 — install

## Installing from scratch

```bash
pip install torch==2.13.0 torchvision==0.28.0        # CPU
# ...or, for CUDA 12.8:
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu128

pip install --no-deps tonic==1.6.0
pip install --no-deps --ignore-requires-python bindsnet==0.3.4
pip install -r requirements.txt
```

torch is installed first and separately because its build differs per machine.
`requirements.txt` pins the **version** (`2.13.0`), not the wheel tag (`+cpu`, `+cu128`),
so one file serves both.

tonic is also installed first and separately, with `--no-deps`: tonic 1.6.0 (its own
latest release) declares `numpy<2.0.0`, which has no wheel on Python 3.13+, so asking
pip to resolve it together with `requirements.txt`'s `numpy==2.3.5` in one command
fails outright (`ResolutionImpossible`). `--no-deps` skips that check; tonic's actual
runtime dependencies are pinned in `requirements.txt` instead. See that file's "Event
data" section for the full story.

bindsnet (the unsupervised STDP track) is installed the same way, for a similar reason: 0.3.4
declares Python `<3.14` and `torch>=2.14`, while this project runs Python 3.14 and pins torch
2.13 for the framework comparison. Both were checked rather than assumed: on torch 2.13 / Python
3.14 it imports and its STDP rule updates weights on the GPU. `--ignore-requires-python` is
needed because pip otherwise hides every release whose declared Python range excludes 3.14.
Its import-time dependencies (gymnasium, ale-py, scikit-image, tensorboardX) are plain pins in
`requirements.txt`.

---

# 1. Train and evaluate

```bash
python learning/main.py
```

Reads the three files in `configuration/`, asks which dataset at the prompt, prints the
configuration and the network layer by layer, calibrates the batch size to the GPU, trains,
evaluates, and writes the results to `./outputs`.

### Before you run: do you want the trained model saved?

**Decide this before you run, not after** — the model is trained fresh every time
`learning/main.py` runs, and by default nothing is kept once the process exits, so
if you skip this and want the weights afterward, the only fix is to train again.

Set `output.save_checkpoint` in `SNN_module.yaml` (or an overlay):

```yaml
output:
  save_checkpoint: true   # false by default -- opt in
```

**true** if you want to reuse these exact trained weights afterward — resume
training later, run inference in a separate script, inspect the weights, and so
on. **false** (the default) if this run is only about the numbers it prints and
writes to `runs.csv`/`test.csv` — the common case for a framework-comparison run,
where keeping every run's weights on disk is real, usually-unneeded disk usage.

When on, it writes `model_checkpoint.pt` (model + optimizer state, framework name,
neuron config) into the run's results folder, right after training finishes and
before inference starts — inference itself is unaffected either way, and still
runs immediately after in the same invocation.

**Using the saved weights elsewhere** — a separate script, a notebook, anywhere
outside this run. You need three things: the same model class the checkpoint was
trained with (`training.framework` in the config that produced it — `SNN_SJ` for
`sj`, etc., see `learning/main.py`'s `FRAMEWORK_MODULES`), a `Settings` built from
that SAME config (architecture must match, or the weights won't fit the layers),
and the checkpoint file's path:

```python
from skeleton import Settings
from frameworks.snn_spikingjelly import SNN_SJ   # match the framework the checkpoint was trained with
import torch

cfg = Settings()   # the SAME config the checkpoint was trained under
cfg.apply_dataset_shape(...)   # same dataset shape as the training run

model = SNN_SJ(cfg)
model.load_state(torch.load("path/to/model_checkpoint.pt"))
model.eval_mode()
# model(data) now runs inference with the trained weights
```

If the architecture (dataset shape, layer sizes, framework) doesn't match what the
checkpoint was trained with, `load_state` raises rather than silently loading
mismatched weights.

### Flags

```bash
python learning/main.py --config my_overlay.yaml --framework sinabs --seed 3 --cache-root /content/cache
```

| arg | possible values | default | what it does |
|---|---|---|---|
| `--config` | path to a `.yaml` | *none* | overlay merged over the three base files; states only what differs |
| `--framework` | `torch` · `norse` · `sj` · `sinabs` | `training.framework` | which SNN library runs |
| `--seed` | any integer | `training.seed` (`0`) | fixes weight init and batch order |
| `--cache-root` | any path | `cache.path` | where cached frames go |
| `-h`, `--help` | — | — | print this list |

**Precedence:** command-line flag > `--config` overlay > the three base files.

An overlay is a YAML file with only the keys that change, for example:

```yaml
dataset:
  name: N-MNIST          # names the dataset, so nothing waits at a prompt
training:
  epochs: 1
```

The configuration block printed at the start of every run shows the overlay path and a hash of
the merged config; the same hash is written to `runs.csv`, so two rows can be proven to come
from the same settings.

### Choosing the dataset

`dataset.name` is `null` by default, which means **ask** at the prompt. Set it to skip the prompt.

| value | dataset | classes |
|---|---|---|
| `N-MNIST` or `1` | N-MNIST | 10 |
| `N-Caltech101` or `2` | N-Caltech101 | 101 |
| `DAVIS Camera Pose` or `3` | DAVIS Camera Pose | *regression* |
| `DVS128 Gesture` or `4` | DVS128 Gesture | 11 |
| `DSEC` or `5` | DSEC | *regression* |
| `Eye Tracking` or `6` | 3ET Eye Tracking | *regression* |

Case, spaces, hyphens and underscores are ignored — `DVS128 Gesture`,
`dvs128-gesture` and `dvs128_gesture` are the same. An unrecognised name **raises at
startup** and suggests the closest match:

```
dataset.name = 'N-MNSIT' is not a known dataset. Did you mean: N-MNIST?
```

It never falls back to a default. The three regression datasets are recognised but
`learning/main.py` is classification-only and will say so.

---

# 2. Check the four neurons agree

```bash
python equivalence_check.py
```

Drives ONE neuron per framework with byte-identical input and compares the spike train and
membrane trajectory step by step against snnTorch. It measures and does not judge: no pass/fail,
always exits 0. It also draws three stacked panels per input pattern (input current, membrane
trajectory, spike raster).

| arg | possible values | default | what it does |
|---|---|---|---|
| `--config` | path to a `.yaml` | *none* | overlay, as above |
| `--out` | any path | `outputs/equivalence` | folder for the figures |
| `--formats` | `png`, `pdf`, `svg`, comma-separated | `png` | `png,pdf` gives a vector copy for a report |

It always runs on the CPU: one neuron for 90 steps gains nothing from a GPU.

---

# 3. On Colab

A Colab runtime is a new machine every session, so it starts with Step 0. Then:

- set `dataset.name` in an overlay, since a notebook cell cannot answer a prompt;
- point `output:` (in the same overlay) at mounted Drive, since `/content` disappears with the
  runtime;
- add `--cache-root` if the default cache location is short of space.

```bash
python learning/main.py --config colab.yaml --framework norse --seed 0
python learning/main.py --config colab.yaml --framework sj     --seed 0
```
