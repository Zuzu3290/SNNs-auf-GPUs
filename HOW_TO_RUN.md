# How to run

There are two ways to run this pipeline and **both work**.

- **Flow A — the original way.** No flags at all. Config in `configuration/`, dataset
  chosen at the prompt, output to `./outputs`. Nothing about it changed.
- **Flow B — the experiment way.** One config file per experiment, output routed into
  its own folder, everything selectable from the command line. This is what Colab and
  multi-seed sweeps need.

Every flag is optional, and every default reproduces Flow A. You never have to learn
Flow B to use this pipeline.

---

## The one rule behind both flows

|                      | decides                                                             |
|----------------------|---------------------------------------------------------------------|
| the **config file**  | the experiment — neuron, architecture, framing, epochs, optimizer   |
| the **command line** | this run — which folder, which machine, which framework, which seed |

No path is ever written inside a config file, which is why the same experiment config
runs unchanged on a laptop and on Colab.

**Precedence:** `CLI argument` > `--config overlay` > the three base files > code default.

---
---

# Flow A — the original way, unchanged

## A1. Train and evaluate

```bash
python learning/main.py
```

Reads the three files in `configuration/`, asks which dataset at the prompt, trains,
evaluates, and writes to the `output:` paths in `SNN_module.yaml`
(`./outputs/data`, `./outputs/plots`).

This is exactly what it always did. The additions below are all opt-in.

### The same command, with every flag it accepts

```bash
python learning/main.py \
    --config experiments/ex2/config.yaml \
    --framework sinabs \
    --seed 3 \
    --experiment ex2 \
    --results-root /content/drive/MyDrive/snn_runs \
    --cache-root /content/cache
```

| arg | possible values | default | what it does |
|---|---|---|---|
| `--config` | path to a `.yaml` | *none* | overlay merged over the three base files. States only what differs. |
| `--framework` | `torch` · `norse` · `sj` · `sinabs` | `training.framework` from config | which SNN library runs |
| `--seed` | any integer | `training.seed` (`0`) | fixes weight init and batch order |
| `--experiment` | any folder name, e.g. `ex1` | *none* → `output:` paths | routes output into `<results-root>/<name>/` |
| `--results-root` | any path | `experiments` | where the experiment tree lives. **Requires `--experiment`** |
| `--cache-root` | any path | `cache.path` from config | where cached frames go |
| `-h`, `--help` | — | — | print this list |

There is deliberately **no `--device`**: it stays `training.device` in the config.

## A2. Check the network before spending GPU time

```bash
python check_network.py
```

Builds the network and prints every layer's output shape, the flatten width it
**measured** with a real forward pass, and the neuron each framework actually built.
No dataset, no download, no GPU — it runs on any laptop.

### The same command, with every flag it accepts

```bash
python check_network.py \
    --config experiments/ex2/config.yaml \
    --framework sinabs \
    --seed 1 \
    --experiment ex2 \
    --all \
    --batch 4 \
    --timesteps 16
```

| arg | possible values | default | what it does |
|---|---|---|---|
| `--config` | path to a `.yaml` | *none* | overlay, as above |
| `--framework` | `torch` · `norse` · `sj` · `sinabs` | from config | which one to inspect. Ignored with `--all`. |
| `--seed` | any integer | `0` | which seed's weights to fingerprint |
| `--experiment` | any name | *none* | **label only** — this script writes no files |
| `--all` | flag | off | compare all four instead of inspecting one |
| `--batch` | any integer | `4` | dummy batch size for the shape walk |
| `--timesteps` | any integer | `framing.n_time_bins` | T for the full forward pass |
| `-h`, `--help` | — | — | print this list |

No `--results-root` or `--cache-root`: it writes nothing and reads no cache.

## A3. Check the four neurons agree

```bash
python equivalence_check.py
```

Drives ONE neuron per framework with byte-identical input and compares the spike train
and membrane trajectory step by step. This is what makes the `neuron:` block in
`network_architecture.yaml` trustworthy — the block *claims* all four describe one
neuron, and nothing else enforces it.

**It measures and does not judge: no pass/fail, always exits 0.** One threshold cannot
serve both uses — ex1 forces the neurons to agree, ex2 deliberately varies one, so a
large deviation is a bug in the first case and the result in the second.

It also **draws** three stacked panels per input pattern — input current, membrane
trajectory, spike raster — so the divergence can be read as well as counted.

### The same command, with every flag it accepts

```bash
python equivalence_check.py \
    --config experiments/ex2/config.yaml \
    --experiment ex2 \
    --results-root /content/drive/MyDrive/snn_runs \
    --formats png,pdf
```

| arg | possible values | default | what it does |
|---|---|---|---|
| `--config` | path to a `.yaml` | *none* | overlay, as above |
| `--experiment` | any name, e.g. `ex2` | *none* → `outputs/plots` | figures go to `<results-root>/<name>/plots` |
| `--results-root` | any path | `experiments` | where that tree lives. **Requires `--experiment`.** Point at Drive on Colab. |
| `--formats` | `png`, `pdf`, `svg`, comma-separated | `png` | `png,pdf` gives a vector copy for a report |
| `-h`, `--help` | — | — | print this list |

No `--framework` (it builds all four — picking one would defeat the point), no `--seed`
(the poisson pattern carries its own, so every framework gets identical input), no
`--cache-root` (no dataset is touched), and no `--device` (CPU is hardcoded: one neuron
for 90 steps gains nothing from a GPU, and float non-determinism would undermine an
exact comparison).

Matplotlib runs on the `Agg` backend, so it works headless — a Colab cell or an SSH
session needs no display.

## A4. Run the unit tests

```bash
python tests/run_all.py
```

761 checks across 8 suites, CPU-only, no dataset, about 90 seconds. Exits non-zero if
anything fails, so it works as a pre-push gate.

| arg | possible values | default | what it does |
|---|---|---|---|
| *(positional)* | any substring of a suite filename, e.g. `adapters` | *all suites* | run only matching suites |

```bash
python tests/run_all.py adapters      # just unit_adapters.py
python tests/unit_adapters.py         # one suite, full PASS/FAIL detail
```

---
---

# Flow B — the merged pipeline: experiments, Colab, sweeps

Same scripts, same flags. What changes is that you write **one config file per
experiment** and let the command line decide where output goes.

## B1. Write the experiment config

One YAML per experiment, stating **only what differs**. Everything else is inherited
from the three base files.

```yaml
# experiments/ex2/config.yaml
dataset:
  name: N-MNIST          # names the dataset, so nothing ever waits at a prompt

neuron:
  sinabs:
    tau_mem: .inf        # no leak -- integrate-and-fire
    spike_fn: multi      # sinabs' own default: 2, 3, ... spikes per timestep
    reset_mechanism: subtract
    min_v_mem: -1.0

training:
  epochs: 1
```

It can reach **any key in any of the three base files** — they share no top-level
section name, so one flat overlay is unambiguous:

| section | lives in | example keys |
|---|---|---|
| `dataset` | `SNN_module.yaml` | `name` |
| `architecture`, `training`, `output` | `SNN_module.yaml` | `epochs`, `seed`, `optimizer.lr`, `use_amp` |
| `convolution`, `neuron_types`, `neuron` | `network_architecture.yaml` | `conv1_out`, `neuron.norse.tau_mem_inv` |
| `framing`, `temporal_slicing`, `augmentation`, `cache`, `resource_policy` | `data_workflow.yaml` | `n_time_bins`, `cache.path`, `batch_vram_fraction` |

An overlay may itself inherit from another, so a variant experiment does not copy the
one it varies:

```yaml
# config/ex4_multistep.yaml
extends: ex4_control.yaml
neuron:
  spikingjelly:
    step_mode: m
```

A **misspelled section raises** rather than being ignored — a typo'd `trianing:` would
otherwise leave the run on base values and the experiment quietly would not happen.

## B2. Verify the config before spending GPU time

Run these two first, in this order. Both are CPU-only and take seconds.

```bash
python check_network.py --config experiments/ex2/config.yaml --all
python equivalence_check.py --config experiments/ex2/config.yaml --experiment ex2
```

The first confirms all four frameworks start from **byte-identical weights** under one
seed — if they do not, no accuracy comparison between them means anything. The second
reports how far apart the four neurons actually are.

## B3. Run the experiment

```bash
python learning/main.py \
    --config experiments/ex2/config.yaml \
    --experiment ex2 \
    --framework sinabs \
    --seed 0
```

Output is routed into its own tree instead of `./outputs`:

```
experiments/ex2/
├── results/      training_results.csv, test.csv
├── equivalence/
└── plots/        training_metrics.png, vram_breakdown.png, spike_raster.png, ...
```

Repeat with a different `--framework` and the same `--seed` to compare frameworks; with
the same `--framework` and a different `--seed` to get replicates.

## B4. On Colab

Two things differ on Colab, and both are command-line flags — the config never changes.

```bash
python learning/main.py \
    --config experiments/ex1/config.yaml \
    --experiment ex1 \
    --framework norse \
    --seed 0 \
    --results-root /content/drive/MyDrive/snn_runs
```

**Point `--results-root` at mounted Drive.** `/content` dies with the runtime, and so
would your results. `--results-root` without `--experiment` is **refused** rather than
ignored, so a green run can never quietly write to `./outputs` and vanish.

**Set `dataset.name` in the config.** A notebook cell cannot answer a prompt. With it
set, you can launch several cells at once — each with its own `--framework` and
`--seed` — and none of them will block:

```bash
# cell 1
python learning/main.py --config experiments/ex1/config.yaml --experiment ex1 --framework norse  --seed 0 --results-root /content/drive/MyDrive/snn_runs
# cell 2
python learning/main.py --config experiments/ex1/config.yaml --experiment ex1 --framework sj     --seed 0 --results-root /content/drive/MyDrive/snn_runs
# cell 3
python learning/main.py --config experiments/ex1/config.yaml --experiment ex1 --framework sinabs --seed 0 --results-root /content/drive/MyDrive/snn_runs
```

Add `--cache-root` if the default cache location is short of space.

### Colab-relevant args

| arg | possible values | why it matters on Colab |
|---|---|---|
| `--results-root` | a mounted Drive path | results survive the runtime dying |
| `--cache-root` | any path with space | `/content` is small and ephemeral |
| `--experiment` | any name | required before `--results-root` will be accepted |
| `dataset.name` *(config, not a flag)* | see the table below | a cell cannot answer a prompt |

## B5. Choosing the dataset

`dataset.name` is `null` by default, which means **ask** — the interactive menu, as
always. Set it to skip the prompt.

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

## Reading the run banner

Every script prints the same block first, so a scrolled-back terminal still says which
config produced what follows:

```
==========================================================================
learning/main.py
  config        experiments/ex2/config.yaml   hash 4be0992923e9
  experiment    ex2
  framework     sinabs
  seed          3
  dataset       N-MNIST
  device        cuda
  writing to    experiments\ex2\results
  cli overrides framework=sinabs, seed=3
==========================================================================
```

`hash` is a digest of the fully-merged config, so two runs claiming the same experiment
can be proven to have used the same settings. `cli overrides` lists anything the command
line changed relative to the config, so nothing is hidden.
