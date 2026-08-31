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

# Step 0 — check the environment first

Run this once on every new machine — your laptop, a fresh Colab runtime, a colleague's
GPU box — **before** spending time on a run.

```bash
python check_env.py
```

It prints the version of every dependency, the CPU and RAM, the GPU and whether NVML can
read power from it, then a verdict comparing what is installed against the `==` pins in
`requirements.txt`.

```
====================================================================
VERDICT
====================================================================
OK: all packages import, and every pinned version matches requirements.txt
```

**Why this is not optional here.** This pipeline compares SNN *frameworks*. If Colab
resolves `snntorch` to a different version than the laptop did, part of the difference
between two runs is a difference between versions — and nothing in `runs.csv` says which
part. So a drift is reported loudly:

```
version mismatches against requirements.txt:
  snntorch             wanted 1.0.0        got 0.9.1  RESULT-CRITICAL
```

`RESULT-CRITICAL` marks the seven pins that decide the numbers: the four frameworks plus
`torch`, `tonic` and `numpy`. The rest may drift without changing a result.

### The same command, with every flag it accepts

```bash
python check_env.py --strict
```

| arg | possible values | default | what it does |
|---|---|---|---|
| `--strict` | flag | off | exit non-zero on a version **mismatch** too, not only on a missing package |
| `-h`, `--help` | — | — | print this list |

Exit codes, so it can gate a script or a notebook cell:

| situation | plain | `--strict` |
|---|---|---|
| everything matches | `0` | `0` |
| a package is missing | `1` | `1` |
| all present, a version differs | `0` | `1` |

Two lines in its output are worth reading beyond the verdict:

- **`cpu_cores_physical`** — a Colab runtime with 1 physical core has been measured
  holding GPU utilisation near 11%, the loader unable to keep up. A run in that state
  times the data pipeline rather than the framework, so the script warns at ≤ 2 cores.
- **`nvml_power_readable`** — if this says `NO`, the energy columns in `runs.csv` will be
  empty on this machine. Better to know before the run than after it.

`samna` is read from pip metadata and deliberately never imported: importing it makes
sinabs try to install it from a private index. It is only needed to drive real Speck
hardware, so `not installed (optional)` is the normal answer.

## Installing from scratch

```bash
pip install torch==2.13.0 torchvision==0.28.0        # CPU
# ...or, for CUDA 12.8:
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
python check_env.py
```

torch is installed first and separately because its build differs per machine.
`requirements.txt` pins the **version** (`2.13.0`), not the wheel tag (`+cpu`, `+cu128`),
so one file serves both; `check_env.py` compares the same way and ignores the tag.

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

980 checks across 9 suites, CPU-only, no dataset, about a minute. Exits non-zero if
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

Run these three first, in this order. All are CPU-only and take seconds.

```bash
python check_env.py
python check_network.py --config experiments/ex2/config.yaml --all
python equivalence_check.py --config experiments/ex2/config.yaml --experiment ex2
```

The first proves this machine's library versions match the ones every other run used —
see Step 0. The second confirms all four frameworks start from **byte-identical weights**
under one seed; if they do not, no accuracy comparison between them means anything. The
third reports how far apart the four neurons actually are.

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
├── config.yaml                              the experiment, beside its README
├── README.md
├── results/
│   ├── runs.csv  epochs.csv  layers.csv     APPEND-ONLY, every run of this experiment
│   ├── runs/<run_id>.json                   full record, one per run
│   └── <run_id>/                            this run only
│       └── training_results.csv  batch_metrics.csv  test.csv
├── plots/<run_id>/                          this run's 7 diagnostics
├── figures/                                 make_plots.py -- the cross-run comparison
└── equivalence/
```

`run_id` is `<timestamp>_<framework>_seed<n>`, and it is the **same** id used in the
`runs.csv` row — so any figure traces back to the row describing it.

**Why the per-run subfolder.** `training_results.csv`, `batch_metrics.csv`, `test.csv`
and the seven PNGs all carry fixed names with no framework or seed in them. Four
frameworks writing into one experiment folder would leave only the last one's files.
The three schema CSVs are exempt because they are append-only — that is what they are
for.

**Two families of plot, and they do not mix:**

| folder | written by | shows |
|---|---|---|
| `plots/<run_id>/` | `learning/main.py`, automatically | this ONE run — loss curve, VRAM, spike raster |
| `figures/` | `make_plots.py`, after several runs | the COMPARISON across frameworks (F0–F6) |

Nesting happens only when `--experiment` is given. Without it, everything stays flat in
`./outputs` exactly as before.

Repeat with a different `--framework` and the same `--seed` to compare frameworks; with
the same `--framework` and a different `--seed` to get replicates.

## B4. On Colab

A Colab runtime is a **new machine every session**, so it starts with Step 0:

```bash
!pip install -r requirements.txt
!python check_env.py --strict
```

`--strict` is worth it here: Colab ships its own torch and numpy, and a silent
resolution to a different version is exactly the drift that makes two runs
incomparable. Then two things differ from a laptop run, and both are command-line
flags — the config never changes.

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
| `check_env.py --strict` *(a cell, not a flag)* | — | the runtime is new, so its versions are unproven |

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
