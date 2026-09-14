# Experiment 2 — each framework out of the box

Experiment 1 forces all four frameworks to compute **one** neuron, so only the
implementation can differ. Experiment 2 asks the opposite question:

> If you follow each framework's own defaults and its own documented example for
> event-camera data, what neuron do you actually **get** — and what does that do to
> accuracy, speed, spiking activity and memory?

Config: `experiments/ex2/config.yaml` (lives beside this README, in the experiment it
describes). Translated from `SNNs_2/config/config_ex2.yaml`.

## What "out of the box" means

Two sources, and they sometimes disagree:

| tag | meaning |
|-----|---------|
| **(D)** | constructor default — what `Neuron()` gives with no arguments |
| **(E)** | the framework's own **event-data example** passes it explicitly |

Where they disagree, **the example wins** — that is what a developer following the
framework's documentation would actually run.

## Scope

**Only the neuron block moves.** Data pipeline, architecture, optimiser, learning rate,
loss, timesteps, batch size and seed stay identical to Experiment 1.

Adopting each framework's *whole* recipe instead would make every metric
uninterpretable — their examples differ in dataset, batch size, architecture and loss all
at once, so you would be comparing training recipes rather than frameworks, and no column
in `runs.csv` would mean the same thing in two rows.

## The neuron changes, ex1 → ex2

| framework | parameter | ex1 | ex2 | why |
|-----------|-----------|-----|-----|-----|
| **snntorch** | `beta` | 0.9 | **0.5** | (D)+(E) `snn.Leaky` default and Tutorial 7 |
| | `reset_mechanism` | zero | **subtract** | (D) snnTorch's default is a SOFT reset |
| | `reset_delay` | false | **true** | (D) applies a spike's reset one timestep LATE |
| **spikingjelly** | `tau` | 10.0 | **2.0** | (D) `LIFNode` default |
| | `decay_input` | false | **true** | (D) DIVIDES input by tau → gain 1/tau = 0.5 |
| | `detach_reset` | false | **true** | (D) stops gradient through the reset |
| **norse** | `input_scale` | 10.0 | **1.0** | no compensation out of the box, so the decay/gain lock stands: decay 0.9 ⇒ gain 0.1 |
| | `surrogate` | circ(0.5) | **super(100.0)** | (D) SuperSpike is norse's default |
| **sinabs** | `tau_mem` | 9.4912 | **.inf** | (D) `exp(-1/inf) = 1` → NO leak, integrate-and-fire |
| | `spike_fn` | single | **multi** | (D) one neuron may emit 2, 3+ spikes per timestep |
| | `reset_mechanism` | zero | **subtract** | (D) sinabs' default is a soft subtract |
| | `min_v_mem` | null | **-1.0** | (D) lower bound on the membrane |
| | `surrogate` | single_exponential | **periodic_exponential** | (D) |

Everything else in each neuron block is **already** that framework's own default and is
inherited unchanged.

### Two things worth knowing before reading the results

**norse's `alpha: 100.0` does nothing.** Measured against norse 1.1.0: SuperSpike's
backward never reads `ctx.alpha`, so it behaves as `alpha=1` whatever is written. The
adapter logs a warning on every run. That *is* the out-of-box behaviour, so it stays —
but the value in the config is not the value in effect.

**spikingjelly's surrogate is the one parameter where ex1 and ex2 agree by choice.**
`LIFNode`'s default is `Sigmoid(alpha=4.0)`, but SpikingJelly's own DVS example
(`classify_dvsg.py`) passes `ATan()`. The example wins, so `atan(2.0)` is inherited.

## Non-neuron parameters pinned by this overlay

These are **not** part of ex2's question. This pipeline's base config differs from the
SNNs_2 reference run, so ex2 states them explicitly to stay comparable with ex1:

| parameter | this pipeline's base | ex2 pins it to | note |
|-----------|---------------------|----------------|------|
| `framing.n_time_bins` | 16 | **20** | ex1 ran at T=20. **Changing T rebuilds the frame cache** — first pass is slow. |
| `training.epochs` | 2 | **5** | ex1 ran 5 |
| `training.lr_scheduler` | cosine | **none** | ex1 used a constant learning rate |
| `dataset.name` | null (prompts) | **N-MNIST** | non-interactive runs must state it |

## Running it — one framework per cell

Each framework is a separate invocation; `runs.csv` is **append-only**, so rows
accumulate into one comparable table. Seed counts do **not** have to match across
frameworks — one seed each is a valid experiment.

```bash
python learning/main.py --config experiments/ex2/config.yaml --experiment ex2 --framework torch  --seed 0
python learning/main.py --config experiments/ex2/config.yaml --experiment ex2 --framework sj     --seed 0
python learning/main.py --config experiments/ex2/config.yaml --experiment ex2 --framework norse  --seed 0
python learning/main.py --config experiments/ex2/config.yaml --experiment ex2 --framework sinabs --seed 0
```

Repeat with `--seed 1`, `--seed 2` for replicates. On Colab, `--results-root` can point at
Drive so results survive the session:

```bash
python learning/main.py --config experiments/ex2/config.yaml --experiment ex2 \
    --framework sinabs --seed 0 --results-root /content/drive/MyDrive/runs
```

Output lands in `experiments/ex2/`:

```
results/     runs.csv · epochs.csv · layers.csv · runs/<run_id>.json
             plus training_results.csv and test.csv (this pipeline's own richer per-epoch data)
plots/       the trainer's own plots, and the equivalence figures
figures/     the F0-F6 comparison figures (created by make_plots.py)
```

Then, once the runs are in:

```bash
python make_plots.py --experiment ex2
python equivalence_check.py --experiment ex2     # neuron figures for THIS config
```

`equivalence_check.py` is worth running for ex2 specifically: under these values the four
neurons are **deliberately different**, and the figure shows exactly how.

## What to expect

Measured at initialisation with this config (`T=20`, Poisson input, before any training):

| framework | lif1 | lif2 | output |
|-----------|------|------|--------|
| snntorch | 0.088% | 0.000% | 0.000% |
| norse | 0.000% | 0.000% | 0.000% |
| spikingjelly | 0.000% | 0.000% | 0.000% |
| **sinabs** | **2.414%** | **2.848%** | **6.500%** |

That spread is the finding, not a fault: out of the box, norse's gain drops to 0.1,
spikingjelly's to 0.5 with a decay of 0.5, snntorch's decay halves to 0.5 — while sinabs
stops leaking entirely and integrates until it fires. Three of the four start near-silent
and one starts freely firing.

Whether the quiet three recover during training is exactly what this experiment measures.
Do not read the initialisation table as the result.
