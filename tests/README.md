# tests

Unit suites for the frameworks + network layer. CPU-only, no dataset, no download —
every suite builds models from `Settings` and feeds them synthetic tensors, so a full
run takes well under a minute on any machine.

```
python tests/run_all.py            # every suite, one summary table
python tests/run_all.py adapters   # only suites matching "adapters"
python tests/unit_adapters.py      # one suite, full PASS/FAIL detail
```

`run_all.py` exits non-zero if any suite fails, so it works as a pre-push gate. Each
suite runs in its own subprocess: several mutate global torch RNG state and load four
different SNN libraries, and a shared interpreter would let one suite's leftovers
decide another's result.

Not pytest, deliberately — these run as plain scripts on a bare Colab runtime or inside
the Docker image with no extra dependency and no runner configuration, and they print
the same PASS/FAIL table as the rest of this project's check scripts.

## What each suite covers

| suite | covers |
|---|---|
| `unit_neuron_spec.py` | `skeleton/neuron_spec.py` typed access; the no-silent-defaults rule; that the four `neuron:` blocks really describe ONE neuron (the unit conversions checked as arithmetic); the `Settings` changes — `NEURON`, `SEED`, shared optimizer/loss, `FRAMEWORK_CFG` gone, `display()` still working |
| `unit_adapters.py` | what each adapter actually builds: decay 0.9, input gain 1.0, threshold 1.0, hard immediate reset, one binary spike — each measured from a single layer. Plus the state lifecycle, spike counting, `describe()`, and one test per framework-specific trap |
| `unit_neuron_picker.py` | the per-layer neuron picker: `lif` builds everywhere; `alpha`/`izhikevich`/`iaf`/`alif` raise; the old `leaky`/`lif_cell` names raise with a migration hint; missing key, empty block, typo and unknown framework all raise rather than defaulting |
| `unit_shared_net.py` | the shared network: the FC_IN formula/probe cross-check, one weight fingerprint across all four, `[T,B,C]` everywhere, identical spike rates, one loss value, gradients reaching every weight, and the optimizer/loss factories |
| `unit_pipeline_integration.py` | the seam with the HOST pipeline: `ModelInterface`, `ActivityMonitor`, `DenseTimestepBuffer`, `measure_dense_macs`, `compute_cv_isi`, `aggregate_spike_output`, `sum_over_time_cross_entropy`, checkpoint round-trip, `SpikingNet` introspection |
| `unit_seeding.py` | `skeleton/seeding.py`: reproducible weight init, one fingerprint per seed across all four, immunity to prior RNG use, both fingerprint kinds, `verify_cross_framework_init`, and the data-order generators at generator level |

## The shape of a check

Every assertion goes through `Suite.check(name, condition, detail)` and prints one row.
`Suite.expect_raises` additionally asserts the error MESSAGE names the offending key —
an error that fires without saying what was wrong costs as much time as no error.

## What is deliberately not here yet

- **Data-order seeding against real DataLoaders** — the generators are tested, the
  wiring belongs to the data unit.
- **`check_network.py` / `equivalence_check.py`** — postponed; these suites cover the
  same properties at unit level in the meantime.
- **Training-loop and metric tests** — the training unit.
