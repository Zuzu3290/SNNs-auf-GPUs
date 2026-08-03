# Real-time suitability evaluation — NIR parameter matching + deadline framework

Planning notes and reference material for closing brief task 5 ("evaluation of
suitability for real-time scenarios") properly, instead of citing raw latency
numbers with no deadline to compare them against. Not yet implemented.

## Why current diagnostics aren't a real-time-suitability claim

`docs/results/`'s 6-backend run and any per-framework `test.csv` latency column
report throughput/latency under a fixed training-time config, single seed, no
target deadline. Three gaps to close before "it's real-time suitable" is a
defensible statement:

1. **A deadline to test against** — tie it to an actual dataset's native event
   rate or a stated reaction-time budget, not an arbitrary number. Different
   datasets imply different realistic deadlines (e.g. DVS128 Gesture's live
   gesture-recognition framerate vs. N-MNIST's offline classification setting
   carry different real-time meaning) — this should be evaluated per dataset,
   not as one universal number for the whole pipeline.
2. **Tail latency (p99), not median/mean** — a real-time system cares about
   worst-case jitter, not average speed.
3. **The actual deployment config** — latency should be measured through
   `GPURecordingCache`'s `eval`/`inference` phase budget (`cache_engine.py`),
   not the standard training-loop config.

## NIR — cross-framework numerical parameter matching

Resolves this project's own open item (`docs/Haseeb-open-items.md`: "different params for
different frameworks like Threshold vs v_th... beta and tau clashing").

**Neuromorphic Intermediate Representation (NIR)** is a common graph format
for spiking-network parameters (weights, thresholds, decay/tau, reset
behavior) that multiple frameworks can export to and import from, so a
trained model's neuron parameters can be transferred between frameworks
without hand-translating each framework's own parameterization.

**Framework support, confirmed (not all sources agree — verified against
primary docs, corrected once already in this project's own research):**
- snnTorch, Norse, Sinabs, Spyx, Lava, Rockpool, Nengo — supported per the
  NIR paper / Open Neuromorphic.
- **SpikingJelly — also supported**, via its own `nir_exchange` module
  (documented in SpikingJelly's own docs as "Export to and Import from NIR"),
  requires the `nir` + `nirtorch` packages. An earlier pass at this research
  incorrectly concluded SpikingJelly wasn't covered — it is.

All four frameworks actually used in this project (snnTorch, Norse,
SpikingJelly, Sinabs) are therefore NIR-compatible — parameter matching across
all four can go through NIR uniformly, no framework needs manual mapping.

## TPU / cross-hardware portability

**Not possible:** running SpikingJelly's own execution engine on a TPU.
Its speed advantage comes from CUDA-specific fused kernels via CuPy, which
SpikingJelly's own docs confirm cannot even run on CPU without conversion —
CuPy is CUDA-only, and TPUs execute via XLA, not CUDA, so there's no path
for SpikingJelly's kernels there at all.

**Possible:** extracting a trained model's parameters via NIR, then executing
those parameters on a different, TPU-native backend. Spyx (JAX + Haiku) is
the one backend already in this project built for that — JAX has first-class
TPU support, unlike the CUDA-kernel-bound frameworks. This is "SpikingJelly's
trained parameters, executed by Spyx, on a TPU" — not "SpikingJelly on a TPU."

Also relevant: `torch.compile` reportedly closes much of the raw speed gap
between Norse and SpikingJelly, and SpikingJelly's backends are documented as
`torch.compile`-compatible — a same-hardware way to separate "framework
design difference" from "missing optimization" before reaching for a
different accelerator.

## Feasibility check, run against the real Norse model (not assumed)

`norse.to_nir(model, sample_data=...)` against the actual `SNN_NORSE` instance
(`src/learning/frameworks/snn_norse.py`, real `Settings()` config, input shape
`[T=1, B=1, C=2, H=34, W=34]`) succeeds and returns a real `nir.NIRGraph`.
`conv1`/`conv2` export as `nir.Conv2d` with the real trained weight tensors,
`lif1`/`lif2`/`lif_out` export as `nir.CubaLIF` with the real instantiated
`tau_mem`/`v_threshold` (not just what's set in YAML — the full runtime
parameter set, including Norse's own unset defaults), `fc` exports as
`nir.Affine`. This is exactly the numerical-parameter capture the matching
plan needs.

**Real gap found: `pool1`, `pool2`, and `flatten` all export as `None`.**
NIR's node vocabulary has `AvgPool2d`/`SumPool2d` but no `MaxPool2d`, and
nirtorch 2.0.2 doesn't map `nn.Flatten` either (despite `nir.Flatten`
existing as a node type) — both silently become `None` nodes in the graph
rather than raising an error. Every framework wrapper in this project uses
the same `Conv→LIF→MaxPool→Conv→LIF→MaxPool→FC→LIF` architecture, so this
gap applies equally to snnTorch/SpikingJelly/Sinabs, not just Norse.

**Consequence:** a full graph round-trip (export from framework A, reconstruct
and run in framework B) will not work out of the box — the pooling/flatten
structure has to be supplied by framework-specific glue code on import, NIR
only carries the LIF/Conv/Linear parameters correctly. For the actual goal
here (comparing neuron-level numerical parameters — threshold, tau, weights —
across frameworks under matched config), this is fine: it's the LIF/Conv/
Linear nodes that matter, not full executable reconstruction. Scope the NIR
work to parameter comparison, not full cross-framework graph execution,
unless the pooling gap gets solved first (e.g. substitute `AvgPool2d` for
export purposes only, or reconstruct pooling manually on import).

## Feasibility check, all four frameworks — tested against the real models

Scope decided: parameter comparison only, not full cross-framework graph
execution (the MaxPool2d/Flatten gap above rules out clean round-trip
execution anyway). Tested each framework's own NIR export against its real
`ModelInterface` implementation in this repo, real `Settings()` config,
correct input shapes. This project's `network_architecture.yaml` already
configures plain LIF for all four (`snntorch: leaky`, `norse: lif_cell`,
`spikingjelly: lif`, `sinabs: lif`) — none of the exotic defaults
(`Alpha`/`Izhikevich`) that would be unrepresentable in NIR are actually in
use here, which is a genuine advantage for this plan.

| Framework | Export path | Result |
|---|---|---|
| Norse | `norse.to_nir()` | **Works** — real `Conv2d`/`CubaLIF`/`Affine` nodes with real trained values. `pool1`/`pool2`/`flatten` export as `None` (see above). |
| snnTorch | `snntorch.export_nir.export_to_nir()` | **Broken.** snnTorch 1.0.0's own bundled integration calls `nir.LIF(..., v_reset=...)` — but installed `nir` 1.0.4's `LIF` node has no `v_reset` parameter at all (checked its real `__init__` signature). Version-incompatibility bug in snnTorch's own NIR code against the current `nir` release, not something wrong on this project's side. |
| SpikingJelly | `spikingjelly.activation_based.nir_exchange` | **Not present.** This module doesn't exist in the installed build (`spikingjelly 0.0.0.0.14`), despite being documented on spikingjelly.readthedocs.io — likely documented against a newer/dev version than what's pip-installable right now. |
| Sinabs | `sinabs.nir.to_nir()` | **Explicitly rejects `MaxPool2d`**: `NotImplementedError: Module <class 'torch.nn.modules.pooling.MaxPool2d'> not supported` — same root cause as Norse's silent `None`, but Sinabs' mapper raises instead of skipping. |

**Net result: none of the four frameworks' own NIR integrations currently
export this project's real models cleanly end-to-end.** Norse is closest
(works, with the known/scoped-around pooling gap). The other three need one
of:
- **snnTorch**: work around the `v_reset` bug — e.g. a local monkey-patch of
  the LIF conversion function, or falling back to reading `.beta`/`.threshold`
  off the `snn.Leaky` module directly without going through NIR for this one
  framework (loses the "same extraction method across all four" property,
  but still enables numeric comparison).
- **SpikingJelly**: confirm whether a newer `spikingjelly` release actually
  ships `nir_exchange` and upgrade, or write a manual `model_map` against raw
  `nirtorch.extract_nir_graph()` for just the `LIFNode` type.
- **Sinabs**: call `to_nir()` on the individual `lif1`/`lif2`/`lif_out`
  submodules directly rather than the whole model, sidestepping the
  `MaxPool2d` mapper entirely — consistent with the "parameter comparison
  only" scope, since a single LIF layer doesn't contain any pooling.

## Implementation status

NIR parameter matching (item 2) is parked until a final export stage —
deprioritized in favor of the latency harness (item 3), which doesn't
depend on it. Built and smoke-tested (synthetic batches, not real DVS128
Gesture data — see below):

- `src/learning/inference.py` — `SNNTester.run()` now reports p99 latency
  alongside the existing p50/p90, and calls `set_phase("eval")` on the test
  dataset if it exposes one (duck-typed — only `GPURecordingCache` does).
- `src/learning/realtime_eval.py` — new `RealTimeLatencyEvaluator`: runs
  `SNNTester` across multiple seeds (fresh model per seed via a
  `model_factory` callable), compares each seed's p99 against
  `configuration/data_workflow.yaml`'s `realtime.deadline_ms[dataset]`,
  reports per-seed pass/fail plus a mean/stdev/worst-case summary, writes
  CSVs. Raises clearly if no deadline is configured for the dataset in use.
- `event_data_workflow/workflow_config.py` — parses the new `realtime`
  section into `WorkflowSettings.REALTIME_DEADLINE_MS`.
- **Found and fixed a pre-existing bug while smoke-testing**: `SNNTester.write_csv()`
  used a `→` character in a `print()` call, which crashes with
  `UnicodeEncodeError` on Windows' default `cp1252` console encoding — not
  something introduced by this work, but it blocked the evaluator from
  completing a run on this machine, so fixed at that call site and in the
  new evaluator's own equivalent print. Not swept repo-wide — other files may
  have the same latent issue, out of scope here.
- **Not yet run against real data**: `tmp/data/DVSGesture` exists locally
  but is an empty stub (not actually downloaded) — the smoke test used
  synthetic randomly-generated batches of the right shape/dtype to verify
  the evaluator's own logic (percentile math, seed loop, deadline
  comparison, CSV output), not real accuracy or real latency numbers. First
  real run needs the actual DVS128 Gesture download, which is sizeable —
  not triggered without confirming that's wanted first.

## Planned implementation order

1. **Done** — starting dataset: DVS128 Gesture. Deadline: **105ms**, from
   IBM's original DVS128 Gesture paper (Amir et al., CVPR 2017), which
   reports 105ms gesture-onset detection latency on TrueNorth as the
   real-time target the dataset itself was built around — used as the
   literature anchor rather than an arbitrary number. An alternative,
   more modern framing also found in the literature: SOTA methods output
   classification results every 25ms with a 225ms temporal averaging
   filter — worth comparing against once numbers exist, not as the primary
   target. Set in `configuration/data_workflow.yaml` → `realtime.deadline_ms`.
2. Build NIR-based parameter matching across snnTorch/Norse/SpikingJelly/
   Sinabs (`nir` 1.0.4 + `nirtorch` 2.0.2 — already installed in this
   environment, confirmed).
3. Re-run latency as p99 under the `eval`/`inference` phase config, across
   multiple seeds.
4. Extend to additional datasets already in `DATASET_REGISTRY` beyond
   N-MNIST, since each carries a different realistic real-time meaning.
5. (Optional, lower priority) NIR-export a trained model into Spyx for a TPU
   execution comparison — hardware-portability angle, not a SpikingJelly-on-
   TPU claim.

## Sources

- [Neuromorphic intermediate representation: A unified instruction set for interoperable brain-inspired computing — Nature Communications](https://www.nature.com/articles/s41467-024-52259-9)
- [Neuromorphic Intermediate Representation (NIR) — Open Neuromorphic](https://open-neuromorphic.org/neuromorphic-computing/software/data-tools/neuromorphic-intermediate-representation/)
- [NIR: A unified instruction set for brain-inspired computing — Open Neuromorphic](https://open-neuromorphic.org/workshops/neuromorphic-intermediate-representation/)
- [SpikingJelly documentation](https://spikingjelly.readthedocs.io/) — "Export to and Import from NIR" tutorial, `nir_exchange` module
- [SpikingJelly — GitHub](https://github.com/fangwei123456/spikingjelly)
- [SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence — Science Advances](https://www.science.org/doi/10.1126/sciadv.adi1480)
- [Towards Scalable GPU-Accelerated SNN Training via Temporal Fusion (torch.compile finding)](https://arxiv.org/pdf/2408.00280)
- [Spyx: A Library for Just-In-Time Compiled Optimization of Spiking Neural Networks](https://arxiv.org/pdf/2402.18994)
- [A low power, fully event-based gesture recognition system — IBM Research (Amir et al., CVPR 2017)](https://research.ibm.com/publications/a-low-power-fully-event-based-gesture-recognition-system) — source of the 105ms real-time deadline figure
- [Agreeing to Stop: Reliable Latency-Adaptive Decision Making via Ensembles of Spiking Neural Networks](https://arxiv.org/pdf/2310.16675) — context for the 25ms/225ms alternative framing
