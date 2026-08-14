# Response to Prof. Bauer — Comprehensive Benchmark of SNN Simulation Platforms

## Context

Prof. Bauer's brief (`2025_03_28_SNNSimulationGPUs (1).pdf`) asked us to explore
existing SNN-on-GPU frameworks (naming BindsNET, Brian2CUDA, Norse as examples),
compare runtime vs. scalability vs. accuracy, apply the result to event-vision
tasks, and assess real-time suitability — with explicit freedom to shape our own
individual angle on the topic.

Following that brief, two independent projects grew out of it in this workspace:

- **This repository outside `comparison/`** (`learning/`, `event_data_workflow/`,
  `configuration/`, `docs/`, `diagnostics/`) — built by me. A breadth-first
  platform: multiple frameworks, multiple event-camera datasets, an adaptive
  data/caching pipeline, adversarial-robustness evaluation, deployment-facing
  concerns (hardware export, sensor-resolution scaling).
- **`comparison/Benchmark_SNN_Frameworks/`** — built independently by Haseeb,
  imported here for reference. A depth-first, three-framework, one-dataset
  comparison done to a rigor standard (numerically verified neuron equivalence,
  multi-seed statistics, MLPerf-standard latency definitions) that would survive
  peer review.

Prof. Bauer's follow-up message asks for three things: an explanation of *why*
the SNN-simulator landscape is fragmented, a comparative analysis across
platforms, and a set of concrete "user stories" that turn the comparison into a
decision guide. This document answers all three, states plainly which parts of
the answer come from which project, and — where neither project actually
answers a question — says so rather than overclaiming.

---

## Pillar 1 — The Why: why isn't there just one SNN simulator?

The brief's own hint is the answer: **biological fidelity, hardware accuracy,
and computational efficiency trade off against each other**, and no single
framework design point wins on all three simultaneously. This project ran into
that trade-off directly, not just as theory:

**Computational efficiency vs. everything else — SpikingJelly.** SpikingJelly's
speed advantage comes from fused CUDA kernels via CuPy. That is *why* it is fast
(`docs/frameworks/realtime_nir_evaluation.md`), and also why it is the least
portable of the four frameworks we run: CuPy is CUDA-only, so SpikingJelly's own
execution engine cannot run on CPU or TPU at all without a full re-implementation.
Speed was bought with hardware lock-in — a direct, measured instance of the
trade-off, not a hypothetical one.

**Biological fidelity vs. trainability — BindsNET.** We tried a fourth
backend, BindsNET, built around STDP (spike-timing-dependent plasticity — a
local, unsupervised, biologically-motivated learning rule) rather than
backprop-through-time. It hit a real broadcast bug in BindsNET 0.2.7 and a
`torch._six` compatibility shim, and was removed as the project's scope
converged on the other four (`docs/frameworks/additional_frameworks.md`). The
lesson that survived the removal: STDP and surrogate-gradient BPTT are not two
implementations of the same idea — they are different answers to "how does an
SNN learn," one closer to biology, one closer to what makes GPUs fast. A
platform picks one lane.

**Hardware accuracy vs. training convenience — Sinabs vs. Brian2CUDA.**
Sinabs is PyTorch-based (backprop-trainable, GPU-fast) but is also the one
framework here with a genuine export path to a real neuromorphic chip
(SynSense's Speck). That already required a different data layout internally
(`(B,T,...)` batch-first instead of the other three's `(T,B,...)` time-first —
see `docs/frameworks/additional_frameworks.md`) to stay compatible with the
target hardware's execution model. **Brian2CUDA — named explicitly in the
original brief — sits at the tier past even that**: an explicit
differential-equation neuron simulator that compiles per-neuron dynamics
straight to CUDA C++, built for numerical/biophysical accuracy rather than
backprop training. Neither this project nor Haseeb's uses it. That is a real,
acknowledged gap (see §3, Story B) — not a design choice we're claiming credit
for.

**One paragraph answer for the professor:** there is no single winner because
"good SNN simulator" is not one target — it is three: (1) does it match
biological/neuroscience dynamics closely (BindsNET, Brian2CUDA), (2) does it
match a specific piece of real neuromorphic silicon's timing/power behavior
(Sinabs→Speck, and beyond it, actual chip-vendor toolchains), or (3) does it
train fast enough on a GPU to be useful as a deep-learning research tool
(SNNTorch, Norse, SpikingJelly, Spyx)? Every framework we evaluated picked a
point on that triangle and paid for it somewhere else.

---

## Pillar 2 — Comparative analysis

| Framework | Focus | Performance (this project's / Haseeb's measurements) | Ease of use | Sync→async handling |
|---|---|---|---|---|
| **SNNTorch** | Algorithmic / high-level (PyTorch autograd) | Diagnostic-only in ours (78.6% smoke test); Haseeb: 98.4–98.5% test acc, ~109–133s/epoch, full rigor | Easiest — most SNN-tutorial coverage, straightforward PyTorch idioms | Synchronous only — fixed-timestep dense tensors, no async execution path |
| **Norse** | Algorithmic, but built on explicit neuron ODEs (CubaLIF) — the closest of the four to physically-parameterized dynamics | Ours: 95.9% test acc, 5 epochs full data, 429s total, 9.28% spike rate (2026-08-11 run, `outputs/data/training_results.csv`); Haseeb: 98.38±0.20%, real batch-1 latency 22.9±2.4ms | Straightforward, but our own config carried an unresolved `threshold=1.0` / missing `input_scale` gap vs. Haseeb's tuned config (`docs/framework_comparison_report.md` §3.1) — a live example of how easy it is to get numerically "working but wrong" | Synchronous — but the only one of the four whose NIR export (`norse.to_nir()`) actually produces a clean, real parameter graph (Conv2d/CubaLIF/Affine) in this repo, which matters for any later async/hardware bridging |
| **SpikingJelly** | Algorithmic, CUDA-kernel-optimized for raw speed | Ours: 58.9% smoke test only; Haseeb: 98.25–98.57%, fastest wall-clock of the three (~100–116s/epoch) | Easy once installed; CUDA-only kernels are the cost (§1) | Synchronous, and structurally *cannot* cross into async/CPU/TPU execution — see §1 |
| **Sinabs** | DVS-first, algorithmic, but the only one with a real path to asynchronous neuromorphic hardware | Verified end-to-end on GPU (forward/backward/eval all confirmed — `docs/frameworks/additional_frameworks.md`); no full-scale accuracy run yet | Batch-first tensor layout breaks the assumption the other three share — real integration cost, documented at the point it bit us | The only framework in either project that actually **exits** the synchronous-tensor world: exports to SynSense Speck, a genuinely event-driven asynchronous chip |
| **BindsNET** (tried, removed) | STDP — biologically-plausible local learning, not backprop | Never reached a real run — broadcast bug in 0.2.7 blocked it | Hardest of everything we touched — hit an unmaintained-library bug directly | Still simulates on synchronous GPU tensors internally, despite the biologically-motivated learning rule — "STDP" ≠ "asynchronous execution" |
| **Spyx** (tried, removed) | Algorithmic, JAX/Haiku — different portability story (TPU-native via XLA, not CUDA) | Never reached a real run — `hk.max_pool` axis bug blocked it | Requires JAX-idiom fluency (functional transforms, `jax.jit`/`vmap`) — steepest learning curve of the six we touched | Synchronous, but the one path in this project's design notes toward TPU execution of another framework's trained parameters (via NIR) — see `docs/frameworks/realtime_nir_evaluation.md` |
| **Brian2CUDA** (never integrated) | Cycle-level / explicit-ODE neuroscience simulator | No data — not run in either project | N/A | N/A — named in the original brief as the reference for this tier, acknowledged gap here |

**On "ease of use":** the honest, evidence-backed summary is that ease of use
correlates inversely with how much of the trade-off in §1 a framework has
already made for you. SNNTorch is easiest because it commits hardest to
"just be a differentiable PyTorch module." Sinabs and Spyx cost more precisely
because they carry real obligations toward a downstream target (a chip, a
different XLA backend) that SNNTorch doesn't have to think about.

**On the sync→async transition specifically:** this is the one axis where
almost everything we tested is on the same side of the line. All four
backported/production frameworks (SNNTorch, Norse, SpikingJelly, Sinabs)
train via **synchronous, fixed-timestep, dense-tensor** surrogate-gradient
BPTT — including our own event-camera pipeline, which converts genuinely
asynchronous DVS event streams into fixed `T=16` time-bin frames
(`configuration/data_workflow.yaml`) before any of them ever see it. That
conversion is itself a sync/async boundary crossing, and it happens on the
*input* side, not inside any framework. The only place either project actually
reaches real asynchronous execution is Sinabs' Speck export path — a real
capability, not yet exercised beyond confirming it exists.

**What Haseeb's project adds to this table that ours doesn't:** rigor on the
three rows it covers. Where our numbers are diagnostic-only (SNNTorch,
SpikingJelly) or single-seed (Norse), his are numerically-verified-equivalent,
three-seeded, and cite the MLPerf definitions being used. Where our project
adds rows his doesn't: Sinabs (hardware export), BindsNET/Spyx (documented,
if unsuccessful, attempts at the biological-fidelity and TPU-portability
corners of the triangle).

---

## Pillar 3 — User stories

### Story A — The Algorithm Architect (new topology, cares about convergence + training speed, not hardware)

**Answer: SpikingJelly for raw iteration speed; Norse if the architect needs
to modify low-level neuron behavior.** This is the one story our stack answers
cleanly. Haseeb's fully-rigorous numbers (`comparison/Benchmark_SNN_Frameworks/experiments/ex1/results/epochs.csv`)
show SpikingJelly consistently fastest wall-clock per epoch (~100–116s vs.
Norse's ~117–131s and SNNTorch's ~109–133s) at statistically indistinguishable
final accuracy (all three land at 98.2–98.6%). That's the direct answer: if
you only care about "does it converge and how fast," SpikingJelly wins on
speed at no accuracy cost, *provided you stay on CUDA* (§1's trade-off). If the
architect's new topology needs a neuron model SpikingJelly's kernels don't
already support, Norse is the better sandbox — it's plain PyTorch with no
custom CUDA kernels standing between the researcher and the neuron equations
(confirmed by Norse being the only one of the four whose NIR export gives back
real, correct `CubaLIF` parameters in this repo).

*Who answers this: primarily Haseeb's project (the rigor makes the "SpikingJelly
is faster, not just different" claim trustworthy); ours supplies the "why"
(kernel portability trade-off) and the alternate answer for the harder case.*

### Story B — The Hardware Engineer (exact spike timing + power at gate/neuron level for a new async chip)

**Honest answer: neither project currently has the right tool for this.**
Everything in both `learning/frameworks/` and `comparison/Benchmark_SNN_Frameworks/src/adapters/`
computes forward passes as batched dense-tensor operations on a GPU, timed
externally (NVML polling — `event_data_workflow/gpu_stats.py`,
`system_monitor.py`). That measures *our GPU's* power and *our GPU's* spike
schedule, not the target chip's. The right tool named in the original brief
is **Brian2CUDA** — an explicit-ODE simulator with per-neuron numerical
integration, closer to the biophysical/timing fidelity this story needs — and
even it falls short of true gate-level accuracy, which would require an actual
chip-vendor toolchain (e.g. Intel's Lava for Loihi, or RTL/SPICE-level circuit
simulation) that is a full tier beyond anything named in the brief at all. We
flag this as a real, unclosed gap rather than stretching Norse's ODE-based LIF
implementation (the closest thing we have) to cover a claim it can't support.

*Who answers this: neither. This is the clearest gap in the whole benchmark,
and saying so is the correct answer.*

### Story C — The Edge AI Developer (deploy a trained model to a specific low-power neuromorphic chip, minimize the performance gap)

**Answer: Sinabs, with NIR as the general-purpose (but currently incomplete)
mechanism behind it.** This is the story our project answers most directly and
concretely, because we have *already built and tested* the two relevant pieces:

1. Sinabs ships a genuine export path to SynSense's Speck chip — a real,
   already-integrated deployment target, not a hypothetical one
   (`docs/frameworks/additional_frameworks.md`).
2. We tested whether the more general mechanism — NIR (Neuromorphic
   Intermediate Representation), meant to carry trained parameters between
   *any* two NIR-compatible frameworks/hardware targets — actually round-trips
   our real trained models. It doesn't, cleanly, yet:
   `norse.to_nir()` works but silently drops `MaxPool2d`/`Flatten` as `None`
   nodes; snnTorch's own NIR export hits a real version-incompatibility bug
   (`v_reset` unsupported by installed `nir` 1.0.4); SpikingJelly's documented
   `nir_exchange` module doesn't exist in the installed build; Sinabs'
   `to_nir()` explicitly rejects `MaxPool2d` outright
   (`docs/frameworks/realtime_nir_evaluation.md`).

That last finding is the actual, evidence-based answer to "what minimizes the
performance gap": **the gap isn't in the neuron math — LIF/Conv/Linear
parameters transfer correctly wherever tested — it's in the pooling/flatten
layers, which none of the four frameworks' NIR integrations currently handle
without extra glue code.** A developer following this story needs to either
avoid MaxPool2d in the deployed architecture (use AvgPool2d, which NIR does
support) or write the pooling reconstruction by hand on import. This is a
sharper, more actionable answer than "use tool X" — it names the exact
architectural choice that breaks portability.

*Who answers this: entirely our project. Haseeb's repo has no hardware-export
or cross-framework-portability story at all — a fair scope choice on his part
(§8 of `docs/framework_comparison_report.md`), but it means this story is ours
alone to answer.*

### Story D — The Large-Scale Researcher (millions of neurons, emergent behavior, speed over bit-perfect accuracy)

**Answer: neither project has run at that scale, but ours found the concrete
wall the researcher will hit first, empirically.** Our `event_data_workflow`
work diagnosed — not theorized — a memory-scaling failure mode that generalizes
directly to this story: BPTT activation memory scales with **sensor
resolution², not neuron count directly**, because every timestep's activations
must stay live for the backward pass. Moving from N-MNIST (34×34) to
N-Caltech101 (240×180) — an ~46× increase in one conv layer's spatial extent —
took a working `batch_size=128` config to a hard CUDA OOM on an 8GB card
(`event_data_workflow/vram_batch_scaling_task.md`). A "millions of neurons"
network hits the same wall from the population-size axis instead of the
resolution axis, but it's the same underlying constraint: **T timesteps ×
activation size × batch size, held simultaneously, is the actual ceiling**,
not raw neuron count in isolation. Our adaptive cache tiering
(`event_data_workflow/cache_engine.py`) addresses the *data*-side half of
that (don't hold more of the dataset in VRAM than necessary) but not the
*activation*-side half — the vram_batch_scaling doc's own candidate list
(gradient accumulation, truncated BPTT/checkpointing, mixed precision) is the
honest next-step menu, none of it applied yet.

For raw large-scale simulation speed with lower accuracy demands, the
literature-standard tools are population-level simulators (NEST, or Brian2CUDA
at scale) — again outside anything either project has integrated. What we can
honestly claim is the *diagnosis* of why naive scaling breaks, backed by a real
measured OOM, not the *solution* at million-neuron scale.

*Who answers this: ours, partially — a real, evidence-backed problem statement,
not a full solution. Haseeb's project never touches this axis (single small
dataset, by design).*

---

## Why we built it this way

The brief explicitly invited individual scoping ("Eigenes Thema möglich? Ja –
individuelle Ausgestaltung ist ausdrücklich erwünscht!"). Faced with that
freedom, the two halves of this workspace made *opposite, complementary* bets
about what "comprehensive" should mean:

- **Haseeb optimized for trustworthiness on a narrow question**: hold
  everything constant except the framework, verify the "constant" claim
  numerically instead of assuming it, run enough seeds to separate signal from
  noise. That is what makes his three rows in §2's table the ones worth
  quoting without a caveat.
- **This project optimized for coverage of the deployment questions a
  benchmark eventually has to answer**: what happens when the sensor isn't
  N-MNIST-sized, what happens when the target isn't a GPU at all, what happens
  under adversarial input, what does "real-time" actually require as a
  measured deadline rather than an adjective. Those are exactly the questions
  Stories B–D above are asking, and they were the harder ones to leave
  unaddressed precisely because the brief's own task list ends on "Bewertung
  der Eignung für Echtzeitszenarien" (real-time suitability) and "Anwendung
  auf typische Event-Vision-Aufgaben" (application to real event-vision
  tasks) — not on producing one more accuracy table.

Neither scoping decision is sufficient alone for what Prof. Bauer is now
asking for — a single deliverable covering the Why, the comparative analysis,
and actionable user stories. This document is the merge: Haseeb's rigor
supplies the numbers we can defend without caveats (Story A, most of §2's
table); this project's breadth supplies the deployment-facing answers
(Stories C and D, and the honest admission on Story B) that a numbers-only
comparison never reaches.

---

## Open gaps, stated plainly

Matching the professor's own framing — understanding the concepts matters
more than pretending the benchmark is finished:

1. **Brian2CUDA is still unintegrated.** It's the framework the original brief
   names for exactly the hardware-accuracy tier Story B needs, and neither
   project has run it. This is the single biggest missing row in §2's table.
2. **NIR round-tripping is parameter-only, not full-graph**, and even that is
   broken for 3 of 4 frameworks right now (§3, Story C) — real, fixable bugs
   (a version mismatch, a missing module, an unsupported layer type), not
   fundamental blockers, but not done.
3. **No million-neuron (or even thousand-neuron) scale run exists in either
   project** — Story D's answer is a diagnosed constraint, not a demonstrated
   solution.
4. **Our own three-framework overlap with Haseeb's work (SNNTorch, Norse,
   SpikingJelly) is still diagnostic-only** — no seeds, no equivalence proof —
   so §2's table correctly leans on Haseeb's numbers for those three rows
   rather than ours.
