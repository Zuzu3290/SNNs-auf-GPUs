# Diagnostics Report: This Project vs. Haseeb's `Benchmark_SNN_Frameworks`

Comparison of `comparison/Benchmark_SNN_Frameworks/` (Haseeb's work, imported for
reference) against this project's own framework-comparison output
(`docs/results/`, `docs/Haseeb-open-items.md`, `base_line implmentation.md`).
Both projects independently benchmark the same three SNN libraries — snnTorch,
Norse, SpikingJelly — on the same dataset (N-MNIST) with near-identical
network architectures. That overlap is what makes a direct comparison
meaningful rather than apples-to-oranges.

---

## 1. Executive summary

| | This project | Haseeb's `Benchmark_SNN_Frameworks` |
|---|---|---|
| **Scope** | Full research platform: 6 backends (SNNTorch, Norse, SpikingJelly, Sinabs, BindsNET, Spyx), 6 event-camera datasets, adaptive caching, adversarial (TRADES) training, regression + classification | Single focused experiment: 3 backends (snnTorch, SpikingJelly, Norse), 1 dataset (N-MNIST), no adversarial/regression scope |
| **Depth on the 3 shared frameworks** | Norse: real full-scale run now exists (5 epochs, ~51k/60k train samples, full 10k test set, 1 seed — see §4.1). snnTorch/SpikingJelly: still diagnostic-only (1 epoch, 20 batches) | Production-grade — 5 epochs, full dataset, 3 seeds × 3 frameworks = 9 runs, mean ± std |
| **Neuron equivalence verified before benchmarking?** | No — parameters chosen by design note (`base_line implmentation.md`) but never numerically verified to overlay | **Yes** — membrane traces + spike times checked to agree to `1.19e-07` (float32 limit) before any run counted |
| **Statistical treatment** | None — single run per framework, no error bars | Full — mean ± std over 3 seeds, explicit noise-floor derivation, paired vs. unpaired reading discipline |
| **Latency definition** | Batch latency ÷ batch size (throughput restated as a per-item number) | MLPerf Single-Stream: batch size 1, median + p90 |
| **Energy methodology** | NVML polling, cold/hot idle baseline, dynamic-energy subtraction, automated warning flags | Same architecture (NVML, cold/hot idle, dynamic subtraction) — independently converged on an identical design, plus explicit instrument-error citation (±73% vs. physical meter) |
| **Verdict quality** | "Diagnostic run, not a quality benchmark" — stated explicitly in `docs/results/README.md` | Publishable-grade — findings are load-bearing (bugs cited against source, claims tested against noise floor) |

**Bottom line:** the two projects are not competitors measuring the same
thing — they sit at different rigor tiers on the *same question*. This
project's strength is breadth (more backends, more datasets, a working
production pipeline with caching and adversarial training). Haseeb's
strength is depth on the narrow 3-framework comparison this project also
attempts: it is the version of that specific experiment done to a standard
that would survive peer review. Sections 4–7 identify exactly what to import.

---

## 2. What each project actually is

**This project** (`SNNs-auf-GPUs`) is an end-to-end platform: dataset
registry → adaptive cache (memory/disk/hybrid/GPU-VRAM) → framework-agnostic
trainer/tester → adversarial robustness (TRADES) → per-framework diagnostics.
`docs/results/run_benchmark.py` is one script inside that platform, used to
sanity-check all 6 backends run correctly end-to-end and to compare the
*shape* of their behavior — its own README calls it a diagnostic, not a
benchmark, in bold.

**Haseeb's project** is a single, narrow experiment done properly: hold
everything constant except the framework, verify that "everything constant"
claim numerically rather than assuming it, then run enough seeds to know
whether an observed gap is real or noise. It has no data pipeline, no
adversarial training, no caching system — `train.py` runs one framework once,
by design.

These are not the same kind of artifact. Judging Haseeb's repo by platform
breadth, or this project's diagnostic run by benchmark rigor, would both be
unfair. The comparison below is scoped to where they actually overlap: LIF
equivalence methodology and the snnTorch/Norse/SpikingJelly numbers on
N-MNIST.

---

## 3. Architecture and config overlap

Both use essentially the same network on the same dataset — this was not
coordinated; it converged independently, which is itself a useful sanity
check that the problem framing was reasonable on both sides.

| | This project | Haseeb |
|---|---|---|
| Dataset | N-MNIST, 34×34, 2 polarities | N-MNIST, 34×34, 2 polarities |
| Network | `12C5–MP2–32C5–MP2–FC10` (`configuration/network_architecture.yaml`) | `12C5–MP2–32C5–MP2–FC10`, 18,254 params |
| Timesteps | 25 (`configuration/SNN_module.yaml`) — `docs/Haseeb-open-items.md` notes an earlier 16 vs. Haseeb's 20 | 20 |
| Norse threshold | `1.0`, **no compensating gain** | `1.0` + `input_scale: 10.0` |
| snnTorch threshold | `0.5` | `beta=0.9` target, reset forced to `"zero"` |
| SpikingJelly | `tau: 2.0`, `threshold: 0.5` | `tau=10.0`, `decay_input=False`, `v_reset=0.0` |

### 3.1 The Norse threshold gap is not cosmetic

This is the single most consequential technical finding in this comparison,
and it's already flagged (unresolved) in `docs/Haseeb-open-items.md:121-132`:

- This project's own prior finding (**EXP-004**, referenced in that file) is
  that `threshold=1.0` **kills Norse** — dead neurons, no spikes — because
  Norse's `tau_mem_inv`/`dt` parameterization silently attenuates input
  relative to snnTorch/SpikingJelly at that threshold.
- `configuration/SNN_module.yaml:63` currently sets Norse's threshold to
  `1.0` with **no `input_scale`** — `learning/frameworks/snn_norse.py` has no
  `input_scale` parameter implemented at all (confirmed by grep — zero
  matches under `learning/`).
- Haseeb's config runs the *same* `threshold=1.0` but pairs it with
  `input_scale=10.0`, which is exactly the compensating gain this project's
  own EXP-004 note says is missing.

**Update from the real run (§4.1): the neurons are not dead** — the 2026-08-11
full run shows a healthy 9.28% final spike rate and 95.36% train accuracy, so
EXP-004's dead-neuron failure mode is not currently occurring outright.
But that spike rate is **over 4× Haseeb's Norse** (2.21 ± 0.16%) for a
network reaching *lower* accuracy, which is consistent with — not proof of —
Norse operating at a different, less efficient point on its input-gain curve
without the compensating `input_scale`. The fix Haseeb already applies is
still worth implementing and A/B-testing against the current config; the
claim is downgraded from "likely dead neurons" to "likely a suboptimal
operating point," and only a real A/B run resolves it.

### 3.2 Neuron equivalence was designed here, never verified

`base_line implmentation.md` is this project's own design note for exactly
Haseeb's equivalence-first methodology — it independently proposes "translate
one physical baseline into each framework's native parameters" and lists a
"Resolve the Norse asymmetry" step. **It was never executed**: there is no
`equivalence_check.py` equivalent in this project, no membrane-trace overlay,
no numerical proof that the three frameworks compute the same neuron here.
Haseeb's project is the version of this project's own design note that
actually got run — down to flagging the same Norse asymmetry as the thing
that must be resolved before the comparison means anything.

---

## 4. Direct numeric comparison (same 3 frameworks, same dataset)

Two different vintages of this project's own numbers exist and should not be
conflated:

- `docs/results/README.md` — the committed **diagnostic** table (1 epoch, 20
  train batches, 10 test batches). Explicitly labeled a smoke test, not a
  benchmark.
- `outputs/data/{test,training_results}.csv` — a **real, full-scale, local
  run** (`main.py`, current `configuration/SNN_module.yaml`, framework=Norse,
  5 epochs, ~400 iterations/epoch ≈ 51k of 60k train samples, full 10,000-item
  test set), generated 2026-08-11 and not yet reflected in `docs/results/`.
  This supersedes the diagnostic Norse row below.

### 4.1 Norse — real run vs. diagnostic vs. Haseeb

| Metric | Old diagnostic (1 epoch/20 batches) | **New full run (5 epochs, full data, 1 seed)** | Haseeb (5 epochs, full data, 3 seeds, mean ± std) |
|---|---|---|---|
| Test accuracy | 73.7% | **95.90%** (10,000/10,000 test samples, weighted over 79 batches) | 98.38 ± 0.20% |
| Final train accuracy | 59.4% | **95.36%** | — |
| Train energy (total) | — | **12.51 kJ / 429 s ≈ 29.2 W avg** | 35.0 ± 1.3 kJ total, 18.35 ± 0.83 W dynamic |
| Latency/sample (batch-amortized) | 0.516 ms | **0.484 ms** | 22.9 ± 2.4 ms (**true batch-size-1** — see §5, not the same metric) |
| Spike rate | — | **9.28%** (final epoch) | 2.21 ± 0.16% |

The new run is a real result, not a smoke test — full test set, full-ish
training pass, actual NVML energy — and it's the number to use going
forward, not 73.7%. It still trails Haseeb's 98.38% by **~2.5 pp**, which is
well outside Haseeb's own noise floor (~0.15 pp), so this is a real gap worth
explaining, not measurement noise. The most likely levers, in order of
probable impact: (a) 1 seed vs. 3 — some of the 2.5 pp could be an unlucky
draw, but Haseeb's own seed-to-seed spread tops out at 0.20 pp, so this alone
almost certainly isn't 2.5 pp of it; (b) no `input_scale` compensation on
Norse's threshold=1.0 (§3.1) — spike rate here (9.28%) is over 4× Haseeb's
Norse (2.21%), which points at a genuinely different operating point for the
neuron, not just an undertrained one; (c) `timesteps=25` vs. Haseeb's 20 and
different activity-regularization terms this project adds and Haseeb's
doesn't. (b) is the one with a concrete, already-documented fix.

### 4.2 snnTorch / SpikingJelly — still diagnostic-only

No equivalent full-scale local run exists yet for these two — only the old
1-epoch/20-batch numbers (78.6% / 58.9%) are available, and those remain
smoke-test numbers, not results. Running `docs/results/run_benchmark.py
--full` (or `main.py` with `training.framework` switched in
`configuration/SNN_module.yaml`) for both is the natural next step, now that
the Norse run shows what a real pass looks like.

### 4.3 Latency — now measured apples-to-apples (Norse)

`inference.py`'s built-in metric is batch-amortized, not comparable to
Haseeb's — see §5. To actually answer "is my latency higher than his,"
a standalone batch-size-1 script was run against this project's real Norse
model on the real N-MNIST test set, 200 timed samples after a 20-sample
warmup, CUDA-event timed — the same methodology as Haseeb's report:

| | This project (batch-amortized, `inference.py`) | **This project (true batch=1, measured just now)** | Haseeb (true batch=1) |
|---|---|---|---|
| Norse median latency | 0.484 ms | **2.706 ms** | 22.9 ± 2.4 ms |
| Norse mean | — | **3.183 ms** | — |
| Norse p90 | — | **5.035 ms** | 26.51 ± 3.27 ms |
| Norse p99 | — | **5.655 ms** | — |

**Answering the actual question: no, once measured the same way, this
project's Norse latency is not higher than Haseeb's — it's about 8.5×
lower.** But that comparison is confounded by hardware and should not be
read as "this pipeline is faster": this machine has an **RTX 5060 Laptop
GPU** (Blackwell, 2025), Haseeb's numbers were measured on a **Tesla T4**
(Turing, 2018, a cloud inference card, not a fast one even when it launched).
A modern consumer GPU beating a 7-year-old cloud inference card at
single-sample latency is expected regardless of which SNN pipeline is
running on top — this is not evidence about SNNs-auf-GPUs vs.
Benchmark_SNN_Frameworks as software. A fair same-hardware comparison would
need Haseeb's `train.py` run on this machine, or this project's model run on
a T4.

---

## 5. Finding: the two projects measure "latency" differently, not just with different rigor

This project's latency ("ms per sample") comes from
`learning/inference.py:286`: it times a full batch, then divides by the batch
size — `per_sample_latencies_ms.extend([latency_ms / B] * B)`. That is
**amortized batched latency**, i.e. inverse throughput restated in
milliseconds. It is a legitimate metric (MLPerf calls this the *Offline*
scenario), but it is not comparable to Haseeb's number.

Haseeb's latency (`experiments/ex1/report_ex1.md` §5.2) is measured with
`batch size = 1`, explicitly citing MLPerf's *Single-Stream* definition — the
number a live event camera reacting to one frame at a time would actually
see. The two numbers differ by roughly 30–80×, and that gap is entirely
explained by batching, not by a Norse/snnTorch/SpikingJelly performance
difference:

> snnTorch's latency penalty appears only at batch size 1... at batch 128 it
> amortises away — [Haseeb's report, §7.2]

This project's `inference.py` already computes p90/p99 tail latency — good
instinct — but always over batched runs. **This project currently has no
Single-Stream (batch-size-1) latency measurement at all.** If a real-time /
event-camera use case is ever a claim this project wants to make (the README
frames the whole project around "reacting to a live event camera"), that
number does not exist yet and needs its own code path, not a relabeling of
the existing one.

---

## 6. Energy: independently converged, one project draws the conclusion further

Both projects built the same three-part energy pipeline without coordinating:
NVML polling during training, a cold idle baseline before training, a hot
idle baseline after, and dynamic energy = total − (idle power × duration).
This project's `docs/results/run_benchmark.py:136-155` (`energy_warnings`)
and Haseeb's report's negative-dynamic-energy / baseline-pollution checks are
functionally the same guardrail, written independently.

Where they diverge:

- Haseeb's report cites the actual literature bound on NVML sensor error
  (±73% vs. a physical meter, 9.75–14.5 Hz refresh, arXiv:2312.02741) and
  therefore **refuses to draw a conclusion from the energy numbers** in that
  report — they're published as raw instrument output, explicitly not yet a
  finding, with the exact self-contradiction that justifies the caution
  (snnTorch's longest run recorded the *lowest* energy — §5.3 of that
  report).
- This project's diagnostic table (`docs/results/README.md`) reports actual
  GPU energy/power numbers without that caveat attached in the table itself
  (the caveat about EPOCHS=1 covers accuracy, not the energy-instrument
  reliability point specifically).

**Recommendation:** adopt Haseeb's discipline here directly — it costs
nothing beyond a caveat sentence, and this project already has all the
instrumentation needed (cold/hot idle, dynamic subtraction) to run the same
self-contradiction check once multi-seed data exists.

---

## 7. What "industrial-grade" would require here

Using Haseeb's report as the reference standard (it already cites MLPerf and
follows its throughput/latency definitions), here is the concrete gap between
this project's current diagnostic run and a benchmark that could be published
or shown to a supervisor as a real comparison result — not a to-do list, a
gap analysis against what Haseeb's report already demonstrates is achievable
with this exact codebase's stack:

| Gap | Current state | What closes it |
|---|---|---|
| **No equivalence proof** | Neuron params chosen by design note, never checked to overlay | Port Haseeb's `equivalence_check.py` approach: same input current into one neuron per framework, overlay membrane traces, report max disagreement |
| **No statistical treatment** | 1 run, 1 epoch, no seeds | Minimum 3 seeds × full epochs per framework, report mean ± std, define a noise floor before ranking anything |
| **Norse dead-neuron risk** | `threshold=1.0`, no `input_scale` | Implement `input_scale` in `learning/frameworks/snn_norse.py` (Haseeb's exact fix is already documented in `docs/Haseeb-open-items.md:121-132`) |
| **No Single-Stream latency** | Only batch-amortized latency exists | Add a batch-size-1 timing path in `inference.py`, alongside the existing batched one — keep both, label both |
| **Energy conclusions outrun the instrument** | Numbers reported without sensor-reliability caveat | Cite the NVML error bound, run the "longest-run-lowest-energy" self-contradiction check once multi-seed data exists |
| **No paired-vs-unpaired reading discipline** | N/A (no seeds to pair) | Once seeds exist: compare within-seed (same session, cancels machine drift) and across-seed (conservative) separately, as Haseeb's §5.2 does |

None of this requires new infrastructure — this project's caching, NVML
integration, and per-framework trainer/tester already exist and are, if
anything, more capable than Haseeb's single-purpose repo (this project also
handles regression, adversarial robustness, and 3 more backends Haseeb's
doesn't cover at all). The gap is entirely in **how the existing
instrumentation is used for the 3-framework comparison specifically** — one
diagnostic run vs. a seeded, verified, noise-floor-aware one.

---

## 8. What this project has that Haseeb's doesn't

To keep this balanced — the comparison isn't one-directional:

- **6 backends vs. 3** — Sinabs, BindsNET, and Spyx are exercised end-to-end
  here; Haseeb's repo doesn't touch them.
- **Adversarial robustness (TRADES/FGSM/PGD)** — no equivalent in Haseeb's
  repo. The 2026-08-11 Norse run's `outputs/data/adversarial_robustness.csv`
  is a real result: 96.28% clean → 92.06% under FGSM (ε=0.01) → collapses to
  7.6% by ε=0.1; PGD-20 is slightly stronger as an attack at the same budget
  (91.63% at ε=0.01, down to 1.7% at ε=0.1). That collapse curve is itself a
  finding worth carrying forward (this network has effectively no robustness
  margin past ε≈0.05) and is a genuine capability gap in Haseeb's repo, which
  doesn't evaluate robustness at all.
- **Regression task support (DSEC optical flow)** — Haseeb's repo is
  classification-only.
- **Adaptive data pipeline** (memory/disk/hybrid/GPU-VRAM cache selection) —
  Haseeb's repo uses a single Tonic disk cache.
- **6 dataset registry vs. 1** — this project's `DATASET_REGISTRY` supports
  N-MNIST, DVS128 Gesture, ASL-DVS, N-Caltech101, DSEC, and one more,
  selectable at runtime; Haseeb's repo is N-MNIST-only by design (that's a
  deliberate scoping choice on Haseeb's part, not a gap).

---

## 9. Recommended next actions, in priority order

1. **Fix the Norse `input_scale` gap** (§3.1) — this is a correctness bug
   with an already-documented fix, independent of any benchmarking work.
2. **Run snnTorch and SpikingJelly at the same full scale as the new Norse
   run** (§4.1–4.2) — switch `training.framework` in
   `configuration/SNN_module.yaml` and rerun `main.py`, or use
   `docs/results/run_benchmark.py --full`. Right now Norse is the only one of
   the three with a real result; the other two still only have smoke-test
   numbers (78.6% / 58.9%) that should not be quoted anywhere. Then repeat
   with 3 seeds each before ranking anything, matching Haseeb's design.
3. **Add a batch-size-1 latency path** to `inference.py` if a live/real-time
   claim is ever made about this project (§5).
4. **Port the equivalence-check step** before trusting any cross-framework
   comparison this project produces — the design note for it
   (`base_line implmentation.md`) already exists; it was never executed.
5. **Don't compare wall-clock/latency numbers across the two projects without
   normalizing for hardware** (§4.3) — this machine's RTX 5060 vs. Haseeb's
   Tesla T4 alone accounts for an 8.5× gap that has nothing to do with either
   codebase. Any future speed comparison needs same-GPU runs, or the
   comparison should stick to accuracy/spike-rate/memory, which aren't
   GPU-generation-sensitive the way latency is.
