# Project context for Claude

This file is committed to the repo (branch `scalability_tests`) so it travels
with `git clone`/`git pull` regardless of which machine or which Claude
account opens this project. Read it fully before making changes.

## What this project is

A comparison of SNN (Spiking Neural Network) framework backends — Norse,
SNNTorch, SpikingJelly (`sj`), Sinabs — for real-time suitability, efficiency,
and **scalability**, behind one shared `SpikingNet`/`BaseLIF` adapter
architecture. This branch (`scalability_tests`) is the scalability-study arm:
a ladder of experiments (`experiments/ex6`, `ex7`, ...) that vary network
width/depth and measure how cost and "capacity" metrics scale.

- Owner: Haseeb Ahmed (haseeb.ahmed@gameforge.com / git user "Haseeb").
- Collaborator: Muhammad Zuhair (zuhairmuhammad16@gmail.com), works on his own
  branch (`pipelines_merged`) — his history has diverged; do not assume his
  branch and this one are in sync. Coordination happens by hand-porting
  specific fixes, not merging wholesale (see "RAM leak" below for why).

**Start here:** `scalability_tests/experiment_plan_final.md` is the master plan —
what ex6 through ex10 are, why, and the current status of each. Read it
before proposing new experiment design.

## Architecture map

- `configuration/{SNN_module.yaml, network_architecture.yaml, data_workflow.yaml}`
  — merged by `skeleton/config_loader.py`. Experiment overlays
  (`experiments/exN/*.yaml`) use `extends:` chains, recursive, multi-level.
- `skeleton/results.py` — CSV schema (`RUN_COLUMNS`, `EPOCH_COLUMNS`,
  `LAYER_COLUMNS`, `SCHEMA_VERSION`). Bump `SCHEMA_VERSION` when columns
  change.
- `learning/training.py` (`SNNTrainer`) / `learning/inference.py`
  (`SNNTester`) — the training and test loops. Both follow a **deferred-sync
  design**: no `.item()`/`.cpu()` inside per-iteration loops; GPU-resident
  accumulation, synced back once per epoch (training) or once at the end
  (inference, gradient norms). This rule has caused real, hard-to-see bugs
  before (see below) — respect it when touching either file.
- `learning/capacity_metrics.py` — Participation Ratio, spike entropy,
  mutual information I(Z;Y), all accumulated over the **full test set** in
  `SNNTester.run()`, not per-batch. Per-channel, not per-pixel. Raw +
  normalized variants reported. See
  `docs/superpowers/specs/2026-09-04-capacity-metrics-design.md` (especially
  §6, "Revision 2") for the full rationale and the methodology corrections
  Zuhair requested.
- `frameworks/spiking_net.py` — `named_lif_layers()` is role-based (last LIF
  is always named `lif_out`), not positional. `ActivityMonitor` hooks every
  named layer dynamically.
- `event_data_workflow/` — the data pipeline. `cache_engine.py`'s
  `determine_dataset_strategy()` picks memory-cache vs. disk-cache based on
  `dataset_size_gb * num_workers` vs. available RAM/disk.
  `system_monitor.py`'s `worker_count()` clamps the configured worker count
  to the machine's physical core count — this differs per machine (1 core on
  Colab, 2 on default Kaggle), which caused a real bug (see below).

## Known, already-fixed bugs worth knowing about (don't reintroduce)

1. **The system RAM leak (fixed in `0a7037f3`).** `SNNTrainer.train()` used
   to accumulate a list (`raw_epoch_records`) holding every epoch's full
   `activity_snapshot` (per-layer spike tensors) and only process it in one
   bulk pass (`finalize_epoch_reports`) after the whole run finished. Every
   epoch's snapshot stayed resident in host RAM simultaneously, growing
   linearly for the entire run — confirmed on real Colab/Kaggle runs (>10GB
   growth by epoch 9 on Colab; steady ~1GB/epoch decline crashing at epoch 7
   on Kaggle), in **both** disk-cache and memory-cache modes, which is what
   proved the cache tier was never the cause. Fixed by processing and
   printing each epoch's report immediately inside the loop
   (`finalize_one_epoch_report`), so each record is freed right after use.
   The fix was found by reading Zuhair's `pipelines_merged` commit
   `092b80ea`, which made the identical change on his branch and said so in
   its own commit message. **If you ever see `raw_epoch_records`-style
   "accumulate everything, process once at the end" reappear for anything
   that holds real tensors (not scalars), that's this bug again.**
2. **`ActivityMonitor` not resumed before `SNNTester.run()`** (fixed) — capacity
   metrics were silently computed on zeroed activity. `run()` now
   resumes/clears the monitor and restores prior state in a `finally` block.
3. **Kaggle resource halt** (fixed via `experiments/ex7/config.yaml`'s
   `resource_policy.worker_count_fallback: 1`) — a 2-core Kaggle box doubles
   the "effective size" `determine_dataset_strategy()` checks against
   available RAM, versus a 1-core Colab box, so the *same* dataset picks a
   *different* cache tier per machine. Pinning workers to 1 makes the
   decision reproducible across machines. This is unrelated to the RAM leak
   above — keep both fixes.
4. **ex7's classifier-dominance caveat** (documented, not architecturally
   fixed) — on N-Caltech101, the final linear classifier holds 99.3-99.9% of
   total parameters across the f8-f64 ladder, because the flattened conv
   output feeding it is large. Width-scaling conclusions from ex7 should be
   read with this caveat; see `experiments/ex7/README.md` section 3.

## Working agreements (how Haseeb wants this project run)

- **Keep responses brief and precise.** No long walls of text.
- **Do not use popup-style multiple-choice UI for options** — present choices
  as plain chat text instead.
- When asked to "just explain, don't change code," take that literally —
  no edits until explicitly asked to implement.
- Git commit messages and PR descriptions end with the standard Claude Code
  attribution lines (commit: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`;
  PR body: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`).
- Prefer porting a specific fix by hand over merging a diverged branch
  wholesale when the overlapping file has moved on independently on both
  sides (see the RAM-leak fix above for why — a `git merge-tree` simulation
  is a safe way to check this before deciding).

## ex7 redesign, 2026-09-29 (read before touching the scalability ladder)

The width ladder was re-scoped after the study designer re-read the brief
(`scalability_tests/Scalability.pdf`). Four changes, all recorded in
`experiments/ex7/README.md` section 0:

1. **Goal is the sweet spot, not peak accuracy.** Two separate equilibrium
   points — the bottleneck side (capacity metrics plateau) and the fragility
   side (cost grows without return) — reported as two trade-off tables, not
   averaged into one number. Brief p.2, Strategy item 4.
2. **Rungs are now 8/16/32/64**, not 16/32/64/128. `f8.yaml` added, `f128.yaml`
   retired (kept on disk, out of the ladder).
3. **All four frameworks**, not SpikingJelly only — 4 rungs x 4 backends = 16
   runs. This absorbs ex10, which is now redundant as originally scoped.
   ex8/ex9 stay SpikingJelly-only. (Superseded 2026-10-04 for ex8 — see below.)
4. **Epochs fixed at 50** (was 15) — *reverted to 15 on 2026-09-30, see "Open /
   unresolved" below.* Measured: the first f16 run ended with train
   accuracy still climbing (28.6 -> 59.6% over 15 epochs, ~+1.5pp/epoch, loss
   still falling), so 15 was too short for every rung. The separate
   extended-epoch arm collapsed into this budget and is retired.

**Two known gaps in the instrumentation vs. the brief**, both flagged in
`experiments/ex7/README.md` section 9 and `experiment_plan_final.md` section 5:
- **I(X;Z) is not implemented** — and it is the brief's central bottleneck
  metric (p.3). We compute I(Z;Y) instead, deliberately (design doc section 6e).
  Needs an explicit decision before the write-up, not a silent omission.
- **SynOps is not in the cross-run schema** — `synops_energy_pj` lives only in
  each run's own `training_results.csv`, so the brief's "spikes rise without
  accuracy" diagnostic can't be read across rungs without extra aggregation.

**Also worth knowing when running:** `skeleton/results.py`'s `write_results()`
writes `runs.csv`/`epochs.csv`/`layers.csv` only AFTER training *and* inference
finish. An interrupted or timed-out run loses everything — there are no partial
results. On Kaggle that means committing (Save & Run All), not interactive
sessions, and projecting total runtime before starting a long rung.

## ex8 depth design, final as of 2026-10-04

Full design in `scalability_tests/experiment_plan_final.md` §6 ("ex8 design") and §9.
- **All four frameworks** (not SJ-only). **ex10 dropped** — ex7 + ex8 cover all four.
- **Width fixed at f12** (`conv1_out = conv2_out = 12`) for ex8/ex9; ex7's best accuracy
  was f8, f12 chosen between f8 and f16 (the original 12-filter size).
- **Depth = count of hidden FC layers, size fixed at 128.** Ladder d0/d1/d2/d4.
  At f12 the flatten feeding the FC part is 28,728; output is 101 (`lif_out`).
- **No 3-seed pass in ex7** (width fixed at f12, seeded via ex8's winning rung and ex9).
  ex7's rung ordering is single-seed — report it as a caveat.
- Stop rule (<1% acc for >10% time/params) applied afterwards, not as abort. Theoretical
  expectations in the plan are hypotheses, not acceptance criteria.
- **Built 2026-10-04:** `fc_hidden: {layers, size}` in `network_architecture.yaml`
  (default `layers: 0` = original net), hidden blocks in `frameworks/spiking_net.py`,
  neuron key `neuron_types.<fw>.lif_hidden`, `experiments/ex8/` (README + d0/d1/d2/d4,
  extends ex7's config). Run instructions: `experiments/ex8/README.md` §9.

## Open / unresolved as of 2026-10-03

- **Epoch budget reverted 15 -> 50 -> 15.** `experiments/ex7/config.yaml` briefly set
  `epochs: 50` (2026-09-29), then was reverted to 15 the next day (commit
  "ex7 epochs now 15") by deliberate decision — the goal is comparing width, not
  reaching peak accuracy, so every rung uses the same fixed early checkpoint rather
  than a converged one. Known, accepted trade-off: rungs/frameworks may differ in how
  far along their own convergence curve they are at epoch 15, mixing "effect of width"
  with "effect of convergence speed." See the config.yaml comment at that line for the
  full note. **Stale reference:** the "ex7 redesign" section above still says "Epochs
  fixed at 50" (item 4) — that was true 2026-09-29, no longer true as of the revert.
- **Cross-framework convergence-rate check — not yet done, do before trusting the
  4-framework comparison.** A standalone SpikingJelly calibration run (f16, 50 epochs,
  interrupted at epoch 47 but complete through 46 — logged in
  `experiments/ex7/calibration_runs/f16_50ep_20260930_180757_incomplete/`) found f16's
  train accuracy still climbing ~0.8pp/epoch at epoch 15, plateauing around epoch
  25-30. That calibration was SpikingJelly-only. The other three frameworks
  (snnTorch/Norse/Sinabs) may converge at different rates than SJ even on an identical
  config — same confound as the width axis, just across frameworks instead. Cheap
  first check once each framework's normal 15-epoch f16 run is in: compare their
  `train_accuracy_pct` at epoch 15 against SJ's 53.9%. Close and still climbing at a
  similar rate -> SJ's calibration likely transfers, skip further checks for that
  framework. Very different (much lower and barely moving, or already flat) -> that
  framework needs its own extended (~50-epoch) calibration run before its 15-epoch
  numbers are compared against the others. Don't commit to 3 blanket extended runs
  without checking this first — expensive and the normal runs already carry the signal
  needed to decide which frameworks actually need one.

- `PR_Entropy_Sample_Size_Question.md` (repo root) is **untracked** — a short
  doc for Zuhair about the PR/entropy sample-size ceiling. Decide whether to
  commit it or keep it local-only.
- The RAM-leak fix (`0a7037f3`) was verified by `py_compile` and manual
  read-through, but the `pytest tests/unit_training_metrics.py` confirmation
  run was interrupted by a session end and never confirmed to pass. Rerun it
  before fully trusting the fix.
- Zuhair's `pipelines_merged` branch still has 2 more commits not ported
  here: a `numpy==1.26.4` → `numpy==2.3.5` bump, and a `output.save_checkpoint`
  feature (`frameworks/model_interface.py`'s `load_state()`,
  `learning/main.py` checkpoint-save block). Neither has been brought over —
  only the RAM-leak fix was.
- `CHAT_RECAP_2026-09-21.md` (repo root) has a fuller narrative of the RAM-leak
  investigation if more detail is needed than this file gives.
