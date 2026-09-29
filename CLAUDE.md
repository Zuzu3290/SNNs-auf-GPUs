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

**Start here:** `experiment_plan_final.md` at repo root is the master plan —
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
   fixed) — on N-Caltech101, the final linear classifier holds 98.7-99.8% of
   total parameters regardless of conv filter count, because the flattened
   conv output feeding it is large. Width-scaling conclusions from ex7 should
   be read with this caveat; see `experiments/ex7/README.md`.

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

## Open / unresolved as of 2026-09-29

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
