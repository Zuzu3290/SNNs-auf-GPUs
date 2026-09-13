# Cleanup before final commit / publication

Working list of files to remove before the final commit. Nothing here has been
deleted yet — this is only the reminder list, to action at the very end. More items
will be added over time.

## Confirmed for removal

- `story b.md`, `story_c.md`, `story_d.md` (repo root) — working/draft notes
- `event_data_workflow/Async_CPU_GPU_Pipeline.md`
- `plots/` folder (`__init__.py`, `data.py`, `figures.py`, `primitives.py`,
  `style.py` — the F0-F6 comparison figure code). `make_plots.py` is the only other
  file importing from it (`from plots import ...`), so that import breaks the moment
  this is deleted — either remove/rewrite `make_plots.py` alongside it, or confirm
  whether `make_plots.py` itself is also going before the final commit.

## To review (candidates, not yet decided)

Everything under `docs/` — internal working notes accumulated during development,
likely not meant for the final publication. Needs a pass to decide what (if anything)
is worth keeping as real documentation vs. removing:

- `docs/Event-Based_camera.md`
- `docs/event_data_workflow/` (architecture_reference.md, caching_pipeline_refactor.md,
  Dataset_workflow.md, Framing_&_TensorShape.md, monitoring_consolidation_session.md,
  pipeline_coordinator.md, Pipeline_Monitor_Fix.md, session_log.md)
- `docs/frameworks/` (additional_frameworks.md, config_wiring.md, README.md,
  realtime_nir_evaluation.md, researc.md, session_log.md)
- `docs/framework_comparison_report.md`
- `docs/functions.md`
- `docs/Hardware/` (event_data_workflow.md, resources.md)
- `docs/Haseeb-open-items.md`
- `docs/known-issue-sinabs-dead-spikes.md`
- `docs/learning_changes.md`
- `docs/results/README.md`
- `docs/vram_batch_scaling_task.md`
- `docs/v_model_diagnosis.md`

Other root-level working/planning docs not yet triaged:

- `Hardcoded_Values_Review.md`
- `Presentation_Slides_EventDataWorkflow.md`
- `Scalability_RealTime_Enhancement_Plan.md`
- `System_Boundaries_and_Tuning.md`
- `scalability.md`
- `SNN_GPU_Evaluation_Metrics.md`

## Needs a decision, not a quick delete

- `experiments/ex2/` — flagged as "an old thing" to possibly remove, but this is NOT
  dead content: `tests/unit_cli_config.py` (lines ~184-269) reads
  `experiments/ex2/config.yaml` directly as a fixture and asserts the whole Flow B
  output-routing layout (`results/`, `plots/`, `figures/`, `equivalence/`) against it.
  It's also the only worked example of `--experiment`/`--results-root`, which
  `HOW_TO_RUN.md` documents as Flow B ("what Colab and multi-seed sweeps need") and
  which 11 files wire into (`learning/main.py`, `equivalence_check.py`,
  `check_network.py`, `make_plots.py`, `skeleton/cli.py`, `skeleton/config_loader.py`,
  `skeleton/snn_config.py`, plus `tests/unit_cli_config.py` and
  `tests/unit_entrypoints.py`). Deleting it outright breaks 173 currently-passing
  tests. Options to weigh at the final pass:
  1. Leave it alone — it's the Flow B fixture, not stale.
  2. Drop Flow B entirely (`--experiment`/`--results-root` flags, `experiments/`, and
     the test coverage for both) if multi-seed/Colab routing via ex2-style folders is
     genuinely not how runs will happen going forward.
  3. Reset just ex2's example content (its config/README/generated results) while
     keeping the `--experiment` mechanism supported — needs a replacement or
     regenerated test fixture either way.

- `skeleton/seeding.py` (the multi-seed reproducibility/variance-testing experiment) —
  flagged for removal. Still unresolved from an earlier question, so flagging rather
  than guessing: this file does TWO distinct things, and it matters which is meant.
  1. Cross-framework weight-init seeding (`seed_model_init`, `weight_fingerprint`,
     `verify_cross_framework_init`) — guarantees snnTorch/norse/spikingjelly/sinabs
     start from byte-identical weights for a given seed. This is what makes "same
     seed, different framework" an isolated comparison rather than partly a comparison
     of random initial weights. `tests/unit_seeding.py` (71 tests, currently passing)
     covers this.
  2. Multi-seed variance/reproducibility runs (repeating one config across several
     seeds to check run-to-run spread) — a specific EXPERIMENT built on top of (1),
     not the weight-init guarantee itself.
  "seeding experiment" most likely means (2) only, but confirm before touching this
  file — removing (1) changes what a cross-framework comparison in this project can
  actually claim, not just what it measures.

- `tests/` (the whole folder: `run_all.py` + all 9 `unit_*.py` suites) — flagged for
  removal. Recording my pushback here rather than silently dropping it: this is the
  only automated check that a change didn't break the pipeline. It's what verified
  every edit made during this session's requirements.txt/tonic/check_env.py rework and
  the skeleton/ cleanup (snn_logging.py, etc.) — 9/9 suites, ~1000+ individual checks,
  each time. Once this is gone, "did that change break anything" has no automated
  answer; someone has to notice by hand or in a real run. Worth deciding deliberately
  at the final pass: keep it for CI/collaborator confidence, trim it down, or genuinely
  remove it because it no longer matches how this project will be validated going
  forward (e.g. if manual/notebook-driven checks replace it).

## Confirmed structural changes (relocation, not deletion)

- `configuration/` folder to be deleted, but NOT its contents: `SNN_module.yaml`,
  `network_architecture.yaml`, `data_workflow.yaml` (and `README.md`) move to the repo
  root instead. Needs a real code change, not just a file move:
  `skeleton/config_loader.py` hardcodes `CONFIG_DIR = .../  "configuration"` and the
  `BASE_FILES` dict's three filenames — that path needs updating to the new root
  location, and anything else that references `configuration/<file>.yaml` by path
  (grep before doing this) needs to move with it.

## Needs a thorough pass (not a removal — a rewrite)

- `HOW_TO_RUN.md` — needs a full review once the requirements.txt/tonic setup settles.
  Known issues right now:
  - Line ~115 points readers to requirements.txt's "Event data" section for why tonic
    is installed separately — that section no longer exists (all comments were
    stripped from requirements.txt). Dangling reference.
  - The `python check_env.py` step in "Installing from scratch" only helps someone who
    gets past `pip install -r requirements.txt` — a direct-Python user who skips the
    `pip install --no-deps tonic==1.6.0` pre-step hits a raw pip `ResolutionImpossible`
    crash before check_env.py ever runs. Docker users don't face this (the Dockerfile
    bakes the order in), so it's a real asymmetry worth resolving deliberately rather
    than leaving as an unstated assumption.

## Keep

- `README.md`
