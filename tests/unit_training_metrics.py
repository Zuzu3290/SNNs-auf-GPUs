"""Unit tests for the training-loop corrections: warm-up, and the iteration cap.

    python tests/unit_training_metrics.py

Two fixes are covered, and each one was a number that came out wrong rather than a
crash -- which is why they need tests rather than a glance:

  D7   no warm-up, so CUDA kernel compilation was charged to training, and in a
       multi-framework run charged only to whichever framework ran first
  D12  the iteration cap was checked after the batch had already been trained on, so an
       epoch ran num_iters + 1 iterations

CPU-only, no dataset, no download.
"""
from __future__ import annotations

import pathlib

import torch

from _harness import FRAMEWORKS, Suite, build_model, fresh_cfg, spike_input
from learning.utilities import (
    spikes_per_neuron_per_inference, warm_up,
)
from skeleton.config_loader import deep_merge, load_base
from skeleton.snn_config import Settings
from skeleton.workflow_config import WorkflowSettings

import tempfile

suite = Suite("unit_training_metrics")
REPO_TMP = pathlib.Path(tempfile.mkdtemp(prefix="snn_unit_"))


def settings_pair(overrides: dict):
    merged = deep_merge(load_base(), overrides)
    return Settings(config=merged), WorkflowSettings(config=merged)


# ---------------------------------------------------------------------------
# 1. phantom-attribute regression guard
# ---------------------------------------------------------------------------
def test_the_attribute_the_old_code_read_does_not_exist() -> None:
    """The root cause, asserted so it cannot quietly come back."""
    cfg, wf = settings_pair({})
    suite.check("cfg has no TEMPORAL_SLICE_DURATION_US",
                not hasattr(cfg, "TEMPORAL_SLICE_DURATION_US"))
    suite.check("wf has no TEMPORAL_SLICE_DURATION_US",
                not hasattr(wf, "TEMPORAL_SLICE_DURATION_US"))
    suite.check("cfg has no TIMESTEPS", not hasattr(cfg, "TIMESTEPS"))
    # The slice duration now lives with the rest of the slicing config, on
    # WorkflowSettings, not on Settings.
    suite.check("the slice duration lives on wf", hasattr(wf, "SLICE_DURATION_US"))
    suite.check("and no longer on cfg", not hasattr(cfg, "TEMPORAL_SLICE_DURATION"))


def test_spikes_per_inference_is_time_unit_free() -> None:
    suite.check("rate x T", spikes_per_neuron_per_inference(0.05, 16) == 0.8)
    suite.check("a silent layer gives 0", spikes_per_neuron_per_inference(0.0, 16) == 0.0)
    suite.check("scales with T", spikes_per_neuron_per_inference(0.05, 32) == 1.6)


# ---------------------------------------------------------------------------
# 2. D7 -- warm-up
# ---------------------------------------------------------------------------
def test_warmup_leaves_the_weights_untouched() -> None:
    """The whole point: it compiles kernels without training. If the weights moved,
    the timed epochs would not start from the seeded state."""
    from skeleton.seeding import shared_weight_fingerprint

    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        before = shared_weight_fingerprint(model)
        report = warm_up(model, spike_input(cfg, 6, 2), 3)
        suite.check(f"warm-up reports weights unchanged: {framework}",
                    report["weights_unchanged"])
        suite.check(f"the fingerprint really is unchanged: {framework}",
                    shared_weight_fingerprint(model) == before)


def test_warmup_clears_gradients() -> None:
    """It runs backward, so gradients exist. Left behind, the first real step would
    apply warm-up gradients on top of the first batch's."""
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        warm_up(model, spike_input(cfg, 6, 2), 2)
        leftover = [n for n, p in model.named_parameters()
                    if p.requires_grad and p.grad is not None]
        suite.check(f"no gradients survive warm-up: {framework}", not leftover,
                    f"{len(leftover)} left")


def test_warmup_leaves_no_activity_recordings() -> None:
    """Warm-up spikes must not reach the metrics."""
    for framework in FRAMEWORKS:
        model, cfg = build_model(framework)
        warm_up(model, spike_input(cfg, 6, 2), 2)
        suite.check(f"activity buffers are empty after warm-up: {framework}",
                    all(v is None for v in model.activity.recordings().values()))


def test_warmup_zero_is_a_no_op() -> None:
    model, cfg = build_model("torch")
    report = warm_up(model, spike_input(cfg, 6, 2), 0)
    suite.check("0 iterations does nothing", report["iterations"] == 0)
    suite.check("and reports no change", report["weights_unchanged"] is True)


def test_warmup_actually_runs_the_requested_count() -> None:
    model, cfg = build_model("norse")
    suite.check("the reported count matches what was asked",
                warm_up(model, spike_input(cfg, 6, 2), 4)["iterations"] == 4)


def test_warmup_iterations_is_configurable() -> None:
    cfg = Settings()
    suite.check("warmup_iterations is in the shipped config",
                "warmup_iterations" in load_base()["training"])
    suite.check("Settings exposes it", hasattr(cfg, "WARMUP_ITERATIONS"))
    suite.check("it defaults to a non-zero count", cfg.WARMUP_ITERATIONS > 0,
                str(cfg.WARMUP_ITERATIONS))
    cfg2, _ = settings_pair({"training": {"warmup_iterations": 0}})
    suite.check("it can be switched off", cfg2.WARMUP_ITERATIONS == 0)


# ---------------------------------------------------------------------------
# 3. D12 -- the iteration cap
# ---------------------------------------------------------------------------
def cap_iterations(n_batches: int, num_iters: int) -> int:
    """The loop shape now used in SNNTrainer.train(): checked BEFORE the work."""
    ran = 0
    for i in range(n_batches):
        if i >= num_iters:
            break
        ran += 1
    return ran


def test_iteration_cap_runs_exactly_what_was_asked() -> None:
    for n_batches, num_iters in [(400, 3), (400, 10), (400, 100), (50, 49)]:
        expected = min(n_batches, num_iters)
        suite.check(f"loader {n_batches}, asked {num_iters} -> runs {expected}",
                    cap_iterations(n_batches, num_iters) == expected,
                    str(cap_iterations(n_batches, num_iters)))


def test_cap_above_the_loader_length_is_harmless() -> None:
    suite.check("a cap larger than the loader just exhausts it",
                cap_iterations(5, 400) == 5)


def test_the_calibrated_case_is_unaffected() -> None:
    """Why the bug was invisible in normal use: with calibrate_batch_size on,
    ITERA == len(loader), so the loader runs out before the cap is ever reached."""
    suite.check("ITERA == loader length runs the whole loader",
                cap_iterations(400, 400) == 400)


def test_the_old_shape_overran_by_one() -> None:
    """Kept as a regression witness -- this is what the code used to do."""
    def old(n_batches, num_iters):
        ran = 0
        for i in range(n_batches):
            ran += 1
            if i == num_iters:
                break
        return ran

    suite.check("the old shape ran one extra iteration", old(400, 3) == 4, str(old(400, 3)))
    suite.check("a 33% overrun on a 3-iteration diagnostic",
                old(400, 3) / 3 > 1.3)
    suite.check("the new shape does not", cap_iterations(400, 3) == 3)


# ---------------------------------------------------------------------------
# 4. the trainer wires all three in
# ---------------------------------------------------------------------------
def test_trainer_exposes_the_new_pieces() -> None:
    import inspect

    from learning import training

    source = inspect.getsource(training.SNNTrainer.train)
    suite.check("train() calls warm_up", "warm_up(" in source)
    suite.check("train() checks the cap before the body",
                "if i >= num_iters:" in source)
    suite.check("the old trailing check is gone", "if i == num_iters:" not in source)
    suite.check("the phantom attribute is not read anywhere in training",
                "getattr(self.cfg, 'TEMPORAL_SLICE_DURATION_US'" not in inspect.getsource(training))


def test_inference_no_longer_reads_phantom_attributes() -> None:
    import inspect

    from learning import inference

    source = inspect.getsource(inference)
    # Matches the READ, not the name -- the name still appears in the comment that
    # explains what went wrong, and that comment is worth keeping.
    suite.check("inference does not read TEMPORAL_SLICE_DURATION_US",
                "getattr(self.cfg, 'TEMPORAL_SLICE_DURATION_US'" not in source)
    suite.check("inference does not read cfg.TIMESTEPS",
                "getattr(self.cfg, 'TIMESTEPS'" not in source)
    suite.check("inference reports the time-unit-free figure too",
                "spikes_per_neuron_per_inference" in source)


def test_batch_size_is_recorded(  ) -> None:
    """D11: calibration legitimately picks a different batch size per machine, so the
    value has to travel with the results or two rows cannot be compared on speed."""
    import inspect

    from learning import training

    source = inspect.getsource(training.SNNTrainer.finalize_one_epoch_report)
    suite.check("the epoch row records batch_size", '"batch_size"' in source)
    suite.check("and whether it was calibrated", '"batch_size_calibrated"' in source)


# ---------------------------------------------------------------------------
# 5. spike recording is outside every timed region
# ---------------------------------------------------------------------------
class FakeLoader:
    """Stands in for PrefetchedLoader: yields already-device-resident batches."""

    def __init__(self, batches: int, cfg, timesteps: int = 6, batch: int = 2):
        self.batches, self.cfg, self.T, self.B = batches, cfg, timesteps, batch

    def __len__(self):
        return self.batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(0)
        shape = (self.T, self.B, self.cfg.IN_CHANNELS, self.cfg.SENSOR_H, self.cfg.SENSOR_W)
        for _ in range(self.batches):
            yield ((torch.rand(*shape, generator=generator) < 0.5).float(),
                   torch.randint(0, self.cfg.NUM_CLASSES, (self.B,), generator=generator))


def run_short_training(iterations: int = 3, epochs: int = 2, available: int = 10):
    """A real SNNTrainer.train() on CPU with a fake loader. Slower than the rest of
    this suite, but the iteration cap and the activity/timing separation are properties
    of the LOOP -- testing the pieces in isolation would not show them."""
    from learning.training import SNNTrainer

    cfg = fresh_cfg()
    cfg.EPOCHS, cfg.ITERA, cfg.BATCH_SIZE = epochs, iterations, 2
    cfg.ENABLE_PIPELINE_MONITOR = False
    model, _ = build_model("torch", cfg)
    trainer = SNNTrainer(model, FakeLoader(available, cfg), cfg, torch.device("cpu"))
    trainer.train(csv_path=str(REPO_TMP / "unit_training_results.csv"))
    return trainer, model


def test_training_runs_exactly_the_requested_iterations() -> None:
    """D12 end to end: 10 batches available, 3 requested, so every epoch must log 3."""
    trainer, _ = run_short_training(iterations=3, epochs=2, available=10)
    counts = [row["n"] for row in trainer.epoch_log]
    suite.check("every epoch ran exactly the requested iterations", counts == [3, 3],
                str(counts))


def test_recording_is_paused_during_training() -> None:
    trainer, model = run_short_training(iterations=2, epochs=1, available=5)
    suite.check("the monitor is left paused after training", model.activity.paused is True)
    suite.check("no recordings are left dangling",
                all(v is None for v in model.activity.recordings().values()))


def test_activity_metrics_still_land_despite_the_pause() -> None:
    """Pausing must not silently cost the metrics -- that is what the untimed pass is
    for. SynOps and CV_ISI both have to be real, non-zero numbers."""
    trainer, _ = run_short_training(iterations=2, epochs=2, available=5)
    synops = [row["synops_energy_pj"] for row in trainer.epoch_log]
    cv_isi = [row["cv_isi_mean"] for row in trainer.epoch_log]
    suite.check("SynOps is recorded for every epoch", all(v > 0 for v in synops), str(synops))
    suite.check("CV_ISI is recorded for every epoch", all(v > 0 for v in cv_isi), str(cv_isi))


def test_synops_is_per_batch_not_an_iteration_sum() -> None:
    """It used to be summed over every iteration, so it scaled with iteration count and
    two runs of different length could not be compared. Sampling once per epoch means
    doubling the iterations must NOT roughly double it."""
    few, _ = run_short_training(iterations=2, epochs=1, available=10)
    many, _ = run_short_training(iterations=8, epochs=1, available=10)
    a, b = few.epoch_log[0]["synops_energy_pj"], many.epoch_log[0]["synops_energy_pj"]
    suite.check("4x the iterations does not scale SynOps", b < a * 2.0,
                f"2 iters {a:.0f}, 8 iters {b:.0f}")


def test_iteration_series_lengths_line_up() -> None:
    """Every per-iteration series must be the same length as the loss history, or the
    CSV writer's zip() would silently truncate the file."""
    trainer, _ = run_short_training(iterations=3, epochs=2, available=10)
    series = trainer.iteration_series()
    expected = len(trainer.loss_hist)
    suite.check("loss history has one entry per iteration", expected == 6, str(expected))
    for key in ["synops_energy_pj", "gpu_energy_j", "learning_rate",
                "spikes_per_neuron_per_inference"]:
        suite.check(f"{key} matches the loss history length",
                    len(series[key]) == expected, f"{len(series[key])} vs {expected}")
    suite.check("vram history matches too", len(trainer.vram_current_hist) == expected)


def test_epoch_row_carries_the_new_columns() -> None:
    trainer, _ = run_short_training(iterations=2, epochs=1, available=5)
    row = trainer.epoch_log[0]
    for column in ["energy_j_total", "energy_j_dynamic", "idle_power_w",
                   "spikes_per_neuron_per_inference",
                   "batch_size", "batch_size_calibrated", "synops_energy_pj"]:
        suite.check(f"epoch row has {column}", column in row)


def test_training_runs_on_cpu_at_all() -> None:
    """There was an unguarded torch.cuda.synchronize() at the epoch boundary, so a
    CPU-only torch build died at the end of epoch 1 -- which is every laptop run."""
    try:
        run_short_training(iterations=2, epochs=2, available=5)
        suite.check("a two-epoch CPU run completes", True)
    except AssertionError as error:  # "Torch not compiled with CUDA enabled"
        suite.check("a two-epoch CPU run completes", False, str(error))


# ---------------------------------------------------------------------------
# 6. single-stream (batch-size-1) latency
# ---------------------------------------------------------------------------
class _OneBatchLoader:
    def __init__(self, cfg, batches=3, timesteps=6, batch=8):
        self.cfg, self.n, self.T, self.B = cfg, batches, timesteps, batch

    def __iter__(self):
        for _ in range(self.n):
            yield (spike_input(self.cfg, self.T, self.B),
                   torch.zeros(self.B, dtype=torch.long))


def test_collect_single_samples_gives_batch_of_one() -> None:
    from learning.utilities import collect_single_samples

    _, cfg = build_model("torch")
    samples = collect_single_samples(_OneBatchLoader(cfg), torch.device("cpu"), 20)
    suite.check("collects the requested count", len(samples) == 20, str(len(samples)))
    suite.check("each sample has batch dimension 1", all(s.shape[1] == 1 for s in samples),
                str(tuple(samples[0].shape)))
    suite.check("time dimension preserved", samples[0].shape[0] == 6,
                str(tuple(samples[0].shape)))


def test_measure_latency_is_a_real_per_sample_measurement() -> None:
    """The distinction that matters: dividing a batch time by B gives every sample in
    that batch the SAME number, so its percentiles describe batch-to-batch variation.
    A real bs1 pass produces genuinely different values per sample."""
    from learning.utilities import collect_single_samples, measure_latency

    model, cfg = build_model("torch")
    samples = collect_single_samples(_OneBatchLoader(cfg), torch.device("cpu"), 15)
    result = measure_latency(model, samples, torch.device("cpu"), warmup=2)

    for key in ["latency_ms", "latency_mean_ms", "latency_p90_ms",
                "latency_min_ms", "latency_max_ms", "latency_samples"]:
        suite.check(f"reports {key}", key in result)
    suite.check("one timing per sample", result["latency_samples"] == 15,
                str(result["latency_samples"]))
    suite.check("timings are positive", result["latency_ms"] > 0)
    suite.check("min <= median <= p90 <= max",
                result["latency_min_ms"] <= result["latency_ms"] <= result["latency_p90_ms"]
                <= result["latency_max_ms"])
    suite.check("the samples are not all identical -- a real spread was measured",
                result["latency_max_ms"] > result["latency_min_ms"])


def test_measure_latency_refuses_an_empty_sample_list() -> None:
    from learning.utilities import measure_latency

    model, _ = build_model("torch")
    suite.expect_raises("empty sample list raises", ValueError,
                        lambda: measure_latency(model, [], torch.device("cpu")))


def test_run_row_bs1_columns_come_from_the_real_measurement() -> None:
    """Same column names as SNNs_2, and now the same measurement behind them."""
    import inspect

    from skeleton import results_collect

    source = inspect.getsource(results_collect.build_run_row)
    suite.check("bs1 columns read the latency dict",
                '(latency or {}).get("latency_ms")' in source)
    suite.check("they no longer read the amortised per-sample figure",
                "median_latency_per_sample_ms" not in source)
    suite.check("build_run_row accepts a latency argument",
                "latency" in inspect.signature(results_collect.build_run_row).parameters)


def test_latency_samples_is_configurable() -> None:
    cfg = Settings()
    suite.check("latency_samples is in the shipped config",
                "latency_samples" in load_base()["training"])
    suite.check("Settings exposes it", hasattr(cfg, "LATENCY_SAMPLES"))
    suite.check("it defaults to a usable count", cfg.LATENCY_SAMPLES >= 20,
                str(cfg.LATENCY_SAMPLES))
    cfg2, _ = settings_pair({"training": {"latency_samples": 0}})
    suite.check("0 disables the pass", cfg2.LATENCY_SAMPLES == 0)


# ---------------------------------------------------------------------------
# 7. every artefact follows --results-root
# ---------------------------------------------------------------------------
def test_training_writes_every_csv_beside_the_given_path() -> None:
    """A real Colab run put training_results.csv on mounted Drive but
    batch_metrics.csv in ./outputs/data -- so that one file stayed on the runtime and
    died with it. Nothing may keep its module default when a path was given."""
    from learning.training import SNNTrainer

    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "ex9" / "results"
        cfg = fresh_cfg()
        cfg.EPOCHS, cfg.ITERA, cfg.BATCH_SIZE = 1, 2, 2
        cfg.ENABLE_PIPELINE_MONITOR = False
        model, _ = build_model("torch", cfg)
        trainer = SNNTrainer(model, FakeLoader(4, cfg), cfg, torch.device("cpu"))
        trainer.train(csv_path=str(target / "training_results.csv"))

        written = sorted(p.name for p in target.glob("*.csv"))
        suite.check("training_results.csv is routed", "training_results.csv" in written,
                    str(written))
        suite.check("batch_metrics.csv is routed too", "batch_metrics.csv" in written,
                    str(written))

        # And nothing leaked to the module default.
        stray = pathlib.Path("outputs/data/batch_metrics.csv")
        before = stray.stat().st_mtime if stray.exists() else None
        trainer.train(csv_path=str(target / "training_results.csv"))
        after = stray.stat().st_mtime if stray.exists() else None
        suite.check("nothing was written to the ./outputs default", before == after,
                    "outputs/data/batch_metrics.csv was touched")


def test_per_run_folder_keeps_two_frameworks_apart() -> None:
    """The collision this layout exists to prevent. training_results.csv,
    batch_metrics.csv and the seven diagnostic PNGs carry no framework or seed in their
    names, so two runs writing into one experiment folder would leave only the last.
    Simulated here with the same paths main.py builds."""
    from learning.training import SNNTrainer
    from skeleton.results import make_run_id

    with tempfile.TemporaryDirectory() as tmp:
        experiment = pathlib.Path(tmp) / "ex9"
        seen = []
        for framework in ("torch", "norse"):
            cfg = fresh_cfg()
            cfg.FRAMEWORK = framework
            cfg.EPOCHS, cfg.ITERA, cfg.BATCH_SIZE = 1, 2, 2
            cfg.ENABLE_PIPELINE_MONITOR = False
            run_id = make_run_id(cfg.FRAMEWORK, cfg.SEED)
            run_results = experiment / "results" / run_id
            run_plots = experiment / "plots" / run_id
            run_results.mkdir(parents=True, exist_ok=True)
            run_plots.mkdir(parents=True, exist_ok=True)

            model, _ = build_model(framework, cfg)
            trainer = SNNTrainer(model, FakeLoader(4, cfg), cfg, torch.device("cpu"))
            trainer.train(csv_path=str(run_results / "training_results.csv"))
            trainer.plot_training(save_dir=str(run_plots))
            seen.append((run_id, run_results, run_plots))

        ids = [r for r, _, _ in seen]
        suite.check("the two runs got different run_ids", ids[0] != ids[1], str(ids))
        for run_id, run_results, run_plots in seen:
            suite.check(f"{run_id}: its own training_results.csv survives",
                        (run_results / "training_results.csv").is_file())
            suite.check(f"{run_id}: its own batch_metrics.csv survives",
                        (run_results / "batch_metrics.csv").is_file())
            suite.check(f"{run_id}: its own plot survives",
                        (run_plots / "training_metrics.png").is_file())

        suite.check("nothing was written flat into results/",
                    not list((experiment / "results").glob("*.csv")),
                    str([p.name for p in (experiment / "results").glob("*.csv")]))


def test_run_id_ties_the_folder_to_the_results_row() -> None:
    """main.py must reuse ONE run_id for the folder and the runs.csv row -- generating
    a second one would break the link between a figure and the row describing it."""
    import inspect

    from learning import main as main_module

    source = inspect.getsource(main_module)
    suite.check("run_id generated once", source.count("make_run_id(") == 1)
    suite.check("the run folder uses it", "results_dir / run_id" in source)
    suite.check("the plots folder uses it", "plots_dir / run_id" in source)
    suite.check("the results row is given the same one", "run_id=run_id" in source)
    suite.check("nesting only happens when routed", 'run_info["routed"]' in source)


def test_adversarial_evaluator_is_routed_in_main() -> None:
    import inspect

    from learning import main as main_module

    source = inspect.getsource(main_module)
    suite.check("evaluate() is given an explicit csv_path",
                "evaluate(csv_path=" in source)


def test_training_memory_peaks_reach_runs_csv() -> None:
    """Both were hardcoded None with a comment saying only the inference phase is
    measured. That stopped being true once the per-epoch GPU report was added: the
    numbers were in epoch_log and printed every epoch, but never reached runs.csv."""
    from skeleton.results_collect import _peak_over_epochs

    # Shaped like the real thing: NVML usage flat, allocator high-water mark climbing.
    log = [{"gpu_mem_peak_gb": 2.89, "max_memory_reserved_gb": 5.05 + 1.48 * i}
           for i in range(5)]
    used = _peak_over_epochs(log, "gpu_mem_peak_gb")
    reserved = _peak_over_epochs(log, "max_memory_reserved_gb")
    suite.check("in-use peak is the max epoch, in MB", abs(used - 2.89 * 1024) < 0.01,
                str(used))
    suite.check("reserved peak is the max epoch, in MB",
                abs(reserved - (5.05 + 1.48 * 4) * 1024) < 0.01, str(reserved))
    suite.check("reserved is the larger of the two -- it includes memory held unused",
                reserved > used)

    suite.check("no epochs means no figure, not a zero",
                _peak_over_epochs([], "gpu_mem_peak_gb") is None)
    suite.check("a missing key means no figure",
                _peak_over_epochs([{"other": 1.0}], "gpu_mem_peak_gb") is None)
    suite.check("a non-numeric value is skipped rather than crashing the row",
                _peak_over_epochs([{"gpu_mem_peak_gb": "n/a"}], "gpu_mem_peak_gb") is None)
    suite.check("a CPU-only run reports 0.0, which is true, not missing",
                _peak_over_epochs([{"gpu_mem_peak_gb": 0.0}], "gpu_mem_peak_gb") == 0.0)


def test_runs_csv_records_the_timesteps_that_ran() -> None:
    """time_steps prefers the MEASURED value and falls back to the config. The measured
    one was computed in SNNTester and then dropped, so the fallback always won -- and a
    run framed at 16 was recorded as 20. The column that should have caught that bug was
    reading from the same place the bug was in."""
    import inspect

    import learning.inference as inference

    source = inspect.getsource(inference.SNNTester)
    suite.check("SNNTester puts its measured timesteps in the summary it returns",
                '"timesteps":                 mean_timesteps' in source
                or '"timesteps": mean_timesteps' in source)

    from skeleton.results_collect import build_run_row
    signature = inspect.signature(build_run_row)
    suite.check("build_run_row still takes an explicit timesteps argument",
                "timesteps" in signature.parameters)


def main() -> int:
    return suite.run([
        test_training_writes_every_csv_beside_the_given_path,
        test_per_run_folder_keeps_two_frameworks_apart,
        test_run_id_ties_the_folder_to_the_results_row,
        test_adversarial_evaluator_is_routed_in_main,
        test_collect_single_samples_gives_batch_of_one,
        test_measure_latency_is_a_real_per_sample_measurement,
        test_measure_latency_refuses_an_empty_sample_list,
        test_run_row_bs1_columns_come_from_the_real_measurement,
        test_latency_samples_is_configurable,
        test_the_attribute_the_old_code_read_does_not_exist,
        test_spikes_per_inference_is_time_unit_free,
        test_warmup_leaves_the_weights_untouched,
        test_warmup_clears_gradients,
        test_warmup_leaves_no_activity_recordings,
        test_warmup_zero_is_a_no_op,
        test_warmup_actually_runs_the_requested_count,
        test_warmup_iterations_is_configurable,
        test_iteration_cap_runs_exactly_what_was_asked,
        test_cap_above_the_loader_length_is_harmless,
        test_the_calibrated_case_is_unaffected,
        test_the_old_shape_overran_by_one,
        test_trainer_exposes_the_new_pieces,
        test_inference_no_longer_reads_phantom_attributes,
        test_batch_size_is_recorded,
        test_training_runs_exactly_the_requested_iterations,
        test_recording_is_paused_during_training,
        test_activity_metrics_still_land_despite_the_pause,
        test_synops_is_per_batch_not_an_iteration_sum,
        test_iteration_series_lengths_line_up,
        test_epoch_row_carries_the_new_columns,
        test_training_runs_on_cpu_at_all,
        test_training_memory_peaks_reach_runs_csv,
        test_runs_csv_records_the_timesteps_that_ran,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
